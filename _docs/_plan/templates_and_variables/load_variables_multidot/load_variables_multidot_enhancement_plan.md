# `TemplateManager.load_variables` Multi-Dot Support — Enhancement Plan

**Author:** Tony Chen (with Rovo Dev assistance)
**Date drafted:** 2026-05-17 00:33
**Status:** Ready for review and implementation
**Scope:**
- `CoreProjects/RichPythonUtils/src/rich_python_utils/string_utils/formatting/template_manager/template_manager.py` — core fix to `load_variables`
- `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/templated_inferencer_base.py` — REMOVE the `_inject_mode_flags_and_content` workaround
- `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_behavior_variable_injection.py` — existing 156-line test (needs to GREEN after fix)
- New tests in both RichPythonUtils and AgentFoundation that pin the multi-dot contract permanently

> **Why this plan exists.** A bug in `TemplateManager.load_variables` (line 716 splits on FIRST dot only) caused `_inject_mode_flags_and_content` to be added as a workaround in `templated_inferencer_base.py`. That workaround now blocks the cleaner pattern of `_variables/instructions/behavior/file_reading_fallback.jinja2` accessed via `{{ instructions.behavior.file_reading_fallback }}`. Fix it at the source, remove the workaround, no hack.

---

## 1. The bug — precise diagnosis

### 1.1 What the user wants to work

A template file like `plan/main/initial.jinja2` should be able to write:
```jinja2
- {{ instructions.behavior.file_reading_fallback }}
```

And have it resolve from the on-disk file:
```
_variables/instructions/behavior/file_reading_fallback.jinja2
```

Producing the nested feed dict:
```python
{"instructions": {"behavior": {"file_reading_fallback": "<content>"}}}
```

### 1.2 What actually happens today

`load_variables` (RichPythonUtils, line 716) does:

```python
# Dot-key: "notes.local_search_efficiency" → nested dict output
# for Jinja2 {{ notes.local_search_efficiency }} access.
# Splits into var_name="notes" (directory) + nested_key="local_search_efficiency" (file).
if "." in raw_key:
    base_var, nested_key = raw_key.split(".", 1)  # ← THE BUG: maxsplit=1
else:
    base_var, nested_key = raw_key, None
```

The `maxsplit=1` means:
- `"notes.local_search_efficiency"` → `("notes", "local_search_efficiency")` ✅ works (2-level)
- `"instructions.modes.deep_mode"` → `("instructions", "modes.deep_mode")` ❌ wrong

The wrong split is then passed to `_cascade_load_variable` which looks for a file literally named `modes.deep_mode.jinja2` (with a dot in the filename), not at `modes/deep_mode.jinja2`. File not found → returns `None` → variable disappears from the feed → Jinja2 renders empty.

### 1.3 The downstream `_find_variable_file` already handles multi-dot — but unreachably

`FileBasedVariableManager._find_variable_file` (RichPythonUtils, lines 712–718) correctly converts `instructions.modes.deep_mode` → `instructions/modes/deep_mode.jinja2`:

```python
if "." in variable_name:
    slash_name = variable_name.replace(".", "/")
    result = self._find_variable_file_with_slash(slash_name, cascade_paths, version)
```

**But this path is unreachable** for the dotted-key case because `load_variables` has already mangled the input before it gets to `_find_variable_file`.

### 1.4 The current workaround (must be removed by this plan)

`templated_inferencer_base.py:213` defines `_inject_mode_flags_and_content` which bypasses `load_variables` entirely and calls `_cascade_load_variable("instructions/modes", mode_name, ...)` with a pre-slashed path. The class also carries this self-incriminating comment:

```python
# Note: load_variables() in TemplateManager only splits on the
# FIRST dot (so "instructions.modes.deep_mode" resolves wrong).
# We use _cascade_load_variable directly for proper nested access.
```

This workaround handles `modes` only. It does NOT handle `behavior` (or any future N-level subdirectory). So users hit the same bug again when they try `{{ instructions.behavior.file_reading_fallback }}`.

**Fixing at the source eliminates both the workaround and the recurring user-pain.**

---

## 2. Target design

### 2.1 Field contract for `load_variables` keys

| Key shape | Splits as | Resolves to file | Feed dict shape |
|---|---|---|---|
| `"task_preamble"` | flat | `_variables/task_preamble.jinja2` (or via default_version) | `{"task_preamble": "<content>"}` |
| `"notes.local_search_efficiency"` | dir/file | `_variables/notes/local_search_efficiency.jinja2` | `{"notes": {"local_search_efficiency": "<content>"}}` |
| `"instructions.modes.deep_mode"` | dir/dir/file | `_variables/instructions/modes/deep_mode.jinja2` | `{"instructions": {"modes": {"deep_mode": "<content>"}}}` |
| `"instructions.behavior.file_reading_fallback"` | dir/dir/file | `_variables/instructions/behavior/file_reading_fallback.jinja2` | `{"instructions": {"behavior": {"file_reading_fallback": "<content>"}}}` |
| `"a.b.c.d.e.f"` | dir×5/file | `_variables/a/b/c/d/e/f.jinja2` | nested 6 levels deep |

**Rule:** All dots in a key become directory separators except the last segment, which is the file stem.

### 2.2 The deeply-nested feed dict

For `instructions.modes.deep_mode`, the feed must be:
```python
feed["instructions"]["modes"]["deep_mode"] = "<content>"
```

NOT a flat key `feed["instructions.modes.deep_mode"]` (Jinja2 attribute access can't traverse flat keys with dots in them).

### 2.3 Sibling-resolution (already works — preserve it)

When `file_reading_fallback_for_review.jinja2` references `{{ file_reading_fallback }}` internally, the existing `_resolve_variable` logic (RichPythonUtils `file_based.py:1024-1038`) already does sibling-first resolution using `current_level_path`. **This works today** and must NOT be regressed.

The fix in this plan only touches the *outer* key-splitting in `load_variables`. It does NOT touch the inner sibling resolution.

---

## 3. The fix — exact code change in RichPythonUtils

### 3.1 Edit to `template_manager.py:711-720`

**OLD (the bug):**
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
    # Dot-key: ALL dots become path separators; last segment is the file stem.
    #   "notes.local_search_efficiency"
    #     → folder_segments=["notes"], file_stem="local_search_efficiency"
    #     → file: _variables/notes/local_search_efficiency.jinja2
    #     → feed: {"notes": {"local_search_efficiency": "<content>"}}
    #   "instructions.modes.deep_mode"
    #     → folder_segments=["instructions", "modes"], file_stem="deep_mode"
    #     → file: _variables/instructions/modes/deep_mode.jinja2
    #     → feed: {"instructions": {"modes": {"deep_mode": "<content>"}}}
    if "." in raw_key:
        segments = raw_key.split(".")
        folder_segments = segments[:-1]
        file_stem = segments[-1]
        var_path = "/".join(folder_segments)   # for _cascade_load_variable's first arg
        nested_path = segments                  # for _store_nested
    else:
        var_path = raw_key
        file_stem = None
        nested_path = [raw_key]
```

### 3.2 Update `_store` to handle N-level nesting

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
def _store_nested(path: List[str], content: Any) -> None:
    """Store ``content`` at ``result[path[0]][path[1]]...[path[-1]]``,
    creating intermediate dicts as needed.

    Raises TypeError if an intermediate path segment already holds a
    non-dict value (e.g., a previous flat key shadowed a nested key).
    """
    if len(path) == 1:
        result[path[0]] = content
        return
    cur = result
    for seg in path[:-1]:
        existing = cur.get(seg)
        if existing is None:
            new_dict: Dict[str, Any] = {}
            cur[seg] = new_dict
            cur = new_dict
        elif isinstance(existing, dict):
            cur = existing
        else:
            raise TypeError(
                f"load_variables: cannot store nested key {'.'.join(path)!r} — "
                f"intermediate segment {seg!r} is already bound to a non-dict "
                f"value ({type(existing).__name__}). Did you mix flat and "
                f"dotted keys that share a prefix?"
            )
    cur[path[-1]] = content
```

### 3.3 Update the resolution-loop body to pass `var_path` and `file_stem`

Wherever the current loop calls `_cascade_load_variable(base_var, version, ...)` or `_store(base_var, nested_key, ...)`, swap to:
- `_cascade_load_variable(var_path, file_stem or default_version, root_space, tmpl_type)`
- `_store_nested(nested_path, content)`

Each spec semantic (`"@strict"`, `"=literal"`, plain) routes through `_store_nested` with the full `nested_path` list. The behavior for value semantics is unchanged.

### 3.4 Backward compatibility — explicit guarantees

After this change:
- Flat keys: identical behavior.
- 2-level dotted keys: identical behavior (e.g., `notes.local_search_efficiency` still produces `{"notes": {"local_search_efficiency": ...}}`).
- 3+-level dotted keys: previously broken; now correct.

**This is a strict superset of today's behavior.** No existing caller breaks.

---

## 4. The removal — eliminate `_inject_mode_flags_and_content`

### 4.1 Edit to `templated_inferencer_base.py:213`

**Delete:**
- The entire `_inject_mode_flags_and_content` method (~50 lines).
- The self-incriminating comment at the call site (lines 195–205 in the snippet above).

**Replace** the call site in `_build_template_feed`:

```python
# OLD (with workaround):
if self.modes:
    self._inject_mode_flags_and_content(feed)

# NEW (via load_variables):
if self.modes:
    # Flag derivation (unconditional so {%- if enable_X %} can check):
    for mode_name, enabled in self.modes.items():
        feed[f"enable_{mode_name}"] = bool(enabled)
    # Content loading (only for enabled modes):
    enabled_modes = [m for m, on in self.modes.items() if on]
    if enabled_modes and self.template_manager:
        mode_specs = {
            f"instructions.modes.{m}": ""
            for m in enabled_modes
        }
        try:
            loaded = self.template_manager.load_variables(
                mode_specs,
                root_space=self.template_root_space or "",
            )
        except FileNotFoundError as e:
            # Preserve existing behavior: enabled-but-missing is logged as debug.
            logger.debug(
                "Mode enabled but no instruction file found: %s", e,
            )
            loaded = {}
        # Deep-merge loaded into feed (loaded has shape
        # {"instructions": {"modes": {...}}}; merge with any existing
        # feed["instructions"] dict).
        _deep_merge_into(feed, loaded)
```

Add a small private helper:

```python
def _deep_merge_into(target: dict, source: dict) -> None:
    """Recursively merge ``source`` into ``target``. Dicts merge;
    non-dict leaves in ``source`` overwrite. Used to fold loaded
    template variables into the feed without losing other namespaces."""
    for k, v in source.items():
        if (
            isinstance(v, dict)
            and isinstance(target.get(k), dict)
        ):
            _deep_merge_into(target[k], v)
        else:
            target[k] = v
```

### 4.2 Why the workaround can be removed safely

The workaround did two things:
1. **Set `enable_<name>` flags unconditionally.** New code does the same in a 3-line loop.
2. **Load `instructions/modes/<name>` content for enabled modes.** New code does this via `load_variables` with the fixed multi-dot support.

Behavior matches; the workaround is no longer load-bearing.

### 4.3 Behaviors preserved (regression checklist)

- `enable_deep_mode` / `enable_elegant_mode` etc. always present in feed.
- Content only loaded for enabled modes.
- Disabled modes contribute nothing to `feed["instructions"]["modes"]`.
- Missing instruction file → debug log (not error).
- Other exceptions → warning log.
- Sibling resolution within variable files (e.g., `file_reading_fallback_for_review` referencing `file_reading_fallback`) untouched.

---

## 5. Test plan

### 5.1 RichPythonUtils unit tests (NEW file)

`CoreProjects/RichPythonUtils/test/string_utils/formatting/template_manager/test_load_variables_multidot.py`

| # | Test | What it pins |
|---|------|---------------|
| 1 | `test_flat_key_unchanged` | Backward compat: `{"task_preamble": "default"}` resolves identically |
| 2 | `test_two_level_dotted_key_unchanged` | Backward compat: `{"notes.local_search_efficiency": ""}` produces nested 2-level dict |
| 3 | `test_three_level_dotted_key_NEW` | `{"instructions.modes.deep_mode": ""}` produces `{"instructions": {"modes": {"deep_mode": "<content>"}}}` |
| 4 | `test_four_level_dotted_key_NEW` | `{"a.b.c.d": ""}` produces 4-level nested dict |
| 5 | `test_multiple_keys_share_root_merge` | `{"instructions.modes.deep_mode": "", "instructions.behavior.file_reading_fallback": ""}` produces a single `instructions` dict with both `modes` and `behavior` sub-dicts |
| 6 | `test_flat_then_nested_raises_clear_error` | `{"instructions": "literal", "instructions.modes.x": ""}` raises `TypeError` with the documented message |
| 7 | `test_strict_prefix_still_raises_on_missing` | `{"instructions.modes.nonexistent": "@strict"}` raises `FileNotFoundError` |
| 8 | `test_literal_prefix_still_skips_file` | `{"instructions.modes.x": "=hello world"}` produces `{"instructions": {"modes": {"x": "hello world"}}}` without filesystem read |

### 5.2 AgentFoundation integration tests

The existing `test/agent_foundation/common/inferencers/test_behavior_variable_injection.py` (156 lines) was failing on the multi-dot case. After this fix, those tests should **GREEN**. Specifically:

- `test_base_fallback_variable_loads` — currently asserts `BASE_FALLBACK_TEXT in rendered`; FAILS today because `instructions.behavior.file_reading_fallback` doesn't resolve. GREENS after fix.
- `test_review_fallback_variable_loads` — same pattern with `file_reading_fallback_for_review`. GREENS after fix.
- `test_followup_fallback_variable_loads` — same pattern with `file_reading_fallback_for_followup`. GREENS after fix.

### 5.3 New permanent regression in AgentFoundation

`test/agent_foundation/common/inferencers/test_templated_inferencer_base_modes.py` (NEW):

```python
def test_inject_mode_flags_and_content_method_is_removed():
    """The workaround method must NOT come back. If it does, someone
    re-introduced a hack. See plan §4."""
    from agent_foundation.common.inferencers.templated_inferencer_base import (
        TemplatedInferencerBase,
    )
    assert not hasattr(TemplatedInferencerBase, "_inject_mode_flags_and_content"), (
        "The _inject_mode_flags_and_content workaround was re-added. "
        "Mode injection should go through TemplateManager.load_variables "
        "with proper multi-dot support. See plan §4."
    )


def test_deep_mode_renders_via_load_variables(tmp_template_manager):
    """End-to-end: a class with modes={'deep_mode': True} produces a
    feed where instructions.modes.deep_mode is the file content."""
    # ...uses an in-process TemplatedInferencerBase subclass + a temp
    # _variables/instructions/modes/deep_mode.jinja2 fixture...
```

### 5.4 Run the existing `test_behavior_variable_injection.py` as the smoke test

Phase 0 of implementation MUST run this test file against the current code (it fails today). After the fix lands, it must GREEN. This is the strongest validation signal because it was written from the user-facing perspective.

---

## 6. Phased rollout

| Phase | What | Files | Risk | Reversible? |
|---|---|---|---|---|
| 0 | Pre-flight: confirm `test_behavior_variable_injection.py` currently FAILS | none (just run) | none | n/a |
| 1 | RichPythonUtils fix: extend `load_variables` for N-dot, add `_store_nested` | 1 file (`template_manager.py`) | medium (cross-repo) | yes |
| 2 | RichPythonUtils unit tests (NEW) | 1 new test file | none | yes |
| 3 | AgentFoundation: delete `_inject_mode_flags_and_content`; rewrite mode handling via `load_variables` | 1 file (`templated_inferencer_base.py`) | medium | yes |
| 4 | AgentFoundation: add `_deep_merge_into` helper or import equivalent | same as Phase 3 | low | yes |
| 5 | AgentFoundation: add the permanent "workaround-must-not-return" regression test | 1 new test file | trivial | yes |
| 6 | Run `test_behavior_variable_injection.py` → assert GREEN | none | n/a | n/a |

### 6.1 Cross-repo PR ordering

- **PR-A (RichPythonUtils):** Phase 1 + Phase 2. Lands first. Strictly backward-compatible.
- **PR-B (AgentFoundation):** Phases 3 + 4 + 5 + 6. Depends on PR-A. Lands after PR-A is published / pinned in AgentFoundation's dependency.

Do NOT bundle. Cross-repo bundles are hard to revert.

---

## 7. Risk register

| # | Risk | Severity | Mitigation |
|---|------|---------|------------|
| 1 | Some existing caller relies on the FIRST-dot-split bug (e.g., a key like `"foo.bar.with.dots.in.it"` that today resolves to a file literally named `bar.with.dots.in.it.jinja2`) | 🟡 Medium | Grep AgentFoundation + RichPythonUtils for `load_variables(` and inspect every call. Investigation already done: only flat and 2-level dotted keys exist in the codebase. New behavior is a strict superset for those. |
| 2 | The new `_store_nested` raises `TypeError` for prefix collisions (e.g., `{"instructions": "x", "instructions.modes.y": ""}`). | 🟢 Low | This is a NEW error mode; existing callers don't mix flat and nested under the same root. The error message is clear; documented in §3.2. Add explicit test #6. |
| 3 | Removing `_inject_mode_flags_and_content` breaks a downstream subclass that overrides it. | 🟡 Medium | Grep for overrides: `def _inject_mode_flags_and_content` should appear only in `templated_inferencer_base.py`. Verify; if any subclass overrides, migrate them in the same PR. |
| 4 | The `_deep_merge_into` helper is needed only because `feed` may already contain `feed["instructions"]` from elsewhere. | 🟢 Low | Today `feed["instructions"]` is set ONLY by the workaround. After removal, the only writer is `load_variables`'s output. Deep-merge is defensive (handles future namespace additions). |
| 5 | Python `dict.setdefault(...).update(...)` would have worked for 2-level but not N-level. Using a recursive helper is correct but slightly slower for deep dicts. | 🟢 Low | N is bounded by template-key dot count (typically 2–3). Performance non-issue. |
| 6 | Cross-repo coordination: PR-A merges, PR-B doesn't get rebased onto new RichPythonUtils. | 🟡 Medium | Pin RichPythonUtils version in AgentFoundation's pyproject.toml as part of PR-B. CI catches version mismatch via the new tests. |
| 7 | A future contributor re-adds `_inject_mode_flags_and_content` as a "fix" to some local issue. | 🟢 Low | Permanent regression test §5.3 makes this immediately visible in CI. |
| 8 | `test_behavior_variable_injection.py` makes assumptions about `TemplateManager.__init__` signature that change in a future RichPythonUtils release. | 🟢 Low | The test already uses kwargs (`templates=`, `template_formatter=`); pin a minimum RichPythonUtils version. |

---

## 8. Acceptance criteria

PR-A (RichPythonUtils) is mergeable when:
- ☐ All 8 new unit tests in §5.1 pass.
- ☐ The full existing RichPythonUtils test suite passes (no regression).
- ☐ Code review confirms the docstring of `load_variables` is updated to describe N-level support.

PR-B (AgentFoundation) is mergeable when:
- ☐ `_inject_mode_flags_and_content` is removed from `templated_inferencer_base.py`.
- ☐ The 3 existing tests in `test_behavior_variable_injection.py` GREEN.
- ☐ The new "workaround-must-not-return" regression test (§5.3) is in CI.
- ☐ `grep -rn "_inject_mode_flags_and_content" CoreProjects/AgentFoundation/` returns zero matches.
- ☐ Manual smoke: render `plan/main/initial.jinja2` with `modes={"deep_mode": True}` and verify the output contains the deep-mode instruction text (not literal `{{ instructions.modes.deep_mode }}`).

---

## 9. What this plan deliberately does NOT do

- ❌ **Does NOT rename `load_variables`.** The name is fine; only the implementation is buggy.
- ❌ **Does NOT change the `@strict` / `=literal` / plain value semantics.** Those are orthogonal to the dot-splitting bug.
- ❌ **Does NOT change `_find_variable_file` or `_resolve_variable`.** They already handle multi-dot correctly; the bug was only in the caller (`load_variables`).
- ❌ **Does NOT change the sibling-first resolution** for nested variable references inside variable files.
- ❌ **Does NOT introduce a feature flag** for the new behavior. The fix is strictly backward-compatible; flagging it would add complexity for no safety gain.
- ❌ **Does NOT touch any prompt template files.** The user already updated them to use `{{ instructions.behavior.file_reading_fallback }}`; this plan makes those references resolve correctly.

---

## 10. Design principles applied

1. **Fix at the source.** The bug is in RichPythonUtils. Patching AgentFoundation around it (the current workaround) is the wrong layer.
2. **Strict backward compatibility.** The fix is a superset of today's behavior. No existing caller is affected.
3. **Eliminate workarounds when fixing the source.** Don't leave the bypass behind once it's redundant — and pin its absence with a permanent regression test.
4. **Clear error messages over silent corruption.** The new `_store_nested` raises `TypeError` with a documented message when flat and nested keys collide, instead of silently shadowing.
5. **Test the user-facing path.** The existing `test_behavior_variable_injection.py` was written from the user perspective; making it GREEN is the strongest validation.
6. **Cross-repo discipline.** Two PRs, ordered, not bundled. Easier review, easier rollback.

---

## 11. Estimated effort

| Phase | Implementation | Tests | Review | Total (h) |
|---|---|---|---|---|
| 1 (RichPythonUtils fix) | 1 | 0 | 1 | 2 |
| 2 (RichPythonUtils tests) | 0 | 3 | 1 | 4 |
| 3 (AgentFoundation removal) | 2 | 0 | 1 | 3 |
| 4 (deep_merge helper) | 0.5 | 0.5 | 0.5 | 1.5 |
| 5 (regression test) | 0.5 | 0.5 | 0.5 | 1.5 |
| 6 (smoke validation) | 0 | 0.5 | 0.5 | 1 |
| **Total** | **4** | **4.5** | **4.5** | **13** |

Roughly **1.5 engineer-days** total across both repos. Small, surgical, high-leverage.

---

*End of plan. Reviewers: please challenge §3 (the exact code edit), §4.2 (workaround removal safety), and §7 risk #1 (existing-caller compatibility) most carefully.*
