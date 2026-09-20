# Template Variable Population — Unified Fix Plan (v2.0 — semantic fix)

**Status**: ACTIVE (v2.0 — clean conceptual fix; no tactical patches)
**Severity**: 🔴 HIGH — silent failure; `task_instructions` (and any default-having variable) missing from prompts
**Total Effort**: ~2 hours (single architectural fix + tests)

---

## §1 The Bug

**Symptom**: Templates reference `{{ task_instructions }}` inside `{% if task_instructions %}` guards. The variable is never populated. Guards silently skip. Rendered prompts have NO instruction content.

**Hard evidence** (run `task_task-cbaf8f2b_20260510_033521`):
- `grep -c task_instructions <fixer prompt>` → **0**
- 14 template references vs 0 actual usages = 100% silent skip

## §2 Root Cause — Semantic Conflation

`templated_inferencer_base.py:163`:
```python
if self.template_variables and self.template_manager:  # ← THE BUG
    resolved = self.template_manager.load_variables(...)
```

This conflates **TWO orthogonal concerns** into ONE condition:

| Concern | Should be | Is currently |
|---|---|---|
| **"Should I scan `_variables/` for defaults to auto-load?"** | Always yes (when `template_root_space` set) | Gated on `template_variables` non-empty |
| **"What overrides do I apply?"** | Whatever's in `template_variables` | Same gate |

The `default.jinja2` filename convention (e.g., `_variables/task_instructions/default.jinja2` exists with 2300 bytes) PROVES the original design intent was "auto-load defaults; `template_variables` is for overrides."

But the code never implemented the auto-discovery side. So:
- `template_variables = {}` → no loading at all (BUG)
- `template_variables = {task_preamble: aggregation}` → ONLY `task_preamble` loaded; `task_instructions`/`task_response_format` still NOT loaded (also BUG)

**The conflation is the bug.** `template_variables` should be PURELY about overrides, never about whether discovery happens.

## §3 The Fix — Decouple Discovery from Override

**ONE conceptual change**: separate discovery (always-on) from override (driven by `template_variables`).

### Code Change in `templated_inferencer_base._build_template_feed()`

```python
def _build_template_feed(self, inference_input: Any, *, extra_feed=None) -> dict:
    feed: dict = {}

    if self.template_manager and self.template_root_space:
        # Step A (NEW): Auto-load all available defaults from _variables/.
        # Unconditional — always scan if root_space is set. This implements
        # the documented design intent: `_variables/<name>/default.jinja2`
        # is THE default for the variable named <name>. Cached per instance.
        defaults = self._load_default_variables()
        feed.update(defaults)

        # Step B (EXISTING, but gate fixed): Apply overrides.
        # `template_variables` is PURELY for overrides — empty means
        # "use all defaults as-is." Non-empty means "for these specific
        # variables, use this variant instead of default."
        if self.template_variables:  # only RESOLVE if there ARE overrides
            resolved = self.template_manager.load_variables(
                variable_specs=self.template_variables,
                root_space=self.template_root_space,
                default_version=self.template_version or "",
            )
            feed.update(resolved)  # overrides win over defaults

    # Steps C, D, E (unchanged): template_extra_feed, extra_feed, modes, input.
```

### New Helper Method

```python
def _load_default_variables(self) -> dict:
    """Auto-load `_variables/<name>/default.jinja2` files under root_space.

    Implements the design intent: any `_variables/` subdirectory with a
    `default.jinja2` file is auto-loaded as a feed variable, with the
    variable name = directory name and value = file content.

    Cached per inferencer instance (filesystem scan once).

    Returns:
        Dict mapping variable name → rendered content. Empty if no
        `_variables/` exists or no `default.jinja2` files found.
    """
    cache_attr = "_default_variables_cache"
    if hasattr(self, cache_attr):
        return getattr(self, cache_attr)

    defaults = {}
    try:
        var_names = self.template_manager.discover_default_variables(
            root_space=self.template_root_space,
            template_type=self.template_type or "main",
        )
        for var_name in var_names:
            try:
                content = self.template_manager.load_variable(
                    var_name=var_name,
                    version="default",
                    root_space=self.template_root_space,
                )
                defaults[var_name] = content
            except (FileNotFoundError, AttributeError):
                pass  # variant changed between scan and load
    except (AttributeError, NotImplementedError):
        pass  # template_manager doesn't support discovery yet

    object.__setattr__(self, cache_attr, defaults)
    return defaults
```

### Companion Method in `TemplateManager`

```python
def discover_default_variables(
    self,
    root_space: str,
    template_type: str = "main",
) -> list[str]:
    """Scan `<root_space>/<template_type>/_variables/*/default.jinja2`.

    Returns list of variable names (directory names that contain default.jinja2).

    Returns empty list if `_variables/` doesn't exist (graceful degradation).
    """
    candidates = [
        f"{root_space}/{template_type}/_variables",
        f"{root_space}/_variables",
        "_variables",
    ]
    found = set()
    for base in candidates:
        for search_path in self.search_paths:
            full = os.path.join(search_path, base)
            if not os.path.isdir(full):
                continue
            for entry in os.listdir(full):
                entry_path = os.path.join(full, entry)
                if os.path.isdir(entry_path) and \
                   os.path.isfile(os.path.join(entry_path, "default.jinja2")):
                    found.add(entry)
    return sorted(found)
```

## §4 Why This is Elegant (Not Hacky)

| Property | Verdict |
|---|---|
| Fixes the conceptual bug, not a symptom | ✅ Decouples discovery from override |
| Implements documented design intent | ✅ `default.jinja2` was always meant as default-fallback |
| Zero per-slot declarations needed | ✅ No SLOT_DEFAULTS patches; no `variable_names=[...]` |
| Backward-compatible | ✅ Existing `template_variables` still works as overrides |
| Idiomatic | ✅ Same caching pattern as other inferencer state |
| Cached for performance | ✅ Filesystem scan once per instance |
| Future variables auto-work | ✅ Drop `_variables/foo/default.jinja2` → `{{ foo }}` works everywhere |

## §5 What This Plan Replaces

| Old approach (REJECTED) | Why rejected |
|---|---|
| **Phase 1: Patch SLOT_DEFAULTS** to add `variable_names=[VAR_TASK_PREAMBLE, ...]` | Tactical patch around the same conceptual bug. Each new variable would need adding to every SLOT_DEFAULTS. Doesn't fix the underlying conflation. |
| **3-phase staged plan** with separate "tactical" + "elegant" + "safety" phases | Splitting the fix delays the elegance. The conceptual fix is small enough to do in one PR. |
| **Per-YAML `template_variables` declarations** | YAML clutter; same bug recurs every time someone forgets to declare. |

## §6 Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | Fixer rendered prompt contains `task_instructions` content (≥ 1 occurrence in grep) |
| 2 | Reviewer rendered prompt contains `task_instructions` content |
| 3 | Aggregator prompts unchanged (already worked, must not regress) |
| 4 | `template_variables: {task_preamble: aggregation}` overrides default `task_preamble` BUT `task_instructions` STILL loaded as default |
| 5 | Adding a new `_variables/foo/default.jinja2` → `{{ foo }}` auto-populates everywhere |
| 6 | `template_variables = {}` (default) → all defaults loaded automatically |
| 7 | All 214 existing tests still pass |
| 8 | Performance: filesystem scan happens ONCE per inferencer instance (cached) |

## §7 Tests

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_defaults_auto_load_when_template_variables_empty` | Empty `template_variables` still loads `task_preamble`/`task_instructions` defaults |
| 2 | `test_explicit_overrides_win_over_defaults` | `{task_preamble: aggregation}` makes `task_preamble` the aggregation variant; `task_instructions` STILL default |
| 3 | `test_partial_overrides_dont_disable_other_defaults` | Overriding ONE variable doesn't suppress loading of OTHER defaults |
| 4 | `test_no_variables_dir_graceful_degradation` | If `_variables/` doesn't exist, returns `{}` without error |
| 5 | `test_discovery_cached_per_instance` | Two `ainfer()` calls trigger ONE filesystem scan |
| 6 | `test_dual_fixer_renders_with_task_instructions` | Mock leaf, run propose→review→fix, verify rendered fixer prompt contains task_instructions |
| 7 | `test_dual_reviewer_renders_with_task_instructions` | Same for reviewer |
| 8 | `test_aggregator_renders_unchanged` | STRUCTURED_AGGREGATION_DEFAULTS path still works (no regression) |
| 9 | `test_default_jinja2_missing_skipped_silently` | Subdirectory without `default.jinja2` → not in feed (correct skip) |

## §8 Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Performance: filesystem scan per inferencer instance | Cached on first access (`_default_variables_cache` instance attribute); subsequent calls O(1) |
| Backward compat: existing inferencers might suddenly auto-load variables they don't use | These get added to feed dict but unused (no template reference) — no behavior change. Templates without the matching `{{ var }}` simply ignore the value. |
| `discover_default_variables()` not yet implemented in TemplateManager | Helper catches `AttributeError`/`NotImplementedError` → graceful degradation |
| Existing aggregator slot might double-load (defaults + STRUCTURED_AGGREGATION_DEFAULTS overrides) | Overrides win over defaults (Step B updates feed AFTER Step A); no conflict |
| Some `_variables/<name>/` dirs only have `<variant>.jinja2`, no `default.jinja2` | Discovery skips them (only loads when `default.jinja2` exists) — explicit `template_variables` still works for these |

## §9 Resolved Open Questions

**Q: What about `template_variables = None`?**
A: Can't happen — typed as `dict` with `factory=dict` (line 95). Always at least `{}`. Same code path as `{}`.

**Q: Should `template_variables = {}` mean "load nothing" or "load all defaults"?**
A: Load all defaults. `template_variables` is for OVERRIDES only. To load nothing, set `template_root_space = None` (which disables both discovery and resolution).

**Q: What if multiple `_variables/` dirs along the cascade have the same variable?**
A: Use the most-specific one (mirror existing `load_variable()` cascade). E.g., `<space>/<type>/_variables/<name>/default.jinja2` wins over `<space>/_variables/<name>/default.jinja2` wins over global `_variables/<name>/default.jinja2`.

**Q: Should we also scan for non-default variants and warn if multiple exist without explicit selection?**
A: No. That's a different feature ("variant discovery"). This plan is about default-loading.

## §10 Provenance

- 2026-05-10 11:00 — Issue surfaced during cross-review of MFDual hygiene plan v4.6
- 2026-05-10 11:04 — Hard evidence: 0 occurrences in fixer prompt despite 14 template references
- 2026-05-10 11:05 — First diagnosis: SLOT_DEFAULTS don't cascade `template_variables`; tactical fix proposed
- 2026-05-10 11:09 — User insight: `template_variables` SEMANTIC is overrides, should NOT decide if loading happens
- 2026-05-10 11:10 — Root cause refined: the `if self.template_variables` gate conflates discovery + override into one condition. SLOT_DEFAULTS patch is a symptom-patch, not a root-cause fix.
- 2026-05-10 11:11 — v2.0 plan written: SINGLE conceptual fix decouples discovery (always-on) from override (driven by `template_variables`). No more 3-phase staging.
