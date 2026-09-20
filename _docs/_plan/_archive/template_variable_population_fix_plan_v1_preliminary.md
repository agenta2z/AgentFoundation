# Template Variable Population Fix Plan — `task_instructions` (and friends) Missing in Fixer/Reviewer Prompts

**Status**: ACTIVE — root cause identified with hard evidence (2026-05-10)
**Severity**: 🔴 HIGH — silent failure; entire `task_instructions` section missing from fixer/reviewer prompts
**Effort**: ~30 min code + ~30 min tests = ~1 hour

---

## §1 The Bug

**Symptom**: Templates `plan/main/{review,followup}.jinja2` (and `implementation/main/{review,followup}.jinja2`) reference `{{ task_instructions }}` inside `{% if task_instructions %}` guards. The variable is never populated. Guards silently skip. Rendered prompts have NO instruction content.

**Hard evidence** (from `task_task-cbaf8f2b_20260510_033521` run):
- `grep -c task_instructions <fixer prompt>` → **0**
- `grep -c "Task Instructions:" <fixer prompt>` → **0**

## §2 Root Cause

Dual's `SLOT_DEFAULTS` (line 161-169 of `dual_inferencer.py`):
```python
SLOT_DEFAULTS = {
    "review_inferencer": REVIEW_TEMPLATE_DEFAULTS,
    "fixer_inferencer": FOLLOWUP_TEMPLATE_DEFAULTS,
}
```

These defaults (defined in `template_defaults.py:290-307`) only set `template_key`:
```python
REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key=KEY_REVIEW,         # ← ONLY this
)
FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key=KEY_FOLLOWUP,       # ← ONLY this
)
```

**No `template_variables` declared.** Compare with `STRUCTURED_AGGREGATION_DEFAULTS` (line 271) used for aggregator slots, which correctly declares all three variables:
```python
STRUCTURED_AGGREGATION_DEFAULTS = InferencerTemplateVersionDefaults(
    template_version=VARIANT_AGGREGATION,
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)
```

So aggregator prompts get `task_instructions` populated. Fixer/reviewer prompts DON'T.

## §3 The Fix

Change `template_defaults.py` lines 290-307 (REVIEW + FOLLOWUP defaults) from `InferencerTemplateDefaults` to `InferencerTemplateVersionDefaults` (which auto-declares `template_variables`):

```python
REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateVersionDefaults(
    template_key=KEY_REVIEW,
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)
"""For any review slot: render the canonical ``review`` template variant
WITH the standard task variables populated (preamble, instructions, response_format).
Mirrors STRUCTURED_AGGREGATION_DEFAULTS for fixer/reviewer slots."""

FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateVersionDefaults(
    template_key=KEY_FOLLOWUP,
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)
"""For any followup/fixer slot: render the canonical ``followup`` template
variant WITH standard task variables populated."""
```

**Why elegant**: same mechanism that already works for aggregator slots; no YAML clutter; zero new code paths; parallel architecture.

## §4 Why This is the Right Fix (Not Hacky)

| Property | Verdict |
|---|---|
| Same mechanism as already-working slots | ✅ (mirrors `STRUCTURED_AGGREGATION_DEFAULTS`) |
| Zero YAML clutter | ✅ (cascade is automatic) |
| No new code paths | ✅ (just declaring defaults) |
| Architecturally consistent | ✅ (review/followup parallel to aggregation) |
| Backward-compatible | ✅ (variant=None means "use default variant"; templates without `{% if %}` guard would also work) |

## §5 Acceptance Criteria

- [ ] After fix, `grep -c task_instructions <fixer rendered prompt>` returns ≥ 1
- [ ] Rendered prompt contains "Task Instructions:" section (or whatever the variant file outputs)
- [ ] No regression in existing 214-test suite
- [ ] Aggregator prompts unchanged (already worked)

## §6 Tests

| # | Test | File | What it verifies |
|---|---|---|---|
| 1 | `test_followup_defaults_declare_task_variables` | `test/.../test_template_defaults.py` | `FOLLOWUP_TEMPLATE_DEFAULTS.template_variables` contains `task_preamble`, `task_instructions`, `task_response_format` |
| 2 | `test_review_defaults_declare_task_variables` | `test/.../test_template_defaults.py` | Same for REVIEW_TEMPLATE_DEFAULTS |
| 3 | `test_dual_fixer_renders_with_task_instructions` | `test/.../test_dual_inferencer.py` | Mock leaf, run propose→review→fix, verify rendered fixer prompt contains `task_instructions` content (not just placeholder) |
| 4 | `test_dual_reviewer_renders_with_task_instructions` | `test/.../test_dual_inferencer.py` | Same for reviewer |

## §7 Optional Safety Net (Deferred)

Add a preflight validator that scans templates and warns when a referenced variable isn't declared in `template_variables`:

```python
# In TemplatedInferencerBase.__attrs_post_init__:
template_text = self.template_manager.get_raw_template(...)
referenced_vars = re.findall(r'{{\s*(\w+)\s*}}', template_text)
ungrounded_in_if = [v for v in referenced_vars if v not in self.template_variables and v not in IMPLICIT_VARS]
if ungrounded_in_if:
    _logger.warning(f"Template references {ungrounded_in_if} but they're not in template_variables. May render as None.")
```

This catches future regressions but doesn't fix the current bug. Defer to a follow-up if time permits.

## §8 Provenance

- 2026-05-10 11:00 — Issue surfaced during cross-review of MFDual hygiene plan v4.6
- 2026-05-10 11:04 — Hard evidence collected: 0 occurrences in fixer prompt despite 14 template references
- 2026-05-10 11:05 — Root cause identified: `REVIEW_TEMPLATE_DEFAULTS` and `FOLLOWUP_TEMPLATE_DEFAULTS` declared as `InferencerTemplateDefaults` (template_key only) instead of `InferencerTemplateVersionDefaults` (template_key + variable_names). Fix is 4-line change in `template_defaults.py`.
