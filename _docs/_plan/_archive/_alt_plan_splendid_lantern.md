# Integrated Plan: Path-Aware Followup for DualInferencer

> Integrates best of Plan A (edge case clarity, backward compat emphasis) and Plan B (structured feed dict, file-level path, template updates, comprehensive tests).

---

## 1. Problem

When `base_inferencer` is a BTA (or any file-producing orchestrator), `state["base_output_str"]` captures the LLM's textual summary (~2.7KB), NOT the file deliverable (~43KB plan). Four bugs compound:

| Bug | Description |
|-----|-------------|
| **A** | `<ProposedDocument>{{ main_response }}</ProposedDocument>` renders **empty** — feed dict sets `proposal` key but template expects `main_response` |
| **B** | `<ReviewerFeedback>{{ reviewer_response }}</ReviewerFeedback>` renders **empty** — conditional guard + upstream state gap |
| **C** | Prior output **file path** never given to LLM — the MUST-copy directive is mechanically unobeyable |
| **D** | Even `<CurrentProposal>` has the **wrong content** — 2.7KB summary instead of 43KB plan |

This plan fixes A + C, mitigates D (path lets LLM read the actual file), and documents B as a follow-up.

---

## 2. Design

### 2.1 Core Idea

Add a helper on `DualInferencer` that resolves the active proposer's on-disk output **file path** using a deterministic two-tier rule. Plumb the resolved path AND `main_response` into the feed dict as **separate structured variables**. Update per-domain Jinja templates to use `{{ prior_output_path }}` for executable `cp` instructions.

### 2.2 Path Resolution Rule

```
Tier 1: proposer has non-empty final_deliverables/
  → return the deliverable FILE path (preferred basename match, then first non-dotfile)
Tier 2: proposer has outputs/<basename> on disk
  → return that file path
Tier 3: neither exists
  → return None (template renders graceful fallback)
```

### 2.3 Proposer Selection: `_active_proposer()`

Uses the existing `_active_proposer()` method (no iteration parameter needed). Verified correct during prompt building despite one-step lag in iteration record appending:

```
1st fix: iterations=[] → returns base  ✓ (base produced current proposal)
2nd fix: iterations=[r1(cf=set)] → returns fixer  ✓ (fixer produced current proposal)
nth fix: last record has cf set → returns fixer  ✓
```

A code comment should document this timing subtlety.

### 2.4 Why Structured Variables, Not String Concatenation

Path metadata must be a **separate feed dict variable** (`prior_output_path`), not concatenated into the proposal string, because:

1. Concatenation does NOT fix Bug A — `{{ main_response }}` stays empty unless `placeholder_proposal="main_response"`
2. Templates need conditional control: `{% if prior_output_path %}` for `cp` instructions vs graceful fallback
3. The `or ""` sentinel prevents `"None"` literal leaking into rendered prompts

---

## 3. Changes

### Phase 1 — Add `_resolve_prior_proposer_output_path()` helper

**File:** `dual_inferencer.py`, near `_active_proposer()` (~line 471)

```python
def _resolve_prior_proposer_output_path(self) -> Optional[str]:
    """Return the on-disk file path of the active proposer's output.

    Two-tier resolution (deterministic, domain-agnostic):
      Tier 1 — Deliverable file (preferred for orchestrators: BTA, PTI):
        If proposer's workspace has non-empty final_deliverables/,
        return the file matching proposer's output_path basename
        (typically "output.md"), else first non-dotfile alphabetically.
      Tier 2 — Outputs file (canonical for leaf inferencers):
        If proposer's outputs/<basename> exists on disk, return it.
      Tier 3 — None: no usable file on disk.

    Note on timing: called from _build_followup_prompt and
    _build_review_prompt. At those call sites, _active_proposer()
    correctly identifies who produced the current proposal because
    iteration records are appended AFTER _step_fix_impl completes
    (line 1032), creating a one-step lag that aligns with the
    "who produced the CURRENT state[base_output_str]" question.
    """
    proposer = self._active_proposer()
    if proposer is None:
        return None
    ws = getattr(proposer, "_workspace", None)
    if ws is None:
        return None

    # Tier 1: deliverable file
    if ws.has_deliverables:
        preferred_basename = os.path.basename(
            getattr(proposer, "_output_path", None) or "output.md"
        )
        preferred = ws.deliverable_path(preferred_basename)
        if preferred and os.path.isfile(preferred):
            return preferred
        paths = [
            p for p in ws.deliverable_paths()
            if not os.path.basename(p).startswith(".")
        ]
        if paths:
            candidate = os.path.join(ws.deliverables_dir, paths[0])
            if os.path.isfile(candidate):
                return candidate

    # Tier 2: outputs/<basename>
    out_basename = os.path.basename(
        getattr(proposer, "_output_path", None) or "output.md"
    )
    out_path = ws.output_path(out_basename)
    if os.path.isfile(out_path):
        return out_path

    return None
```

**Invariants:**
- Pure read-only: no filesystem mutation
- Returns absolute path or None (never raises)
- Tier 1 dotfile filter prevents returning `.self_promoted` markers
- Tier 1 preferred-basename is deterministic (no glob ordering surprises)

### Phase 2 — Plumb `prior_output_path` and `main_response` into prompt builders

**File:** `dual_inferencer.py`

**2a: `_build_followup_prompt`** (lines 1115-1142) — add to feed dict:

```python
feed = {
    self.placeholder_input: inference_input,
    self.placeholder_proposal: proposal,
    self.placeholder_issues: self._serialize_issues(issues),
    self.placeholder_reasoning: reasoning,
    "enable_counter_feedback": config.enable_counter_feedback,
    "iteration": iteration,
    "attempt": attempt,
    "round_index": iteration,
    # === NEW: populate outer template slots ===
    "main_response": proposal,
    "prior_output_path": self._resolve_prior_proposer_output_path() or "",
}
if review_output is not None:
    feed["reviewer_response"] = review_output
return self._render_role_prompt("followup", feed, inference_config)
```

**2b: `_build_review_prompt`** (lines 1094-1113) — same pattern:

```python
feed = {
    self.placeholder_input: inference_input,
    self.placeholder_proposal: proposal,
    "iteration": iteration,
    "attempt": attempt,
    "round_index": iteration - 1,
    # === NEW ===
    "main_response": proposal,
    "prior_output_path": self._resolve_prior_proposer_output_path() or "",
}
if counter_feedback is not None:
    feed[self.placeholder_counter_feedback] = counter_feedback
return self._render_role_prompt("review", feed, inference_config)
```

**Why `"main_response": proposal` alongside `self.placeholder_proposal: proposal`:**
Zero-cost data duplication. Fixes Bug A (empty `<ProposedDocument>`) regardless of how `placeholder_proposal` is configured. Both the inner template (`{{ proposal }}`) and outer template (`{{ main_response }}`) get the data.

**Why `or ""` not raw `None`:**
Jinja `{% if prior_output_path %}` treats empty string as falsy. But `{{ prior_output_path }}` with `None` would emit literal `"None"` text if the if-block is accidentally omitted. Empty string is the safest sentinel.

### Phase 3 — Update `plan/main/followup.jinja2` to use `prior_output_path`

**File:** `prompt_templates/plan/main/followup.jinja2`, lines 12-15

**Replace:**
```jinja2
**Your Current Proposed Document:**
<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**With:**
```jinja2
**Your Current Proposed Document:**
{%- if prior_output_path %}
The previous version is saved on disk at:
  `{{ prior_output_path }}`

To preserve content, your FIRST tool action MUST be:
    cp {{ prior_output_path }} {{ output_path }}

Then apply targeted in-place edits to `{{ output_path }}`.
DO NOT retype the document — copy the file first and edit it.
{%- else %}
(The previous version path is unavailable — use the inline content
below as reference. Preserve title, section numbering, and length.)
{%- endif %}

<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**Backward compatibility:** If `prior_output_path` is undefined or empty, the `else` branch renders a graceful fallback. `<ProposedDocument>` is always populated (thanks to Phase 2's `main_response` plumbing).

### Phase 4 — Symmetric updates to sibling templates

Same `{% if prior_output_path %}` block (adapted wording) in:
- `plan/main/review.jinja2` — reviewer wording: "inspect it directly via read_file"
- `implementation/main/followup.jinja2` — same cp pattern
- `implementation/main/review.jinja2` — same reviewer wording

---

## 4. Backward Compatibility

| Rendering Path | Effect |
|---|---|
| **Path A** (TemplateManager → outer Jinja) | Fixed: `main_response` and `prior_output_path` populate outer template slots |
| **Path B** (per-role TMs) | Same — feed dict passed through |
| **Path C** (raw default template) | Unchanged — new keys (`main_response`, `prior_output_path`) silently unused by inner template. Existing `{{ proposal }}` still works via `self.placeholder_proposal` |

No new attributes on DualInferencer. No changes to `__attrs_post_init__`. No inheritance changes. Pure additive feed-dict additions + one new method.

## 5. Edge Cases

| Edge case | Behavior |
|---|---|
| No workspace (`_workspace is None`) | `_resolve_prior_proposer_output_path` → None → `prior_output_path=""` → template renders fallback |
| Empty deliverables dir (created by `ensure_dirs()` but never written) | `has_deliverables` checks non-empty → falls through to Tier 2 |
| Only dotfile markers in deliverables (`.self_promoted`) | Dotfile filter skips them → falls through to Tier 2 |
| Fixer == base_inferencer (2-agent mode) | Same workspace → path correct for whichever iteration |
| outputs/ empty | `os.path.isfile(out_path)` → False → returns None |
| Implementation phase | Rule is workspace-based, not template-specific → works for any domain |
| Checkpoint/resume | `_active_proposer()` reads actual runtime state, not iteration heuristic → robust |
| Multiple files in deliverables | Prefers proposer's configured `_output_path` basename; fallback to first alphabetically |

---

## 6. Testing

### Test File 1 — `test_path_resolution.py`
Path resolution helper unit tests. 12 methods covering all 3 tiers + edge cases + active proposer switching.

Uses `_MockWorkspace` class matching exact API surface + `DualInferencer.__new__()` to bypass init:

```
Tier 1 tests:
  test_T1_returns_deliverable_when_output_md_present
  test_T1_prefers_output_path_basename_over_alphabetical
  test_T1_skips_dotfile_and_falls_through_when_only_dotfile
  test_T1_alphabetical_fallback_when_preferred_basename_absent

Tier 2 tests:
  test_T2_returns_outputs_md_when_no_deliverables
  test_T2_uses_configured_output_path_basename

Tier 3 + edge cases:
  test_T3_returns_None_when_neither_exists
  test_proposer_None_returns_None
  test_proposer_without_workspace_returns_None
  test_proposer_workspace_None_returns_None
  test_proposer_without_output_path_attr_uses_default_basename
  test_after_fix_iteration_resolves_fixer_path
```

### Test File 2 — `test_prompt_builder_feed_dict.py`
Feed dict shape assertions. 5-6 methods intercepting `_render_role_prompt` via mock:

```
  test_followup_feed_includes_main_response_and_prior_output_path
  test_followup_feed_emits_empty_string_when_path_none
  test_followup_omits_reviewer_response_when_review_output_none
  test_review_feed_includes_main_response_and_prior_output_path
  test_review_feed_omits_counter_feedback_when_none
```

### Test File 3 — `test_followup_renders_path_aware.py`
End-to-end mock test with **real Jinja rendering** of `plan/main/followup.jinja2`. 4 methods:

```
  test_rendered_prompt_contains_prior_output_path (cp instruction + populated tags)
  test_rendered_prompt_renders_fallback_when_path_unavailable
  test_rendered_prompt_does_not_contain_literal_None
  test_acceptance_criteria_E1_to_E3
```

### Test File 4 — `test_review_renders_path_aware.py`
Same pattern for `plan/main/review.jinja2`. 2-3 methods.

### Test File 5 — Preflight template variable lock
Parametrized over all 4 target templates. Asserts each references both `main_response` and `prior_output_path`:

```python
@pytest.mark.preflight
@pytest.mark.parametrize("template_rel", TARGETS)
def test_template_references_main_response_and_prior_output_path(template_rel):
    content = Path(template_rel).read_text()
    for var in ("main_response", "prior_output_path"):
        assert re.search(rf"\b{var}\b", content)
```

**Total: ~27 test methods across 5 files. All under 1 second each. No LLM calls.**

---

## 7. Ship Order

| # | What | Risk | Time |
|---|------|------|------|
| 1 | Phase 1+2 (helper + feed dict plumbing) | Low — additive only | 30 min |
| 2 | Phase 3 (update plan/main/followup.jinja2) | Low — backward-compat via if/else | 15 min |
| 3 | Tests (Files 1-5) | Low | 45 min |
| 4 | Phase 4 (sibling templates) | Low — same pattern | 15 min |

**MVP = steps 1-3 (~90 min).** Fixes Bug A + Bug C for the most-bitten template.

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `src/.../flow_inferencers/dual_inferencer.py` | Add `_resolve_prior_proposer_output_path`, modify `_build_followup_prompt` + `_build_review_prompt` feed dicts |
| `src/.../prompt_templates/plan/main/followup.jinja2` | Add `{% if prior_output_path %}` block with cp instruction |
| `src/.../prompt_templates/plan/main/review.jinja2` | Same pattern, reviewer wording |
| `src/.../prompt_templates/implementation/main/followup.jinja2` | Same pattern |
| `src/.../prompt_templates/implementation/main/review.jinja2` | Same pattern |
| `test/.../test_dual_inferencer/test_path_resolution.py` | NEW — 12 unit tests |
| `test/.../test_dual_inferencer/test_prompt_builder_feed_dict.py` | NEW — 5-6 unit tests |
| `test/.../test_dual_inferencer/test_followup_renders_path_aware.py` | NEW — 4 E2E render tests |
| `test/.../test_dual_inferencer/test_review_renders_path_aware.py` | NEW — 2-3 render tests |
| Preflight test file | NEW — 4 parametrized assertions |
