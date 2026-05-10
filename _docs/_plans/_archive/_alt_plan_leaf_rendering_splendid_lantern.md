# Final Plan Comparison: Leaf-Owned Template Rendering

**Plan A**: This file (prior versions were implementation plans; current is comparison-only)
**Plan B**: `_docs/_plans/leaf_owned_template_rendering_INTEGRATED_plan.md` (770 lines, 6 phases)

---

## Status After Multiple Comparison Rounds

Plan B has already incorporated the three critical issues I found in earlier rounds:

| Issue | Found by me | Fixed in Plan B? |
|-------|-------------|-----------------|
| `ainfer()` signature drops `inference_config` — breaks 5 callers | Round 3 | ✅ Fixed in §4.3b — preserves `inference_config` as 2nd positional, new params keyword-only via `*,` after it |
| No reserved-key guard for `feed["input"]` overwrite | Round 2 | ✅ Fixed in §4.3d — `PROTECTED = {"input", "__template_space__"}` with ValueError |
| `extra_feed` leakage into `_ainfer()` | Round 2 | ✅ Fixed in §4.3b — "consumed in `_ainfer_single`, never forwarded" |

---

## ONE Remaining Issue in Plan B

### Phase 1c adds FOLLOWUP_TEMPLATE_DEFAULTS before Dual stops rendering → double-rendering

**Verified by tracing the code:**

1. Phase 1 (§4.3c) adds `"fixer_inferencer": FOLLOWUP_TEMPLATE_DEFAULTS` to Dual's SLOT_DEFAULTS
2. SLOT_DEFAULTS auto-fills `template_key="followup"` on the fixer (confirmed: `apply_to()` check `FIELD_TEMPLATE_KEY not in node` — fixer node currently has no `template_key`)
3. Fixer already receives `template_root_space="plan"` via `_template_root_space` cascade and `template_manager` via `_template_manager` cascade
4. Result: fixer has `template_manager` + `template_root_space="plan"` + `template_key="followup"`
5. **BUT Dual still renders** `plan/main/followup.jinja2` in Phase 1 (via `_render_role_prompt("followup", ...)` at line 1500)
6. Dual passes the rendered string to `fixer_inferencer.ainfer(rendered_string)`
7. Fixer's `_render_prompt()` fires → renders `plan/main/followup.jinja2` AGAIN with `{{ input }} = rendered_string`
8. **Double-rendering**: the template envelope wraps around itself

**Phase 2 (§4.4b) is when Dual stops rendering** — it switches to `extra_feed=feed` path instead of `_render_role_prompt`. Only THEN is the fixer's `template_key="followup"` safe.

**Fix**: Define `FOLLOWUP_TEMPLATE_DEFAULTS` in Phase 1 (the constant), but wire it into SLOT_DEFAULTS in Phase 2 (alongside Dual stopping rendering). They must ship together.

---

## Everything Else in Plan B Is Sound

| Dimension | Assessment |
|---|---|
| Architecture (§3) | Correct — orchestrators assemble feed, leaves render |
| Phase 0 (done) | Verified — path-aware fix shipped, 50 tests passing |
| Phase 0a (audit) | Valuable — re-runnable script for Phase 5 verification |
| Phase 1 (`extra_feed` + `render_only`) | Correct except 1c ordering — easy fix |
| Phase 1d (leaf-level loud failure) | Elegant — detects `default_template` fallback, raises ValueError |
| Phase 2 (Dual migration) | Correct — `_build_*_feed` returns dict, step methods branch on `template_manager` |
| Phase 3 (other orchestrators) | Reasonable scope — MFDual, MultiFlow, audit BTA/PTI |
| Phase 4-5 (deprecation + removal) | Standard pattern, well-structured |
| 5 subtleties (§3.3) | All genuine — audit logging, workflow state, non-templated leaves, orchestrator context, cascade |
| Edge cases (§3.4) | Comprehensive — 2-agent mode, no workspace, missing key, custom formatter, `__new__()` tests |
| 16 acceptance criteria (§7) | Thorough — includes no-silent-failure, reserved-key, nested isolation |
| 13 open questions (§8) | All have recommendations; Q10 (non-templated leaf rejection) and Q11 (reserved key) are particularly important |
| Risk assessment (§6) | Complete — mitigated + residual + out-of-scope |
| Provenance (§9) | Tracks all 3 rounds of comparison + fixes applied |

---

## Recommendation

**If I pick ONE plan: Plan B** (`_docs/_plans/leaf_owned_template_rendering_INTEGRATED_plan.md`).

Plan B is the only real implementation plan. My file (Plan A) has been a comparison document since the last round — it has no phases, no code, no tests. Plan B has 770 lines of implementation-ready detail with 16 acceptance criteria, 13 open questions, risk assessment, and provenance tracking.

Plan B already fixed the three critical issues I found in earlier rounds (ainfer signature, reserved-key guard, leakage prevention). It has ONE remaining issue: Phase 1c ordering bug (FOLLOWUP_TEMPLATE_DEFAULTS wired to SLOT_DEFAULTS before Dual stops rendering). The fix is a one-line change: move the SLOT_DEFAULTS wiring from Phase 1 to Phase 2.

**Action needed on Plan B**: In §4.3c, change from "add to SLOT_DEFAULTS in Phase 1" to "define the constant in Phase 1, wire into SLOT_DEFAULTS in Phase 2 (alongside §4.4b when Dual stops rendering)."
