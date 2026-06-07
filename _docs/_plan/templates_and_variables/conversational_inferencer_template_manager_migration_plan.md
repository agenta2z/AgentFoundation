# ConversationalInferencer → TemplateManager Migration Plan

**Status:** Draft v1 — implementation-ready pending review
**Author:** Rovo Dev — 2026-05-27
**Scope:** AgentFoundation + OpenStartup
**Risk:** MED-HIGH (load-bearing class; every chat turn renders a prompt)
**Companion plans:**
- `workflows_and_sop/sop_framework_UNIFIED_v1_plan.md` (v1.5) — SOP framework that depends on prompt rendering being correct
- `templates_and_variables/load_variables_multidot/load_variables_multidot_INTEGRATED_v4_plan.md` — multidot variable resolution (already landed in TemplateManager)

---

## §-1. Provenance and audit history

| Version | When | Author / trigger | What changed |
|---|---|---|---|
| 1.0 | 2026-05-27 00:17 | Tony surfaced design defect ("ConversationalInferencer does not use TemplateManager?") | Initial plan. Empirical baseline: `ConversationalInferencer` is the only `InferencerBase` subclass with its own bespoke prompt-rendering implementation (`JinjaPromptRenderer`, 212 LoC at `inferencers/agentic_inferencers/conversational/prompt_rendering.py`). All other inferencer subclasses go through `TemplatedInferencerBase` → `TemplateManager` (2,209 LoC at `rich_python_utils/string_utils/formatting/template_manager/`). This plan migrates `ConversationalInferencer` onto `TemplateManager` while preserving SOP rendering (load-bearing for `sop_framework_UNIFIED_v1_plan.md`). |
| 2.0 | 2026-05-27 06:10 | v1 vs Claude#1 plan integration (Tony) | Two plans compared empirically: this plan (v1, adapter pattern) and Claude's plan at `~/.claude/plans/can-you-help-take-ticklish-whisper.md` (direct replacement). At the time, v2 adopted Claude's direct-replacement approach as "strictly cleaner." All §3.4 adapter content deleted; §3.6 mapping table added; §3.7 SOPRegistry path added. See §-1 v3.0 row for why this was partially reversed. |
| 3.0 | 2026-05-27 06:23 | v2 vs Claude#2 plan integration (Tony, second round) | Claude updated their plan at 06:20, AFTER v2 was created. **Claude reversed course** and adopted the adapter pattern, arguing: (a) adapter encapsulates API translation in one 80-LoC file (single update point if TemplateManager API evolves), (b) zero changes to the 18 method-translation call sites inside `_render_prompt()` (bug surface contained), (c) `find_sop_file()` in adapter returns `None` (legacy `_variables/workflow/` already empty; SOPRegistry handles discovery per SOP v1.5). After empirical re-evaluation, v3 accepts (a) and (b) as architecturally superior — the method-translation sites (5 of the 25 in v2) are exactly where bugs hide, and the adapter's encapsulation is a real, durable benefit. **v3 patches:** (A) §3 architecture reverted to adapter pattern (Claude#2 design); (B) §3.4 `TemplateManagerPromptRenderer` adapter source restored (~80 LoC); (C) §3.5 simplified construction-site changes (zero call-site edits in `_render_prompt`); (D) §3.6 mapping table kept but reframed as "adapter implementation cookbook" rather than "inferencer call-site cookbook"; (E) §3.7 SOPRegistry path simplified to "adapter's `find_sop_file()` returns `None`; SOPRegistry handles discovery directly when SOP v1.5 lands" — eliminates the back-compat fallback chain; (F) operational rigor (feature flag R10, phased delete) **preserved** vs Claude#2's "clean cut" — production safety outweighs LoC saved; (G) Claude#2's Q-NEW (existence of `PROMPT_TEMPLATES_ROOT`) added to §8. |
| 3.1 | 2026-05-27 06:46 | Audit pass on v3 (Tony, external reviewer) | Reviewer raised 7 issues against v3. Verified each empirically against the codebase. **5 valid (3 stale-text cleanups, 2 substantive architectural gaps); 2 rejected:** (1) Issue 3 (remove feature flag) — REJECTED, no such directive exists in this session; v3 preserves R10 feature flag for production safety. (2) Issue 6 (`hasattr(raw, "path")` returns False) — **VALID HIGH**, empirically confirmed: `_OriginTaggedStr.__slots__ = ("_origin_root",)` — no `.path` attribute exists. Adapter's `template_config` silently returns `{}`, losing YAML sidecar config (tool whitelisting, structural XML escaping). **Fix:** use `_origin_root` attribute (which IS present) to reconstruct path as `Path(raw._origin_root) / template_key_to_relpath(template_key)`. See §3.4 patched. (3) Issue 7 (`template_formatter` is `jinjia_template_format` module-level Callable, not TemplateManager's Environment) — **VALID MED**, empirically confirmed line 296. Templates referencing macros/includes/filters would break in feed self-resolution path. **Fix:** route adapter's `render_string` through `TemplateManager.__call__(formatter=...)` with the raw template content, which reuses TemplateManager's full feed-merging and predefined-variable pipeline. See §3.4 patched. (4) Issues 1, 2, 4, 5: stale §7.1 test file, stale §4 PR-ordering paragraph, stale Q1/Q3, stale AC-EQ4 — all simple text cleanups. |
| 3.2 | 2026-05-27 07:08 | Audit pass on v3.1 (Tony, external reviewer) | Reviewer raised 7 issues against v3.1. Verified each empirically. **All 7 valid; all applied as in-place v3.2 patches:** (1) §3.6 row for line 747 stale (still describes v3 `template_formatter` path) — rewrote to reference transient-key pattern. (2) §3.4 "Key properties" paragraph stale (still describes `Path(raw.path)` design) — rewrote to reference `_origin_root` reconstruction. (3) R4 references nonexistent `tm.get_template_config()` — rewrote to reference module-level `_load_yaml_cascade()` helper. (4) R7 references removed Phase 1 (`PR-1 get_active_sop_path`) — rewrote to reference PR-2 as starting point. (5) §9 Recommendation references nonexistent "PR-1 RichPythonUtils" — corrected to "no RichPythonUtils changes required." (6) **VALID MED:** §4 Phase 5 wording "remove `prompt_renderer` field" contradicts v3's zero-call-site design — if field is removed, all 25 `self.prompt_renderer.*` call sites break. **Fix:** Phase 5 now only deletes `prompt_rendering.py` + removes feature flag scaffolding; `prompt_renderer` field stays permanently (pointing to the adapter). (7) **VALID MED:** R-NEW1 transient-key registration needs empirical verification that TM lookups are not cached. **Verified during this audit:** `grep "lru_cache\|@cache\|cached_template" template_manager.py` returns empty; `__call__` does runtime dict lookup against `self.templates` (template_manager.py:1429-1480). Runtime mutation IS safe. R-NEW1 mitigation updated with this empirical confirmation + Phase 0 verification test added to AC list. |
| 3.3 | 2026-05-27 07:21 | Audit pass on v3.2 — AC-EQ1 specification gap (Tony, external reviewer) | Reviewer raised 1 substantive issue: AC-EQ1 says "representative feed dict" without specification. **Empirically verified:** `_render_prompt` constructs 11 distinct feed categories (`conversational_inferencer.py:719-730`); a test feed missing 5 of them could pass while production rendering breaks (e.g., SOP self-resolution path at line 743 only fires when `feed[active_sops][i].next_step_guidance` contains `{{ session_root_path }}` — silently skipped if `active_sops` is empty). **VALID MED.** Also empirically verified: `test_sop_prompt_integration.py` exists (10 tests, 272 LoC) at both AgentFoundation and rankevolve paths — this is a stronger real-world verification gate than any synthetic AC could be. **v3.3 patches applied:** (a) AC-EQ1 strengthened with full 11-category feed-key table + 5 explicit test fixtures (F1–F5) each exercising different `{% if %}` branches; (b) new AC-EQ-INTEG1 added: the existing `test_sop_prompt_integration.py` 10-test suite must pass unchanged with the new adapter — Phase 0 baselines under legacy; Phase 2 reruns under feature flag. Reviewer's overall assessment ("the tests are conceptually sufficient") confirmed — the gap was specification rigor, not test coverage strategy. |

---

## §1. Problem statement — why this matters

### §1.1 Two parallel template systems

AgentFoundation has **two independent prompt-rendering implementations** that solve the same problem in different ways:

| Concern | `TemplateManager` (RichPythonUtils) | `JinjaPromptRenderer` (AgentFoundation) |
|---|---|---|
| **Location** | `rich_python_utils/string_utils/formatting/template_manager/template_manager.py` (2,209 LoC) | `agent_foundation/common/inferencers/agentic_inferencers/conversational/prompt_rendering.py` (212 LoC) |
| **Used by** | Every `TemplatedInferencerBase` subclass (PTI, BTA, LWI, Dual, `ConversationalFlowNodeAdapter`, ~12 leaf inferencers) | `ConversationalInferencer` ONLY |
| **Multi-root overlay** | ✅ First-write-wins overlay across N template roots | ❌ Single `template_dir` |
| **Template versioning** | ✅ `template_version="enterprise"` etc. with `.<version>` suffix lookup | ❌ Hardcoded `template_path` |
| **Variable cascade** | ✅ Full `load_variables()` with multidot, alias, `_variables/` folder cascade | ⚠ Partial — uses `FileBasedVariableManager` (same backend) but custom resolution path |
| **Root-space namespacing** | ✅ `active_template_root_space` (e.g., `"action_agent"`) | ❌ Implicit in `template_path` string |
| **Active-type switching** | ✅ `active_template_type` (e.g., `"main"`, `"reflection"`) | ❌ Implicit in `template_path` string |
| **YAML sidecar config** | ✅ via `template_config` semantics | ✅ via `template_config` semantics (similar but parallel impl) |
| **SOP discovery** | ❌ No `find_sop_file()` (added in JinjaPromptRenderer) | ✅ `find_sop_file()` returns `_variables/workflow/sop.{jinja2,j2,md,yaml,yml}` |
| **String render path** | ✅ via `template_manager(template_string=...)` overload | ✅ `render_string(template_str, context)` |

The two systems **duplicate functionality**, and `JinjaPromptRenderer` adds one capability (`find_sop_file`) that doesn't exist in `TemplateManager`. The duplication has caused real, observable problems documented below.

### §1.2 Observed problems caused by the duplication

1. **No multi-root overlay** — `ConversationalInferencer` cannot consume an enterprise/consumer template overlay; every other inferencer can. This breaks the established A/B testing + customer-segment pattern.
2. **No version suffix support** — `ConversationalInferencer` cannot use `template_version="end_customers"`-style routing; every other inferencer can.
3. **Standalone test runs lose SOP context** — when called outside OpenStartup's factory (e.g., a CLI test of `role_creation`), `ConversationalInferencer.prompt_renderer=None` triggers `_render_fallback_prompt()` (a bare-bones template at line 759), which has **no SOP awareness at all**. The LLM doesn't even know it's running an SOP. This is the immediate motivation for the migration: SOP test runs (gated by `sop_framework_UNIFIED_v1_plan.md` v1.5) need first-class prompt rendering even in standalone mode.
4. **Parallel YAML/variable resolution** — `JinjaPromptRenderer.template_variables` reimplements ~80 LoC of variable-cascade logic that `TemplateManager.load_variables()` already does correctly. Maintaining both increases drift risk.
5. **Architectural inconsistency** — `ConversationalInferencer` is the ONLY `InferencerBase` subclass not on `TemplateManager`. This violates the convention every other subclass follows and makes the codebase harder to reason about.

### §1.3 Why it was originally done this way (honest pre-mortem)

Reading `prompt_rendering.py` carefully, two motivations are evident:

- (a) **`find_sop_file()` did not exist in `TemplateManager`.** The author needed a way to discover the active SOP markdown file from `_variables/workflow/sop.*`, and rather than extending `TemplateManager`, they built a parallel renderer with that one extra method.
- (b) **`ConversationalInferencer` extends `InferencerBase` directly (not `TemplatedInferencerBase`).** This was probably to avoid bringing in BTA/PTI orchestration machinery that's unrelated to conversational use. But `TemplatedInferencerBase` is decomposable — its prompt-rendering responsibility can be reused without inheriting its full lifecycle.

Both motivations are addressable without forking the template system. See §3 design.

---

## §2. Goals and non-goals

### §2.1 Goals

- **G1.** `ConversationalInferencer` uses `TemplateManager` for all prompt rendering (initial.jinja2 + any other templates added in the future).
- **G2.** All current `JinjaPromptRenderer` behavior is preserved (variable cascade, YAML sidecar config, render_string for feed self-resolution). **SOP rendering** is preserved by delegating to the existing `SOPRegistry` at `resources/sops/` (no longer via `find_sop_file()`) — this aligns with `sop_framework_UNIFIED_v1_plan.md` v1.5 §3.5 registry pattern.
- **G3.** `JinjaPromptRenderer` is removed (no parallel system).
- **G4.** No existing OpenStartup behavior changes (regression test pass).
- **G5.** Standalone `ConversationalInferencer` construction (no `template_manager` arg, no `session_context` injection) renders the full `initial.jinja2` with SOP context — fixing the "fallback prompt has no SOP awareness" defect that motivated this plan.
- **G6.** ~~`TemplateManager` gains a first-class `find_sop_file()` method.~~ **REMOVED in v2.** Empirical finding: `resources/sops/` already exists in both AgentFoundation and OpenStartup (verified via `ls`). The SOP framework v1.5 plan calls for a SOPRegistry-based discovery mechanism (v1.5 §3.5). Adding `find_sop_file()` to TemplateManager would be a parallel discovery system to SOPRegistry — exactly the kind of duplication this plan exists to eliminate.
- **G7.** All 25 references to `prompt_renderer` in `conversational_inferencer.py` (lines 103, 131, 356-357, 587, 633, 638, 735, 747, 753, 755, 757, 1086, etc. — empirically counted via `grep | wc -l`) are migrated to the new `template_manager` + `template_key` fields. The migration is mechanical for most references (replace attribute name); 6 references involve method-call translation per the API mapping table in §3.6.

### §2.2 Non-goals

- **N1.** Do NOT change `TemplatedInferencerBase` itself. `ConversationalInferencer` continues to extend `InferencerBase` directly (not `TemplatedInferencerBase`); only the prompt-rendering concern is migrated.
- **N2.** Do NOT change the `initial.jinja2` template body or any `_variables/` files. The migration is render-engine-only.
- **N3.** Do NOT migrate any other AgentFoundation inferencer (PTI/BTA/LWI/Dual/etc.) — they're already on `TemplateManager` correctly.
- **N4.** Do NOT introduce a new abstract `PromptRenderer` protocol class — the existing duck-typed `prompt_renderer: Any` field continues to be the public contract; only the default implementation changes.

---

## §3. Target architecture (v3 — adapter pattern, encapsulated)

### §3.1 The unified design

After migration:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ConversationalInferencer (extends InferencerBase, unchanged)        │
│                                                                     │
│ Field (UNCHANGED): prompt_renderer: Any = None  (duck-typed)        │
│                                                                     │
│ Default construction (in __attrs_post_init__):                      │
│   if prompt_renderer is None:                                       │
│     self.prompt_renderer = TemplateManagerPromptRenderer(           │
│       template_manager=TemplateManager(                             │
│         templates=PROMPT_TEMPLATES_ROOT,                            │
│         active_template_root_space="conversation",                  │
│         active_template_type="main",                                │
│         predefined_variables=True,                                  │
│         enable_templated_feed=True,                                 │
│       ),                                                            │
│       template_key="initial",                                       │
│     )                                                               │
│                                                                     │
│ ZERO changes to the 18 method-translation call sites inside         │
│ _render_prompt() — they continue calling self.prompt_renderer.*.    │
│ All API translation is encapsulated in the adapter (§3.4).          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TemplateManagerPromptRenderer (~80 LoC, NEW — §3.4)                 │
│                                                                     │
│ Thin adapter implementing the 7 duck-typed members ConversationalI- │
│ nferencer reads from `prompt_renderer`:                             │
│  - render(feed) → tm(template_key, **feed)                          │
│  - render_string(s, ctx) → tm.template_formatter(s, **ctx)          │
│  - template_variables → tm.load_variables(template_key)             │
│  - variable_manager → tm._variable_loader                           │
│  - template_source → str(tm.get_raw_template(template_key))         │
│  - template_config → loads .config.yaml cascade (§3.4)              │
│  - find_sop_file() → returns None (SOP v1.5 SOPRegistry owns it)    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TemplateManager (RichPythonUtils — UNCHANGED)                       │
│                                                                     │
│ All needed capabilities already exist (empirically verified):       │
│  - __call__(template_key, **feed) → rendered string                 │
│  - template_formatter (Callable) → string render path               │
│  - _variable_loader → underlying FileBasedVariableManager           │
│  - load_variables(template_key) → variable cascade                  │
│  - get_raw_template(template_key) → template source                 │
│  - predefined_variables / enable_templated_feed init flags          │
│                                                                     │
│ NO new methods added (G6 dropped — see §3.7 for SOP discovery).     │
└─────────────────────────────────────────────────────────────────────┘
```

### §3.2 Why adapter (NOT direct field replacement)? — v3 reversal

v2 of this plan proposed direct field replacement (replace `prompt_renderer: Any` with `template_manager: TemplateManager + template_key: str`), eliminating the adapter as "strictly cleaner." **v3 reverts that decision** after Claude (in their 06:20 plan update) argued persuasively for the adapter pattern. The honest reasoning for the reversal:

1. **Bug-surface containment.** Of the 25 call sites in v2's mapping table, 5 require method-call translation (lines 357, 633, 638, 735+747, 753, 755, 757). These are where bugs hide — subtle semantic differences between `prompt_renderer.template_variables` (lazy dict) and `tm.load_variables(key)` (eager-resolved dict); between `prompt_renderer.render_string(s, ctx)` (positional args) and `tm.template_formatter(s, **ctx)` (kwargs); etc. **The adapter absorbs these 5 translation points into ONE 80-LoC file** that can be reviewed, tested in isolation (AC-EQ1-7), and audited as a single unit. With direct replacement, those translations are scattered across 5 helper methods + inline rewrites in the most load-bearing file in the system.

2. **Single update point for TemplateManager API evolution.** TemplateManager is a 2,209-LoC class under active development (`enable_templated_feed`, `predefined_variables`, the templated-feed self-resolution path were all added in recent revisions). If TemplateManager's `__call__` signature or `_variable_loader` semantics evolve, the adapter is one file to update — vs touching `conversational_inferencer.py` (the most load-bearing class) again.

3. **The duck-typed contract is a *useful* internal boundary.** v2 argued "no external consumer reads `prompt_renderer.*` so it's not a public API surface." That's true literally, but architecturally `prompt_renderer` is a *test seam* (mock-friendly) and an *encapsulation boundary*. Preserving the seam keeps `ConversationalInferencer` testable without instantiating a full TemplateManager.

**Honest cost accounting** (correcting v2's claim that direct replacement is "fewer total LoC"):

| Approach | New code | Modified code | Deleted code | Net |
|---|---|---|---|---|
| v3 adapter | +80 (adapter) | ~5 (factory + `__attrs_post_init__`) | -212 (prompt_rendering.py) | **-127 LoC + clean test seam** |
| v2 direct | +60 (helpers in `conversational_inferencer.py`) | ~25 (call-site edits) | -212 (prompt_rendering.py) | -127 LoC + 25 edits to load-bearing file |

**Net LoC is identical (-127 either way).** The real difference is *where the complexity lives*: adapter (concentrated in one new file, easy to review) vs direct (distributed across 25 edits in the most critical file). For a load-bearing class touched in every chat turn, **concentrated complexity is safer**.

Source of v3's correction: Claude plan at `~/.claude/plans/can-you-help-take-ticklish-whisper.md` (updated 2026-05-27 06:20, lines 7-13). Verified empirically — the v2 honest-cost analysis was wrong; LoC is identical, and risk profile favors the adapter.

### §3.4 `TemplateManagerPromptRenderer` — the adapter (v3 RESTORED)

New file: `agent_foundation/common/inferencers/agentic_inferencers/conversational/template_manager_renderer.py` (~80 LoC).

```python
"""TemplateManagerPromptRenderer — adapter that exposes the duck-typed
`prompt_renderer` API consumed by ConversationalInferencer, backed by
TemplateManager from RichPythonUtils.

This adapter is the SOLE seam between ConversationalInferencer and
TemplateManager. All API translation (signature differences, semantic
nuances, default-argument application) lives here, NOT in
ConversationalInferencer's render path. See migration plan §3.2 for the
adapter-vs-direct-replacement rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from attr import attrib, attrs

from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    TemplateManager,
)


def _load_yaml_cascade(candidates: list) -> dict:
    """Load the first existing YAML file from the cascade; merge order is
    'first-wins' — earlier entries take precedence.

    Matches `JinjaPromptRenderer._load_yaml_candidates` semantics.
    """
    for candidate in candidates:
        if candidate.is_file():
            try:
                with open(candidate) as fh:
                    loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                continue
    return {}


@attrs(slots=False, eq=False)
class TemplateManagerPromptRenderer:
    """Duck-typed wrapper that exposes the 7-method `prompt_renderer` API.

    Members consumed by ConversationalInferencer (verified via grep — see
    migration plan §3.6 for the full mapping):
      - render(feed: Mapping) -> str
      - render_string(template_str: str, context: Mapping) -> str
      - template_variables: dict           (property)
      - variable_manager: Any              (property)
      - template_source: str               (property)
      - template_config: dict              (property)
      - find_sop_file() -> Optional[Path]
    """

    template_manager: TemplateManager = attrib()
    template_key: str = attrib(default="initial")

    # ---- core rendering -------------------------------------------------

    def render(self, feed: Mapping[str, Any]) -> str:
        """Render the active template with the given feed dict.

        Maps to TemplateManager.__call__(template_key, **feed).
        """
        return self.template_manager(self.template_key, **dict(feed))

    def render_string(self, template_str: str, context: Mapping[str, Any]) -> str:
        """Render an arbitrary Jinja2 template string in the same env.

        Used by ConversationalInferencer's feed self-resolution path
        (conversational_inferencer.py:735+747).

        **v3.1 fix (R-NEW1):** v3 initially called `self.template_manager.
        template_formatter(template_str, **context)` directly. Empirically
        (verified against template_manager.py:296), `template_formatter` is the
        module-level `jinjia_template_format` Callable, NOT the TemplateManager's
        internal Environment. Templates referencing shared macros, filters, or
        `{% include %}` directives would break.

        Correct path: route through `TemplateManager.__call__()` so that the
        full feed-merging + predefined-variable + variable-cascade pipeline is
        applied. Since `__call__` operates on registered `template_key`s (not
        raw strings), we register the raw string under a transient synthetic
        key, render, then drop the registration.
        """
        if not template_str:
            return template_str
        # Synthesize a per-call transient key; isolated from real templates.
        # The leading "__transient__/" prefix prevents collision with user keys.
        transient_key = f"__transient__/{id(template_str):x}"
        try:
            # TemplateManager.templates is the in-memory dict of registered
            # template content. Inject the raw string under the transient key.
            if self.template_manager.templates is None:
                self.template_manager.templates = {}
            # NOTE: TemplateManager's `templates` is hierarchical (root_space/type).
            # The transient key lives in a synthetic "__transient__" root_space.
            tmplates = self.template_manager.templates
            if "__transient__" not in tmplates:
                tmplates["__transient__"] = {}
            tmplates["__transient__"][transient_key.split("/", 1)[1]] = template_str
            return self.template_manager(
                transient_key,
                active_template_root_space="__transient__",
                active_template_type="",
                **dict(context),
            )
        finally:
            # Always clean up the transient registration.
            try:
                tmplates["__transient__"].pop(transient_key.split("/", 1)[1], None)
            except Exception:
                pass

    # ---- variable surface -----------------------------------------------

    @property
    def template_variables(self) -> dict:
        """Resolved variable cascade (YAML sidecar + `_variables/` folder)."""
        return self.template_manager.load_variables(self.template_key) or {}

    @property
    def variable_manager(self):
        """Expose the underlying VariableManager (FileBasedVariableManager).

        Used by widgets to call .set(key, val) for prompt-variable overrides.
        """
        return getattr(self.template_manager, "_variable_loader", None)

    # ---- template introspection -----------------------------------------

    @property
    def template_source(self) -> str:
        """Raw Jinja2 source of the active template."""
        raw = self.template_manager.get_raw_template(self.template_key)
        return str(raw)

    @property
    def template_config(self) -> dict:
        """YAML sidecar config adjacent to the active template.

        Cascade order (matches JinjaPromptRenderer semantics):
          1. .<basename>.config.yaml (highest)
          2. .config.yaml (fallback)
        Returns {} if neither exists.

        **v3.1 fix (R-NEW2):** v3 initially used `Path(raw.path) if hasattr(
        raw, "path") else None`. Empirically (verified against template_manager.py:
        161-181), `_OriginTaggedStr.__slots__ = ("_origin_root",)` — there is NO
        `.path` attribute. The `hasattr` check always returned False; sidecar
        config silently went missing, losing tool whitelisting, structural XML
        escaping, and other config-driven behavior.

        Correct path: reconstruct the file path from `_origin_root` (which IS
        present on `_OriginTaggedStr`) + the template_key's path mapping.
        TemplateManager loads templates from `<root>/<root_space>/<type>/<key>.<ext>`
        per its standard cascade (template_manager.py:1034 +
        _resolve_template_space_key_with_root_space_and_type).
        """
        raw = self.template_manager.get_raw_template(self.template_key)
        if raw is None:
            return {}
        # Reconstruct path from origin tagging — _origin_root is the templates
        # root directory; template_key is "<key>" within active_template_root_space
        # + active_template_type folders.
        origin_root = getattr(raw, "_origin_root", None)
        if origin_root is None:
            return {}
        root_space = self.template_manager.active_template_root_space or ""
        ttype = self.template_manager.active_template_type or ""
        # Try common extensions; the actual file used by TemplateManager
        # for this template_key has one of these.
        for ext in (".jinja2", ".j2", ".md", ".yaml", ".yml"):
            template_path = (
                Path(origin_root) / root_space / ttype / f"{self.template_key}{ext}"
            )
            if template_path.is_file():
                return _load_yaml_cascade(
                    [
                        template_path.parent / f".{template_path.stem}.config.yaml",
                        template_path.parent / ".config.yaml",
                    ]
                )
        return {}

    # ---- SOP discovery (delegated to SOPRegistry per SOP v1.5) ----------

    def find_sop_file(self) -> Optional[Path]:
        """SOP file discovery is owned by SOPRegistry (SOP v1.5 §3.5).

        Returns None unconditionally; ConversationalInferencer's render path
        (conversational_inferencer.py:638) handles None gracefully by reading
        from workflow_manager's active instance instead. See migration plan
        §3.7 for the SOPRegistry path.

        This method exists only to preserve the duck-typed prompt_renderer
        contract; it can be removed in a follow-up after the duck-typed call
        site at line 638 is updated to read from workflow_manager directly.
        """
        return None
```

**Key properties of this adapter:**

- **Slots disabled (`slots=False`)** to match `TemplateManager`'s declaration and avoid attrs slot-mismatch with the `eq=False` interaction.
- **No state beyond `template_manager + template_key`** — the adapter is a pure pass-through.
- **`find_sop_file()` returns `None` unconditionally** — see §3.7 for why this is correct.
- **`template_config` cascade** *(v3.1)* reconstructs the template file path from `_OriginTaggedStr._origin_root` + `active_template_root_space` + `active_template_type` + `template_key` + extension probe (matches TM's loader cascade), then reads `.<basename>.config.yaml` → `.config.yaml` cascade via module-level `_load_yaml_cascade()`. Matches `JinjaPromptRenderer._load_yaml_candidates` semantics exactly. (v3 initial design used `Path(raw.path)` which always returned None — see §-1 v3.1 audit row.)

### §3.6 API mapping table (cookbook) — what the adapter implements

This table is **the adapter's contract**, not a ConversationalInferencer migration guide (v3 reverts v2's framing). Each row maps a `prompt_renderer.*` member to its TemplateManager-backed implementation inside the adapter. **Call sites in ConversationalInferencer do NOT change.**

| Call site (line) — UNCHANGED | What it calls today | Adapter's implementation (delegates to TemplateManager) |
|---|---|---|
| 103 (attrib def) | `prompt_renderer: Any = attrib(default=None, kw_only=True)` | UNCHANGED (adapter satisfies the duck-typed contract) |
| 131 (`supports_prompt_rendering`) | `return self.prompt_renderer is not None` | UNCHANGED |
| 357 (`variable_manager` access) | `vm = getattr(self.prompt_renderer, "variable_manager", None)` | UNCHANGED; adapter's `.variable_manager` property → `tm._variable_loader` |
| 587 (renderer check) | `if not self.prompt_renderer:` | UNCHANGED |
| 633 (`template_variables`) | `getattr(self.prompt_renderer, "template_variables", {})` | UNCHANGED; adapter's `.template_variables` property → `tm.load_variables(template_key)` |
| 638 (`find_sop_file`) | `sop_path = getattr(self.prompt_renderer, "find_sop_file", lambda: None)()` | UNCHANGED; adapter's `find_sop_file()` returns `None`. ConversationalInferencer falls through to its existing `workflow_manager`-based SOP discovery (line 638's `getattr` fallback already handles `None`). See §3.7. |
| 735 (`hasattr render_string`) | `if hasattr(self.prompt_renderer, "render_string"):` | UNCHANGED; adapter exposes `render_string`, so this branch is taken |
| 747 (`render_string` call) | `render_template=self.prompt_renderer.render_string` | UNCHANGED; adapter's `.render_string(s, ctx)` routes through `tm(transient_key, active_template_root_space="__transient__", **ctx)` via transient-key registration per §3.4 v3.1 fix — preserves TM's Environment, macros, includes, predefined vars |
| 753 (`template_source`) | `self._last_template_source = self.prompt_renderer.template_source` | UNCHANGED; adapter's `.template_source` property → `str(tm.get_raw_template(template_key))` |
| 755 (`template_config`) | `getattr(self.prompt_renderer, "template_config", {})` | UNCHANGED; adapter's `.template_config` property → loads `.config.yaml` cascade |
| 757 (`render`) | `return self.prompt_renderer.render(feed)` | UNCHANGED; adapter's `.render(feed)` → `tm(template_key, **feed)` |
| 1086 (renderer check) | `if self.prompt_renderer:` | UNCHANGED |
| 1089, 1184, 1189, 1279, 1379, 1382 (variable_manager) | `self.prompt_renderer.variable_manager` | UNCHANGED; adapter's `.variable_manager` property is used directly |

**The "Type" column from v2 is gone in v3.** Every row is "UNCHANGED" — that's the *point* of the adapter pattern. The translation work happens once in `TemplateManagerPromptRenderer` (§3.4), not 25 times across `conversational_inferencer.py`.

**Net edit to `conversational_inferencer.py`:** only `__attrs_post_init__` to default-construct the adapter when `prompt_renderer is None` (~15 LoC) + delete `_render_fallback_prompt()` (~25 LoC). Total: ~10 LoC net reduction.

**Helper methods added to ConversationalInferencer: ZERO in v3.** All translation lives in the adapter (§3.4). v2's 5 helpers are not needed — see §-1 v3.0 audit row for rationale.

### §3.7 SOP discovery — adapter returns `None`; SOPRegistry owns it (v3 simplified)

**v1 proposed adding `find_sop_file()` to TemplateManager (+20 LoC). v2 replaced that with `WorkflowManager.get_active_sop_path()` + a back-compat fallback chain (+30 LoC). v3 simplifies to a one-line stub.**

**Why v3 is simpler:**

1. The adapter's `find_sop_file()` returns `None` unconditionally (see §3.4).
2. `ConversationalInferencer:638` reads `find_sop_file` via `getattr(self.prompt_renderer, "find_sop_file", lambda: None)()` — meaning it already gracefully handles a `None`-returning implementation. **No call-site change needed.**
3. When SOP v1.5 lands, SOP-related rendering moves to a separate `WorkflowManager.get_active_sop_path()` call path inside the `_render_prompt` workflow-context-injection block (lines 633-665 area). That call lives in the *SOP migration*, not this one.
4. Legacy `_variables/workflow/sop.{jinja2,j2,md,yaml,yml}` files in the current repo are **already empty** (verified via grep — the role_creation SOP at `_variables/workflow_sop/role_creation.md` is in a different subdirectory and is read by `WorkflowRegistry.load_all`, not by `JinjaPromptRenderer.find_sop_file`). So returning `None` is behaviorally equivalent today.

**Net result for v3:**

- **No new method on TemplateManager** (eliminates G6 from v1).
- **No new method on WorkflowManager** (eliminates v2's `get_active_sop_path` + back-compat fallback chain — that becomes part of the SOP v1.5 migration, not this one).
- **No back-compat fallback shim** — adapter just returns `None`.

This cleanly separates the **template-rendering migration** (this plan) from the **SOP discovery migration** (SOP v1.5 plan). They no longer entangle.

**Coordination contract with SOP v1.5:** when SOP v1.5 Phase 4 lands the `WorkflowManager.get_active_sop_path()` method, the SOP rendering block in `ConversationalInferencer._render_prompt` (lines 633-665) will be updated to call it directly (bypassing `prompt_renderer.find_sop_file` entirely). That is a SOP v1.5 task, not this one. This plan only ensures the adapter's `find_sop_file()` returns `None` so the current call site (line 638) continues to behave correctly until then.

### §3.8 ~~Old `TemplateManager.find_sop_file()` section — DELETED in v2 (still deleted in v3)~~

v1 §3.3 proposed adding `find_sop_file()` to `TemplateManager`. v2 deleted that proposal. v3 confirms — discovery moves to `SOPRegistry` per §3.7. `TemplateManager` remains unmodified.

### §3.9 ~~Old "DELETED in v2" marker for the adapter~~ — REVERSED in v3

v2 marked the `TemplateManagerPromptRenderer` adapter as deleted. **v3 restores the adapter** per §3.4 above. See §-1 v3.0 audit row for the reversal rationale.

### §3.10 Construction-site changes (v3 — adapter restored)

#### §3.10.1 OpenStartup factory (factories.py)

Today (`OpenStartup/.../backends/factories.py:151-165`):

```python
from agent_foundation.common.inferencers.agentic_inferencers.conversational.prompt_rendering import (
    JinjaPromptRenderer,
)

prompt_renderer = JinjaPromptRenderer(
    template_dir=str(template_dir),
    template_path="conversation/main/initial.jinja2",
    cross_space_root=str(template_dir),
)

# Later passed to ConversationalInferencer:
conv_inferencer = ConversationalInferencer(
    prompt_renderer=prompt_renderer,
    ...,
)
```

After (v3 — adapter pattern):

```python
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    TemplateManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
    TemplateManagerPromptRenderer,
)

template_manager = TemplateManager(
    templates=str(template_dir),
    active_template_root_space="conversation",
    active_template_type="main",
    predefined_variables=True,    # enables _variables/ folder cascade
    enable_templated_feed=True,   # enables feed-self-resolution (replaces render_string usage)
)

prompt_renderer = TemplateManagerPromptRenderer(
    template_manager=template_manager,
    template_key="initial",       # matches the old "conversation/main/initial.jinja2" path
)

# Pass through the unchanged duck-typed field:
conv_inferencer = ConversationalInferencer(
    prompt_renderer=prompt_renderer,
    ...,
)
```

**Net change:** +5 LoC at the factory (replaces `JinjaPromptRenderer` import + construction with `TemplateManager` + adapter; same call shape downstream).

#### §3.10.2 ConversationalInferencer default construction (G5 — standalone defect fix)

Today (`conversational_inferencer.py:103`):

```python
prompt_renderer: Any = attrib(default=None, kw_only=True)  # PromptRenderer
```

After (v3 — field unchanged; `__attrs_post_init__` auto-builds adapter):

```python
prompt_renderer: Any = attrib(default=None, kw_only=True)  # UNCHANGED

def __attrs_post_init__(self) -> None:
    super().__attrs_post_init__()
    if self.prompt_renderer is None:
        # G5 fix — standalone construction (CLI tests, SOP executor) gets
        # full prompt rendering without needing to construct a renderer
        # externally. The default adapter points at the in-repo conversation
        # templates via a freshly-constructed TemplateManager.
        from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
            TemplateManager,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
            TemplateManagerPromptRenderer,
        )
        from agent_foundation.resources import PROMPT_TEMPLATES_ROOT  # verified per §8 Q-NEW

        self.prompt_renderer = TemplateManagerPromptRenderer(
            template_manager=TemplateManager(
                templates=str(PROMPT_TEMPLATES_ROOT),
                active_template_root_space="conversation",
                active_template_type="main",
                predefined_variables=True,
                enable_templated_feed=True,
            ),
            template_key="initial",
        )
```

This **fixes the standalone-test-run defect** (§1.2 problem 3) — `prompt_renderer` is never `None` after construction; `_render_fallback_prompt()` becomes dead code that can be deleted (see §4 Phase 3).

#### §3.10.3 SOP executor (standalone path)

Today: SOP executor constructs `ConversationalInferencer(prompt_renderer=None)` → falls through to `_render_fallback_prompt` → no SOP context.

After: SOP executor constructs `ConversationalInferencer()` → `__attrs_post_init__` auto-creates the adapter per §3.10.2 → full SOP context.

**Zero code change needed in the SOP executor** — the fix lives entirely in `ConversationalInferencer`'s default construction. SOP executor authors don't need to know about `TemplateManagerPromptRenderer` at all; the adapter is an implementation detail.

#### §3.10.4 ~~`set_prompt_variable.py` migration (Claude#1 §6)~~ — REJECTED in v2 (still rejected in v3)

Claude's plan proposed updating `OpenStartup/.../effects/set_prompt_variable.py:29`. Empirically verified (`find ... -name "set_prompt_variable.py"`): **this file does not exist** in either AgentFoundation or OpenStartup. Claim rejected; no change needed.

#### §3.10.5 ~~`flow_node_adapter.py` migration (Claude#1 §6)~~ — REJECTED in v2 (still rejected in v3)

Claude's plan proposed checking `flow_node_adapter.py` for `prompt_renderer` references. Empirically verified (`grep prompt_renderer flow_node_adapter.py`): **zero matches**. The file does exist but does not pass `prompt_renderer` through. Claim rejected; no change needed.

---

_(End of §3 — v3 architecture fully specified across §3.1-§3.10. The adapter pattern is restored; the duck-typed `prompt_renderer` field is preserved; the call sites inside `_render_prompt()` are untouched.)_


---

## §4. Phased rollout

| Phase | What | Risk | Reversible? | Effort (LoC) |
|---|---|---|---|---|
| **0** | Pre-flight: write RED tests for behavioral equivalence per §5.1 mapping table (§3.6) — all 25 call sites covered | none | n/a | +250 (tests) |
| **1** | ~~Add `WorkflowManager.get_active_sop_path()`~~ — **REMOVED in v3** (deferred to SOP v1.5 plan; adapter returns `None` and current call site at line 638 handles `None` gracefully per §3.7) | n/a | n/a | 0 |
| **2** | Create `TemplateManagerPromptRenderer` adapter (§3.4) in new file; add `__attrs_post_init__` to ConversationalInferencer that default-constructs the adapter when `prompt_renderer is None`; gate behind `OPENTEAM_USE_TEMPLATE_MANAGER_RENDERER=1` env (R10 feature flag) | LOW (deactivated by default) | trivial — env flag off | +80 (adapter) + ~15 (`__attrs_post_init__`) |
| **3** | Delete `_render_fallback_prompt()` from `conversational_inferencer.py` (~25 LoC; no longer reachable since `prompt_renderer` is never `None` after `__attrs_post_init__`). **Zero changes to the 25 call sites** inside `_render_prompt()` — adapter satisfies their contract. | LOW (only dead-code deletion + standalone path enhancement) | revert restores dead code | -25 |
| **4** | Flip `OPENTEAM_USE_TEMPLATE_MANAGER_RENDERER=1` default-on in dev; soak 1 week; then update OpenStartup `factories.py` to construct `TemplateManagerPromptRenderer(template_manager=...)` per §3.10.1 | MED (every OpenStartup chat session affected) | env flag off | +5 net |
| **5** *(v3.2 fix)* | Delete `JinjaPromptRenderer` class + file (`prompt_rendering.py`); remove feature flag scaffolding. **DO NOT remove the `prompt_renderer` field** on `ConversationalInferencer` — it permanently holds the adapter instance. Removing the field would break all 25 `self.prompt_renderer.*` call sites, contradicting the adapter design (§3.1). | LOW (no callers after Phase 3+4 soak; field stays permanent) | revert restores file | -212 |
| **6** | Run regression suite + manual smoke test (CLI + OpenStartup) | none (verification only) | n/a | 0 |

Total effort: ~250 LoC of tests + ~120 LoC of new code − 250 LoC removed = **net −130 LoC of production code**, ~250 LoC of new tests, ~3 days of focused work.

PR ordering (v3.1 — updated, since Phase 1 was removed and 25 call-site migration is no longer needed): **PR-2 (create adapter + `__attrs_post_init__`, feature-flagged off-by-default)** can land first. **PR-3 (delete `_render_fallback_prompt`)** depends on PR-2's `__attrs_post_init__` landing. **PR-4 (OpenStartup factories.py + flip flag default-on, soak 1 week)** depends on PR-2 + PR-3. **PR-5 (delete `JinjaPromptRenderer` + remove flag scaffolding)** depends on PR-4 stability confirmation.

---

## §5. Acceptance criteria (the test contract)

These are written as RED tests in Phase 0 — they should fail against the current `JinjaPromptRenderer` (in subtle ways: e.g., AC-FF1 fails because the fallback path has no SOP context) AND must pass after the migration.

### §5.1 Behavioral equivalence (G2 — no regression)

- **AC-EQ1** *(v3.3 strengthened — full feed-category coverage)* `_render_prompt(feed)` (the inferencer method, lines 585-757) returns byte-identical output for the existing `initial.jinja2` comparing legacy `JinjaPromptRenderer` (feature flag off) vs new `TemplateManager` (feature flag on). The test feed dict **MUST exercise all 11 feed categories** that `_render_prompt` constructs (verified empirically against `conversational_inferencer.py:719-730` v3.3):

  | # | Feed key | Source | Template section it renders |
  |---|---|---|---|
  | 1 | `**template_vars` (splat) | `prompt_renderer.template_variables` | `_variables/` cascade (employee identity, etc.) |
  | 2 | `workflow_nextstep_guidance` | `workflow_manager.nextstep_guidance` | `<WorkflowNextStepGuidance>` XML |
  | 3 | `action_tools` | `_format_action_tools()` | `# Available Tools > ## Action Tools` |
  | 4 | `**self.prior_context` (splat) | `ConversationalInferencer.prior_context` | `workflow_description`, `workflow_status`, employee fields |
  | 5 | `completed_actions` | `all_actions` list | Prior actions summary |
  | 6 | `conversation_history` | `messages` list | `# Conversation` XML section |
  | 7 | `current_turn` | `{"role": "user", "content": current_message}` | `<user>` message block |
  | 8 | `conversation_tools` | `_format_conversation_tools()` | `## Conversation Tools` |
  | 9 | `**workflow_sections` (splat) | `workflow_manager.render_prompt_sections()` | `# Available SOPs`, `# Active SOPs`, `<WorkflowDescription>`, `<WorkflowStatus>` |
  | 10 | `is_auto_advance` (via `prior_context`) | Set by orchestrator on auto-advance turns | `{% if is_auto_advance %}` branch |
  | 11 | SOP-feed self-resolution (`{{ session_root_path }}` etc.) | `resolve_templated_feed()` post-pass at line 743 | Recursively rendered Jinja in any of the above values |

  Test fixtures MUST include 5 scenarios — each exercising a different combination of `{% if %}` branches in `initial.jinja2`:
  - **F1 (empty feed)** — minimum keys to render without error (validates default-branch coverage)
  - **F2 (fully-populated)** — all 11 categories populated with non-trivial values
  - **F3 (override a `_variables/` key)** — feed key overrides a default loaded from `_variables/` cascade (validates merge precedence)
  - **F4 (SOP context active)** — `workflow_manager.render_prompt_sections()` returns non-empty `available_sops` + `active_sops` (validates SOP rendering parity)
  - **F5 (`is_auto_advance=True`)** — auto-advance branch active (validates conditional content gating)

  Each fixture: assert `legacy_output == new_output` byte-for-byte. Failure on any fixture is a Phase 4 blocker.
- **AC-EQ2** `render_string("Hello {{ name }}", {"name": "x"})` returns `"Hello x"` under both renderers (feed self-resolution path).
- **AC-EQ3** `template_variables` returns the same merged dict (YAML sidecar + `_variables/` folder cascade) under both renderers, for `initial.jinja2`.
- **AC-EQ4** Adapter's `find_sop_file()` returns `None` (per v3 §3.7); `JinjaPromptRenderer.find_sop_file()` returns `None` in the current repo (empirically verified: `_variables/workflow/sop.*` files do not exist). Both paths produce the same `None` result, so behavioral equivalence holds for the current codebase. (When SOP v1.5 lands `WorkflowManager.get_active_sop_path()`, that's a separate AC handled by the SOP v1.5 plan.)
- **AC-EQ5** `template_config` returns the same dict from `.initial.config.yaml` (and `.config.yaml` fallback) under both renderers.
- **AC-EQ6** `variable_manager.set(key, val)` then `template_variables[key] == val` works identically under both renderers (override propagation).
- **AC-EQ7** `template_source` returns the raw Jinja2 source of `initial.jinja2` under both renderers (substring match; bytes may differ if loader normalizes whitespace).

### §5.2 Standalone construction (G5 — defect fix)

- **AC-FF1** `ConversationalInferencer()` (no `prompt_renderer` arg, no `session_context`) — after `__attrs_post_init__` per §3.10.2 — has `self.prompt_renderer is not None` and `isinstance(self.prompt_renderer, TemplateManagerPromptRenderer)`.
- **AC-FF2** With above standalone construction, calling `await self._render_prompt(feed_with_workflow_state)` produces output that includes `# Available SOPs`, `# Active SOPs`, `<WorkflowDescription>`, etc. — the full conversation template, NOT the bare fallback.
- **AC-FF3** With above standalone construction, accessing `self.prompt_renderer.find_sop_file()` returns `None` (per §3.4 adapter implementation), and call site at line 638 correctly falls through to its `getattr(..., lambda: None)()` default.

### §5.3 SOP discovery delegation (G6 v3 — adapter returns None; SOPRegistry path handled separately)

- **AC-NEW1** Adapter's `find_sop_file()` returns `None` unconditionally (verified by calling `TemplateManagerPromptRenderer(tm, key).find_sop_file()` directly).
- **AC-NEW2** ConversationalInferencer line 638 (`getattr(self.prompt_renderer, "find_sop_file", lambda: None)()`) returns `None` cleanly when called against the adapter — no exception, no error log.
- **AC-NEW3** When SOP v1.5 Phase 4 lands `WorkflowManager.get_active_sop_path()`, ConversationalInferencer's SOP rendering block (lines 633-665) calls it directly. This plan does NOT make that change; it preserves the current `find_sop_file` call site by ensuring the adapter returns `None`.

### §5.4 OpenStartup integration (G4)

- **AC-OP1** OpenStartup full session lifecycle (POST `/api/sessions`, send user message, receive LLM response) succeeds end-to-end after migration. Asserts no behavior regression.
- **AC-OP2** The rendered prompt sent to the LLM in the AC-OP1 session contains the same sections (tools, conversation history, decision procedure) as before migration — diff-friendly snapshot test.
- **AC-OP3** Existing OpenStartup `test/services/test_conversation_service.py` (or equivalent) test suite passes unchanged.

### §5.5 Cleanup verification (G3)

- **AC-CL1** `import agent_foundation.common.inferencers.agentic_inferencers.conversational.prompt_rendering` raises `ImportError` (file removed in Phase 5).
- **AC-CONC1** *(v3.2)* 10 parallel `await renderer.render_string(s_i, ctx_i)` calls against the SAME `TemplateManager` instance with distinct `s_i`/`ctx_i` all return the correct rendered output (no cross-talk from transient-key collisions). Verifies R-NEW1 mitigation empirically. Use `asyncio.gather` + 10 unique-content templates with `{{ ctx_id }}` substitution; assert each result contains the corresponding `ctx_id`.
- **AC-EQ-INTEG1** *(v3.3)* The existing `test_sop_prompt_integration.py` test suite (10 tests, 272 LoC at `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/conversational/test_sop_prompt_integration.py`) **must pass unchanged** with the new `TemplateManagerPromptRenderer` adapter. These tests already exercise `_render_prompt` end-to-end through 10 real SOP scenarios (idle, running, completed, error, goto cycles, branch convergence, multi-active SOPs, etc.) with real `WorkflowManager` state — they are the strongest existing real-rendering verification we have. Phase 0 runs the suite under the legacy renderer (baseline); Phase 2 reruns it with the feature flag on; **all 10 tests must still pass**. Any regression is a Phase 2 blocker. Same applies to the rankevolve mirror at `CoreProjects/atlassian-packages/rankevolve/test/agentic_foundation/common/inferencers/conversational/test_sop_prompt_integration.py`.
- **AC-CL2** `grep -r "JinjaPromptRenderer" CoreProjects/` returns no matches (other than historical references in `_docs/` and changelogs).
- **AC-CL3** `conversational_inferencer.py:_render_fallback_prompt` no longer exists (deleted in Phase 3 since `prompt_renderer` is never `None` after `__attrs_post_init__`).

### §5.6 SOP framework dependency preservation (G7 + companion plan)

These ACs cross-reference `sop_framework_UNIFIED_v1_plan.md` v1.5. They must continue to pass after this migration lands.

- **AC-SOP1** SOP v1.5 AC9.13 (SOP-scoped prompt template) still passes: `SOPInferencer` (a subclass of `ConversationalInferencer`) uses `prompt_templates/sop/main/initial.jinja2`. After this migration, that translates to `SOPInferencer` constructing `TemplateManagerPromptRenderer(template_manager=TemplateManager(active_template_root_space="sop", active_template_type="main"), template_key="initial")` — the SOP-scoped template space is selected via the existing `active_template_root_space` mechanism, no special-casing needed.
- **AC-SOP2** SOP v1.5 §10 turn record schema is unaffected — the migration doesn't touch turn persistence.
- **AC-SOP3** SOP v1.5 R20 / R21 (interactive serializer + bridge) tests still pass — the migration doesn't touch interactive transport.

---

## §6. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | `TemplateManager.__call__()` and `JinjaPromptRenderer.render()` have subtly different feed-merging semantics (e.g., default variable application order) — output drift on edge-case feeds | **MED** | AC-EQ1 is a byte-equality snapshot test on `initial.jinja2` with representative feeds covering: empty feed, fully-populated feed, feed with override of a `_variables/` key, feed with SOP context, feed with `is_auto_advance=True`. Any drift caught here is a blocker for Phase 4. |
| R2 | SOP discovery semantics change — `JinjaPromptRenderer.find_sop_file()` previously searched `_variables/workflow/sop.*`; adapter now returns `None` unconditionally | LOW | Empirically verified (grep) that `_variables/workflow/sop.*` files do NOT exist in the current repo — the SOP at `_variables/workflow_sop/role_creation.md` is read by a different code path (`WorkflowRegistry.load_all`). So returning `None` is behaviorally equivalent today. AC-NEW1/2 verify. When SOP v1.5 lands its own discovery path, this plan's `None` stub becomes the explicit fallback. |
| R3 | `template_variables` differs because `TemplateManager.load_variables()` does multi-root overlay merge while `JinjaPromptRenderer.template_variables` is single-root | LOW | OpenStartup currently uses a single `template_dir` — multi-root would be a future opt-in. Default construction in §3.5.1 + §3.5.2 uses single-root, matching current behavior. If/when multi-root is opted in, that's a separate enhancement. |
| R4 *(v3.2 rewritten — `get_template_config` does not exist on TM)* | `template_config` semantics must match `JinjaPromptRenderer._load_yaml_candidates` cascade: `.<basename>.config.yaml` → `.config.yaml` | LOW | §3.4 adapter implements the cascade directly via `_load_yaml_cascade(candidates)` module helper — TM has no `get_template_config()` method (verified empirically, see Q2). Adapter reconstructs the template path from `_origin_root` + extension probe and applies the exact `JinjaPromptRenderer` ordering. AC-EQ5 verifies parity. **R-NEW2 covers the path-reconstruction edge case.** |
| R5 | `variable_manager` differences — `JinjaPromptRenderer` lazily creates a `FileBasedVariableManager`; `TemplateManager` has its own variable manager lifecycle. Override semantics (`set`/`clear`) may differ | MED | AC-EQ6 verifies override propagation. If mismatched, the adapter wraps `TemplateManager.variable_manager` with an override-translation layer (preserves the duck-typed API). |
| R6 | Standalone construction (§3.5.2) creates a TemplateManager on every `ConversationalInferencer()` instantiation — could be a perf regression if N inferencers are created in a tight loop (e.g., 100 SOP inferencers in `/sop --autonomous`) | LOW | TemplateManager construction is O(template count) = O(low hundreds of files); not on a hot path. Mitigation if observed: a module-level `_DEFAULT_TEMPLATE_MANAGER` lazy singleton (variable_manager state isolated per ConversationalInferencer via a separate lightweight overrides layer). |
| R7 *(v3.1 — Phase 1 was removed in v3)* | Phase 4 (OpenStartup factory change) is a cross-package change — depends on AgentFoundation adapter being deployed | LOW | Phase 4 must merge AFTER PR-2 (adapter creation + `__attrs_post_init__` + feature flag, off by default) AND PR-3 (delete `_render_fallback_prompt`). PR ordering enforced in §4. No RichPythonUtils PR is needed (TemplateManager remains unchanged). |
| R8 | The `_render_fallback_prompt` path may be relied on by tests that intentionally construct `ConversationalInferencer(prompt_renderer=None)` to test the bare-bones path | LOW | Phase 0 RED tests catalogue all current callers of `_render_fallback_prompt` (grep-based). Any test that explicitly tests the fallback path is updated to construct `ConversationalInferencer(prompt_renderer=<minimal-test-renderer>)` instead. Two such tests are expected (per quick grep — actual count verified during Phase 0). |
| R9 | `template_source` semantics differ — `JinjaPromptRenderer.template_source` returns the raw template body via `env.loader.get_source()`. `TemplateManager.get_raw_template()` may return a different form (e.g., wrapped in `_OriginTaggedStr`) | LOW | AC-EQ7 uses substring match (not bytes-equality) to tolerate the `_OriginTaggedStr` wrapping (it's a `str` subclass, so substring still works). If a caller needs raw `str`, the adapter unwraps via `str(...)`. |
| R10 | The migration touches a load-bearing class (`ConversationalInferencer` is in every chat-turn path) — any subtle defect ships to every user immediately | HIGH | (a) Phase 0 RED tests must include the snapshot test (AC-EQ1) before any production change; (b) Phase 6 manual smoke test on local OpenStartup before merging to main; (c) Feature flag `OPENTEAM_USE_LEGACY_JINJA_RENDERER=1` env var that falls back to `JinjaPromptRenderer` for one release cycle (Phase 5 deletes the file ONE release AFTER Phase 4 lands, not in the same release). |
| R-NEW1 *(v3.1; verified v3.2)* | Adapter's `render_string` registers raw template strings under a transient key in `TemplateManager.templates["__transient__"]`. Concurrent calls from different inferencer instances sharing the same TemplateManager could race on the shared `templates` dict | MED | (a) **Empirically verified v3.2:** TM does NOT cache template lookups (grep `lru_cache\|@cache\|cached_template` returns empty in template_manager.py); `__call__` does runtime dict lookup against `self.templates` (line 1429+), so runtime mutation IS safe; (b) `transient_key` is `id(template_str):x` (unique per call — `id()` collisions only happen if Python reuses memory after GC, which the `finally` cleanup ensures); (c) cleanup in `finally` block always runs; (d) the `"__transient__"` root_space is isolated from real templates; (e) AC-EQ2 + new AC-CONC1 (concurrency test: 10 parallel `render_string` calls with same TM, distinct strings) verify no cross-talk in Phase 0. **Follow-up:** propose `TemplateManager.render_template_string(s, **ctx)` as a first-class API that uses the TM's own Environment without registration. |
| R-NEW2 *(v3.1)* | Adapter's `template_config` reconstructs the template file path from `_OriginTaggedStr._origin_root` + `active_template_root_space` + `active_template_type` + `template_key` + extension probe. If TemplateManager's actual file resolution diverges from this assumption (e.g., nested `template_key="foo/bar"` mapping to `foo/bar.jinja2`), config lookup returns `{}` silently — losing tool whitelisting and structural XML escaping config | MED | (a) AC-EQ5 verifies parity against `JinjaPromptRenderer.template_config` for `initial.jinja2`; (b) extension probe iterates `[.jinja2, .j2, .md, .yaml, .yml]` matching TemplateManager's loader; (c) `_origin_root` is set at load time by TemplateManager itself (template_manager.py:413, 971) — it IS the root from which the template was loaded; (d) **follow-up:** propose `TemplateManager.get_template_file_path(template_key)` as a first-class API to RichPythonUtils, then simplify adapter to a one-liner. |

---

## §7. Files inventory

### §7.1 New files

| File | Purpose | LoC |
|---|---|---|
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/template_manager_renderer.py` | `TemplateManagerPromptRenderer` adapter (§3.4) — RESTORED in v3 | ~80 |
| `AgentFoundation/test/common/inferencers/agentic_inferencers/conversational/test_template_manager_renderer_equivalence.py` | AC-EQ1 through AC-EQ7 + AC-FF1 through AC-FF3 + AC-CL3 | ~200 |
| ~~`RichPythonUtils/test/string_utils/formatting/template_manager/test_find_sop_file.py`~~ | **REMOVED in v3.1** — `find_sop_file` was never added to TemplateManager (G6 deleted in v3); AC-NEW1/2 now test the adapter's `find_sop_file() → None` behavior directly | 0 |

### §7.2 Modified files

| File | What | Why |
|---|---|---|
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | Add `__attrs_post_init__` that default-constructs `TemplateManagerPromptRenderer` when `prompt_renderer is None` per §3.10.2 (~15 LoC added); delete `_render_fallback_prompt` (~25 LoC removed); **ZERO changes to the 25 `self.prompt_renderer.*` call sites** | G5, G7 |
| `OpenStartup/src/openteam/server/backends/factories.py:151-165` | Switch from `JinjaPromptRenderer(...)` construction to `TemplateManagerPromptRenderer(template_manager=TemplateManager(...), template_key="initial")` per §3.10.1 (+5 LoC net) | G4 |
| `AgentFoundation/test/...` (existing tests that rely on `JinjaPromptRenderer` or `_render_fallback_prompt`) | Update import / construction site (~2 tests, exact count verified in Phase 0) | R8 |

### §7.3 Deleted files

| File | Why | When |
|---|---|---|
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/prompt_rendering.py` (212 LoC) | Replaced by direct `TemplateManager` usage in `ConversationalInferencer` (G3) | Phase 5 (one release AFTER Phase 4 + 1-week soak, per R10) |

---

## §8. Open questions to resolve before implementation

- **Q1.** ~~Does `TemplateManager.__call__(template_string=...)` exist?~~ — **RESOLVED in v3.1**: `__call__` accepts `template_key` only (not `template_string`). Adapter `render_string` (§3.4) uses the transient-key registration pattern to route raw strings through `__call__`'s full pipeline.
- **Q2.** ~~Does `TemplateManager.get_template_config(template_key)` exist?~~ — **RESOLVED in v3.1**: No such method exists. Adapter `template_config` (§3.4) reconstructs the file path from `_OriginTaggedStr._origin_root` and reads `.config.yaml` cascade directly.
- **Q3.** ~~Should `find_sop_file()` be SOP-aware?~~ — **RESOLVED in v3 §3.7**: Adapter returns `None`; SOP discovery is owned by SOPRegistry (SOP v1.5 plan).
- **Q4.** Should the feature flag `OPENTEAM_USE_LEGACY_JINJA_RENDERER` (R10 mitigation) default to "new" or "legacy" for the first release? — **Recommended: new** (else nobody exercises the new path) **with one-week kill-switch availability**.
- **Q5.** Does `TemplateManager.variable_manager` have `set()` / `clear()` methods with the same semantics as `FileBasedVariableManager`? — Empirical check during Phase 0. If different, the adapter must wrap with a translation layer (R5).

---

## §9. Honest reality check

This migration is **architectural cleanup, not feature delivery**. The user-visible benefit is:

1. **SOP test runs work in standalone mode** (G5 — the defect that motivated this plan). Estimated impact: every SOP author who tries to test their `.md` outside OpenStartup, including the `role_creation` test that prompted this conversation.
2. **`ConversationalInferencer` can use multi-root template overlay and version suffixes** (G1) — currently impossible. Estimated impact: enterprise/consumer prompt variants, A/B testing in production.
3. **One less parallel system to maintain** (G3) — 212 LoC removed, drift risk eliminated.

The migration is **NOT delivering new conversation behavior** to end users. It's making the codebase consistent and unblocking SOP standalone testing.

**Recommendation:** Run as one focused 3-day PR sequence — **no RichPythonUtils changes required** (TemplateManager remains unchanged per v3.1 design). PR-2 (adapter + `__attrs_post_init__`) lands first; PR-3 (delete `_render_fallback_prompt`); PR-4 (OpenStartup `factories.py` + flip feature flag default-on after 1-week soak); PR-5 (delete `JinjaPromptRenderer` after one release of flag stability). Keep the legacy `JinjaPromptRenderer` file deletable behind feature flag for one release to mitigate R10.

---

## §10. Companion plan cross-references

- **`workflows_and_sop/sop_framework_UNIFIED_v1_plan.md` v1.5** (1,731 lines) — depends on this migration for standalone SOP test runs (§1.2 problem 3). AC-SOP1/2/3 verify the dependency is preserved.
- **`templates_and_variables/load_variables_multidot/load_variables_multidot_INTEGRATED_v4_plan.md`** — already landed in `TemplateManager.load_variables()`. This migration inherits that capability automatically (G1 benefit #1).
- **`inferencer_architecture/terminal_inferencer_axes_v7.2_plan.md`** — separate concern; does not interact with this migration.

---

## §11. Out of scope (defer to future plans)

- Migrating `ConversationalInferencer` to inherit from `TemplatedInferencerBase` (N1) — separate plan if/when motivated. This migration eliminates one of the gaps that would block it.
- Adding new template overlay features beyond what `TemplateManager` already supports — explicitly NOT scope creep.
- Refactoring the `_variables/workflow/` SOP discovery convention into a registry — separate plan in `workflows_and_sop/`.
- Migrating any other inferencer subclass — they're already on `TemplateManager` correctly.
