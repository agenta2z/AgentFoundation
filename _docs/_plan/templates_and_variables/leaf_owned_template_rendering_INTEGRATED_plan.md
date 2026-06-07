# Leaf-Owned Template Rendering — INTEGRATED Refactor Plan

**Status**: Implementation-ready — supersedes both:
- `leaf_owned_template_rendering_refactor_plan.md` (Plan A)
- `_alt_plan_leaf_rendering_splendid_lantern.md` (Plan B)
**Author**: Co-designed with @tchen7; integrated from parallel-agent comparison
**Created**: 2026-05-09
**Scope**: AgentFoundation orchestrator inferencers + companion YAMLs
**Risk**: Lower than originally estimated (real YAML scope is **1 file**, not "all topologies")
**Estimated effort**: ~5 active engineering days across 6 phases

---

## §1 Why This Plan Exists (Synthesis Rationale)

Two prior drafts converged on the same architectural direction (move template rendering from orchestrator → leaf via an `extra_feed` channel) but each had different strengths and weaknesses. A four-agent parallel comparison surfaced these key facts:

| Plan A (mine) | Plan B (alt) |
|---|---|
| ✅ Explicit `render_only` mode for orchestrator-side audit/cache | ❌ Missing — orchestrator loses access to rendered prompt |
| ✅ Open-questions discipline (7 explicit decisions) | ❌ Treats decisions as final |
| ✅ Residual-risk and out-of-scope sections | ❌ Missing |
| ✅ Quantified test target (≥15 across phases) | ⚠️ Less granular |
| ❌ **Wrong scope claim**: "all YAMLs in topologies" — actually 1 file | ⚠️ Doesn't quantify scope |
| ❌ Edge-case scenarios only implicit in "subtleties" | ✅ Explicit table (2-agent, no workspace, missing key, etc.) |
| ❌ `FOLLOWUP_TEMPLATE_DEFAULTS` only implied | ✅ Explicit Phase 1 deliverable |
| ❌ No pre-flight audit before Phase 1 | ❌ Same gap |

**Verdict**: Plan A is the better spine (cleaner abstraction + better discipline). This integrated plan keeps Plan A's spine and adds **3 critical corrections**:

1. **Scope correction** — 4 YAMLs actually use `<role>_prompt` (1 production + 3 examples); migration is small but not 1-file
2. **Edge-case scenarios table** (borrowed from Plan B §5)
3. **Explicit `FOLLOWUP_TEMPLATE_DEFAULTS` deliverable** (borrowed from Plan B §3.1c)
4. **NEW: Phase 0a Pre-Flight Audit** (Agent 3's correct critique — both prior plans missed this)

---

## §2 Background

### §2.1 What's There Today (Two Coexisting Mechanisms)

| Mechanism | Owner | Where it renders | Status |
|---|---|---|---|
| **Orchestrator-side rendering** | `Dual.review_prompt`, `Dual.followup_prompt`, `MultiFlowDual.multi_flow_*_prompt`, `MultiFlow.aggregator_prompt`/`multiflow_followup_prompt` | Inside the orchestrator class via `_render_role_prompt()` | Older — pre-dated leaf TemplateManager |
| **Leaf-side rendering** | `TemplatedInferencerBase.template_manager` + `template_root_space` + `template_key` + `template_extra_feed` | Inside the leaf via the leaf's own `TemplateManager` | Newer — added when leaves got built-in templating |

### §2.2 Concrete Manifestation in YAML

Per the verification done 2026-05-09, exactly **ONE** YAML in production sets these fields:

```bash
$ grep -rln '^[[:space:]]*\(review_prompt\|followup_prompt\|multi_flow_followup_prompt\|...\):' \
    --include='*.yaml' /Users/tchen7/MyProjects/CoreProjects 2>/dev/null \
    | grep -v ".venv\|__pycache__\|/_archive"
/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/resources/tools/task/topologies/breakdown-multiflow-plan.yaml         ← PRODUCTION (1)
/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/examples/.../yaml_configs/02_nested.yaml         ← EXAMPLE (3 total)
/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/examples/.../yaml_configs/04_shorthand.yaml
/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/examples/.../yaml_configs/06_deep_nesting.yaml
```

That's it. **4 YAML files, ~8-10 lines total**:
- 1 production YAML (`breakdown-multiflow-plan.yaml`) with 2 lines (`review_prompt: review`, `followup_prompt: followup`)
- 3 example YAMLs that demonstrate `<role>_prompt` usage and must be updated to demonstrate the new leaf-side approach

This is still good news — the migration's YAML blast radius is **minimal**, but the 3 examples must be updated in Phase 2 to keep documentation aligned (a documentation/example bug post-Phase-5 would teach users the wrong pattern).

### §2.3 Why It's a Problem (Even With Just 4 YAMLs)

| Symptom | Root cause |
|---|---|
| Bug class — double rendering | Two owners for the same artifact |
| Future fragility — adding `critique`, `validate`, `refine` requires N new fields + wiring | Phases tied to class hierarchy, not data |
| Cognitive overhead — "which mechanism wins when?" | Two mechanisms with overlapping scope |
| Debugging difficulty — must check both `_render_role_prompt` AND leaf's `_render_prompt` | Scattered render path |

The single YAML usage *masks* a structural defect. The defect is in the code architecture, and removing it benefits future YAML/topology authors much more than current ones.

### §2.4 What's Already In Place That Helps (Verified)

| Building block | Status | Verified location |
|---|---|---|
| Leaf-side TemplateManager + `template_root_space` + `template_key` | ✅ Production | `templated_inferencer_base.py:91-93` |
| Leaf-side `template_extra_feed` (a per-leaf class-level static dict that gets merged into render feed) | ✅ Production | `templated_inferencer_base.py:142-150` (precedence layer 2) |
| `SLOT_DEFAULTS` cascade mechanism | ✅ Production | `dual_inferencer.py:160` (only `review_inferencer` has it; `fixer_inferencer` does NOT) |
| `_TEMPLATE_TRANSPARENT_SLOTS` (parent defaults pass through Dual to nested children) | ✅ Production | `dual_inferencer.py:170` |
| Reusable role-default bundles | ✅ Production | `template_defaults.py` — `REVIEW_TEMPLATE_DEFAULTS`, `BREAKDOWN_TEMPLATE_DEFAULTS`, etc. |
| `FOLLOWUP_TEMPLATE_DEFAULTS` | ❌ **Does NOT exist yet** | Must be added in Phase 1 |
| Hydra walker that applies SLOT_DEFAULTS before construction | ✅ Production | `rich_python_utils.config_utils._instantiate` |

**Critical insight**: ~50% of the destination architecture is already shipped via `SLOT_DEFAULTS` for `review_inferencer`. The work is to **complete an in-flight migration**, not invent infrastructure.

---

## §3 Architectural Direction

### §3.1 The Principle

```
ORCHESTRATOR responsibilities:
  • Workflow control (when review, when fix, when aggregate, etc.)
  • Feed-dict assembly (compute prior_output_path, gather main_response,
    reviewer_response, output_path, iteration counters, ...)
  • Pass feed dict + role hint to the leaf via the new extra_feed channel
  • Capture leaf's response, advance state machine

LEAF responsibilities:
  • Receive feed dict (via inference_input + extra_feed channel)
  • Render its OWN template (using its own TemplateManager) for its
    declared role
  • Execute the LLM call with the rendered prompt
  • Return raw response

OUTCOME:
  • Single source of truth for "which template renders" (the leaf)
  • Single source of truth for "when/why it renders" (the orchestrator)
  • No duplication, no double-rendering bug class possible
```

### §3.2 The Field-Level Migration

| Field/method | Today | Proposed (post-migration) |
|---|---|---|
| `Dual.review_prompt: Optional[str]` | Renders via Dual's `_render_role_prompt` | ❌ Removed (deprecated through Phase 4) |
| `Dual.followup_prompt: Optional[str]` | Same | ❌ Removed |
| `Dual.initial_prompt: Optional[str]` | Wraps user's first input | ⚠️ **Defer** — different semantics; audit in Phase 3 |
| `MultiFlowDual.multi_flow_*_prompt` (3 fields) | Same pattern | ❌ Removed in Phase 3 |
| `MultiFlow.multiflow_followup_prompt`, `.aggregator_prompt` | Same | ❌ Removed in Phase 3 |
| `Dual._render_role_prompt(role, feed, config)` | Orchestrator-side render | ❌ Removed |
| `Dual._build_review_prompt(...)` returns rendered string | — | ✅ Renamed to `_build_review_feed(...)` returning a dict |
| `Dual._build_followup_prompt(...)` returns rendered string | — | ✅ Renamed to `_build_followup_feed(...)` returning a dict |
| `LeafInferencer.ainfer(input=str, ...)` | Takes raw input string | ✅ Extended with `extra_feed: Optional[dict]` and `render_only: bool` |
| YAML on Dual: `review_prompt: review`/`followup_prompt: followup` | Required for path-aware rendering | ❌ Removed; SLOT_DEFAULTS auto-cascades correct `template_key` to leaves |

### §3.3 Five Subtleties (And Their Mitigations)

#### Subtlety 1 — Orchestrator sometimes needs the rendered prompt for logging/caching/checkpointing

**Mitigation**: Add `render_only=True` mode to leaf — returns `InferenceResult(rendered_prompt=..., response=None)` without invoking LLM. Orchestrator can pre-render for audit, then call leaf again for real execution. Plan B was missing this; Plan A's design correctly anticipates it.

#### Subtlety 2 — Workflow state must flow into the leaf's feed

Items like `iteration_count`, `attempt_num`, `consensus_state` are workflow state owned by the orchestrator. The leaf's template (e.g. `plan/main/review.jinja2`) sometimes wants `{{ iteration_count }}`.

**Mitigation**: This is exactly what `extra_feed` enables. The leaf treats them as opaque variables. The orchestrator passes them via `extra_feed`. The leaf's template substitutes; the leaf doesn't *interpret*. Semantic ownership stays with the orchestrator.

#### Subtlety 3 — Backward compatibility with non-templated leaves

Some leaves (`MockInferencer`, `ClaudeBedrockInferencer`, etc.) don't inherit from `TemplatedInferencerBase` and have no TemplateManager.

**Mitigation**: For non-templated leaves, the orchestrator falls back to the old behavior (orchestrator renders via `_render_role_prompt`, passes string as input). The migration is opt-in per leaf type. Detection: `hasattr(leaf, "template_manager") and leaf.template_manager is not None`.

#### Subtlety 4 — Some templates need orchestrator-only context (e.g. `<OriginalUserRequest>` envelope)

**Mitigation**: This is unchanged. `{{ input }}` continues to be the user's original request via `inference_input`. `extra_feed` provides *additional* feed variables in parallel. The two channels coexist.

#### Subtlety 5 — `_template_root_space` cascade is already leaf-side

The `_template_root_space: plan` on Dual cascades to children via the `_-prefix` mechanism — this is already the leaf-side path. The orchestrator's only responsibility is to set the cascade root; leaves discover their own templates from there.

### §3.4 Edge Cases (From Plan B §5)

| Edge case | Handling |
|---|---|
| **2-agent mode** (fixer == base, no separate fixer leaf) | `extra_feed` is per-call, not mutating child state — safe |
| **No workspace / no prior_output_path** | `or ""` sentinel → Jinja `{% if %}` block is falsy → graceful fallback to "(path unavailable)" branch |
| **Leaf has `template_manager` but no `template_key`** | `_render_prompt` raises ValueError → caught by step impl's existing error handler → orchestrator falls back to old render path with deprecation warning |
| **SLOT_DEFAULTS auto-fills `template_key` on fixer** | Yes via Phase 2 wiring of `FOLLOWUP_TEMPLATE_DEFAULTS` (defined in Phase 1, wired in Phase 2 atomically with Dual stop-rendering — see §4.3c transactional invariant) — fixer gets `template_key="followup"` without explicit YAML |
| **Existing `_build_*_prompt` callers outside DualInferencer** | Verified by grep: zero — `_render_role_prompt` is private and only called from Dual's own `_build_*_prompt` methods. Safe to refactor signature. |
| **Custom prompt_formatter (not TemplateManager)** | Falls through to old render path; deprecation warning fires; works |
| **Inferencer constructed via `__new__()` in tests** | `extra_feed` defaults to `None` — safe, no behavior change |

---

## §4 Phased Plan (6 Phases)

### §4.1 Phase 0 — Already Done (Path-Aware + YAML Cleanup)

**Status**: ✅ COMPLETE (2026-05-08 → 2026-05-09)

- Path-aware fix shipped (`prior_output_path` plumbing in Dual + 4 templates)
- Double-rendering bug fixed in `breakdown-multiflow-plan.yaml`
- 50/50 unit tests passing
- "No silent failure" semantic added (`_RoleDisabledError`)

This phase establishes the safety net for the bigger refactor.

### §4.2 Phase 0a — Pre-Flight Audit (NEW — addresses Agent 3's gap)

**Status**: NOT STARTED
**Estimated effort**: 0.5 day
**Risk**: 🟢 None (read-only investigation)

**Goal**: Before any code changes, produce a definitive map of:
- Every file that references `<role>_prompt` (already done partially: see §2.2)
- Every leaf inferencer class and whether it has `template_manager`
- Every YAML that constructs a Dual/MFDual/MultiFlow

**Deliverables**:

1. **Audit script** (`tools/audit_role_prompt_usage.py`) that prints:
   ```
   YAMLs setting <role>_prompt:
     - breakdown-multiflow-plan.yaml: review_prompt=review, followup_prompt=followup
     [... others if any are added later]

   Code referencing <role>_prompt:
     - dual_inferencer.py: review_prompt (218), followup_prompt (219), ...
     [... etc.]

   Leaf inferencer classes (TemplatedInferencerBase descendants):
     - RovoDevCliInferencer: ✅ has template_manager
     - ClaudeBedrockInferencer: ❌ no template_manager
     - MockInferencer: ❌ no template_manager
     [... etc.]

   Orchestrator-side renderers found:
     - DualInferencer._render_role_prompt
     - MultiFlowDualInferencer._render_role_prompt
     - MultiFlowInferencer._render_role_prompt (if exists)
   ```

2. **Audit report** committed to `_docs/_plans/audit_results_<date>.md` containing the script's output. Becomes the **ground truth** for Phase 1-3 scope.

**Acceptance**:
- Script runs in <5 seconds
- Output is machine-parseable (so Phase 5's "verify removal" can re-run it)
- Audit report committed and reviewed before starting Phase 1

**Why this matters**: Both prior plans guessed at scope and got it wrong (Plan A initially claimed "all YAMLs", second-round verification found 4 YAMLs total — 1 production + 3 examples). The audit eliminates guesswork, prevents future scope creep, and provides automation for Phase 5's "verify everything is migrated" check.

**Phase 0a deliverable also includes** (added after second-round comparison):
- A test-suite spy that catches indirect uses of `_render_role_prompt()` (e.g., subclasses that override `_build_review_prompt()` without calling `super()`, or test mocks patching the method directly). The static grep alone is insufficient.
- A categorized list of leaf inferencer classes split by templated/non-templated. This list informs Q10's hard-requirement validation in Phase 5.

### §4.3 Phase 1 — Add `extra_feed` + `render_only` + `FOLLOWUP_TEMPLATE_DEFAULTS`

**Status**: NOT STARTED
**Estimated effort**: 1 day
**Risk**: 🟢 Low (additive only, no behavioral change)

**Goal**: Build the new infrastructure. No production code path uses it yet.

#### 4.3a — Extend `TemplatedInferencerBase._render_prompt` with `extra_feed`

**File**: `src/agent_foundation/common/inferencers/templated_inferencer_base.py`

**Change**:
```python
def _render_prompt(
    self,
    feed: dict,
    *,
    extra_feed: Optional[dict] = None,  # NEW
    inference_config: Optional[dict] = None,
) -> str:
    """Render the leaf's template against ``feed``.

    Args:
        feed: The base feed dict (typically contains ``input`` and any
            class-level template_variables-resolved values).
        extra_feed: Per-call feed overrides supplied by the orchestrator.
            Merged AFTER class-level template_variables and BEFORE feed
            so that orchestrator-supplied values win over class-level
            defaults but never overwrite the LLM-input slot.

    Precedence (lowest → highest):
        1. Class-level ``template_variables`` (resolved via TemplateManager)
        2. Class-level ``template_extra_feed`` (static, per-leaf-class)
        3. ``extra_feed`` (per-call, supplied by caller)  ← NEW
        4. Caller-supplied ``feed`` (overrides everything; this is the
           sacrosanct ``{{ input }}`` slot and any caller adjustments)
    """
    # ... existing logic ...
    if extra_feed:
        rendered_feed.update(extra_feed)  # at appropriate point in pipeline
```

**Deliverables**:
- `_render_prompt` signature extended
- Internal precedence layer 3 added
- Class docstring updated to document the precedence rule

#### 4.3b — Extend `InferencerBase.ainfer` with `extra_feed` and `render_only`

**File**: `src/agent_foundation/common/inferencers/inferencer_base.py`

**Change**:
```python
async def ainfer(
    self,
    inference_input,
    inference_config=None,                     # ← PRESERVED positional (backward compat)
    *,
    extra_feed: Optional[dict] = None,         # NEW — keyword-only
    render_only: bool = False,                 # NEW — keyword-only
    **_inference_args,
) -> InferenceResult:
    """Run inference with optional extra-feed and render-only modes.

    BACKWARD COMPAT NOTE (CRITICAL):
        ``inference_config`` MUST remain the second positional parameter
        for backward compatibility with existing call sites that pass it
        positionally. Verified positional callers (would break if reordered):
            - inferencer_base.py:1006   — sync fallback chain lambda
            - inferencer_base.py:1595   — async fallback chain lambda
            - inferencer_base.py:1632   — async retry loop
            - multi_flow_inferencer.py:1129  — parent class call
            - openclaw_inferencer.py    — sync wrapper
        The new ``extra_feed`` and ``render_only`` are keyword-only via the
        ``*,`` barrier so they cannot collide with future positional adds.

    Args:
        inference_input: The user/orchestrator prompt input string.
        inference_config: Per-call config overrides (POSITIONAL for backward compat).
        extra_feed: Per-call feed dict for templated leaves. Forwarded
            to ``_render_prompt(extra_feed=...)``. Ignored (with one-time
            warning per leaf class) when leaf is non-templated.
        render_only: If True, render the prompt template and return
            ``InferenceResult(rendered_prompt=..., response=None)``
            without invoking the LLM. Used by orchestrators that need
            to pre-render for logging/cache-key/checkpoint purposes.
    """
```

**Deliverables**:
- `ainfer` signature extended (with `inference_config` preserved as 2nd positional)
- `render_only` skips LLM call and returns `InferenceResult` with only `rendered_prompt` populated
- For non-templated leaves: `extra_feed` triggers a one-time `logger.warning` per leaf class (use a class-level `_warned_about_extra_feed` flag) and is otherwise ignored
- **Leakage prevention** (added after Round-3 audit): `extra_feed` and `render_only` are CONSUMED in `_ainfer_single` before forwarding to `_ainfer()`. They are NEVER forwarded via `**_inference_args` to leaf implementations (which would cause `TypeError: unexpected keyword argument 'extra_feed'` in many leaves). Concretely, `_ainfer_single` extracts them at the top of its body and only forwards the remaining `_inference_args` to `_ainfer()`.

- **Conditional `_render_prompt` call** (added after Round-7 audit, fixing Q15 mistake): `_ainfer_single` / `_infer_single` MUST guard the `_render_prompt` call to avoid `TypeError` on subclass overrides that don't accept `extra_feed` (e.g. `ConversationalInferencer._render_prompt(self, current_message: str)`). Verified: passing `extra_feed=None` (the default for all existing callers) to such an override raises `TypeError`. The conditional-pass pattern preserves byte-identical behavior for today's callers:
  ```python
  # In _ainfer_single (and _infer_single, identical pattern):
  if extra_feed is not None:
      inference_input = await self._render_prompt(inference_input, extra_feed=extra_feed)
  else:
      inference_input = await self._render_prompt(inference_input)
  ```
  When `extra_feed is None` → identical to today (no kwarg passed → no LSP issue). When `extra_feed={...}` → only forwarded to overrides that accept it. Phase 0a audit enumerates `_render_prompt` overrides so we know which need the conditional and which would tolerate the unconditional pass.

**Test (added after Round-3 audit)**:
- `test_ainfer_signature_backward_compat`: Construct a leaf, call `await leaf.ainfer("input", inference_config={"foo": "bar"})` (positional `inference_config`) — must succeed without TypeError
- `test_extra_feed_does_not_leak_to_underlying_ainfer`: Mock `_ainfer`, call `await leaf.ainfer("input", extra_feed={"x": "y"})`, assert that `_ainfer` was NOT called with `extra_feed` in its kwargs

**Sync path**: The same signature change must be made to `def infer(...)` and the same leakage prevention to `_infer_single`. Plan covers this in Phase 1's deliverables but the test list above includes both async and sync variants.

#### 4.3c — Define `FOLLOWUP_TEMPLATE_DEFAULTS` constant (DEFINE only — DO NOT wire into SLOT_DEFAULTS yet)

**⚠️ CRITICAL ORDERING (caught in Round-4 audit)**: Wiring `FOLLOWUP_TEMPLATE_DEFAULTS` into `Dual.SLOT_DEFAULTS` in Phase 1 would re-introduce the same double-rendering bug we manually fixed at the YAML level — because `apply_to()` only fills `template_key` if not already present (verified in `template_defaults.py:apply_to`), AND in Phase 1 the Dual STILL renders `plan/main/followup.jinja2` itself. The fixer leaf would then render the same template AGAIN, wrapping Dual's output. This was the original bug. We must not regress.

**Phase 1 deliverable**: define the constant ONLY.

**File**: `src/agent_foundation/common/inferencers/template_defaults.py`

```python
# Add after REVIEW_TEMPLATE_DEFAULTS:
FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key="followup",
)

__all__ = [
    # ... existing exports ...
    "FOLLOWUP_TEMPLATE_DEFAULTS",
]
```

**Phase 2 deliverable** (NOT Phase 1) — wire into SLOT_DEFAULTS atomically with Dual's stop-rendering change:

**File**: `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

```python
# In the SAME commit as §4.4b (Dual switches from _render_role_prompt
# to _build_followup_feed + extra_feed):
SLOT_DEFAULTS: ClassVar[Dict[str, Any]] = {
    "review_inferencer": REVIEW_TEMPLATE_DEFAULTS,
    "fixer_inferencer": FOLLOWUP_TEMPLATE_DEFAULTS,  # NEW (Phase 2 — atomic with Dual stop-rendering)
}
```

**Why both must ship together** (transactional invariant):
- BEFORE Phase 2: Dual renders `followup.jinja2` itself + leaf MUST NOT have `template_key` (else double-rendering). The current YAML achieves this by NOT setting `template_key` on the fixer (Phase 0 manual YAML fix).
- AFTER Phase 2: Dual stops rendering + leaf MUST have `template_key="followup"` (else fixer has nothing to render). SLOT_DEFAULTS provides this automatically.
- Wiring SLOT_DEFAULTS in Phase 1 (without Dual stop-rendering) → leaf has `template_key` AND Dual still renders → double-rendering bug.

**Test (Phase 2)**:
- `test_phase2_no_double_rendering_after_slot_defaults_wiring`: After Phase 2, render the fixer's prompt and assert the `<ProposedDocument>` envelope appears exactly once (not twice nested).

This makes the leaf fixer **automatically** get `template_key="followup"` cascaded onto it without explicit YAML declaration — symmetric with how `review_inferencer` already gets `template_key="review"`. **But timing matters: define the constant in Phase 1, wire it in Phase 2.**

#### 4.3d — Loud Failure for Misconfigured `template_key` at the Leaf

**Why this exists**: Phase 0 (already shipped) added "no silent failure"
semantics at the DualInferencer level via ``_RoleDisabledError`` and the
explicit-vs-implicit check. After this refactor, rendering happens at the
**leaf level**, so the loud-failure semantic must move there too. Without
this, a typo'd or stale ``template_key`` on a leaf could silently fall back
to TemplateManager's ``default_template`` (the same bug class we're
eliminating, just re-emerging at a different layer).

**Design** — three-state semantic mirroring Phase 0:

| State on the leaf | Resolution behavior | Failure mode |
|---|---|---|
| `template_key` is set explicitly to a non-empty value | Try to resolve. If TemplateManager doesn't have a matching template (and the resolved template would fall through to `default_template`), **raise loud `ValueError`** — exact same semantic as Phase 0's "explicit + missing → ValueError" | LOUD |
| `template_key` is empty/None AND a SLOT_DEFAULTS cascade implied a value | Same as above — once SLOT_DEFAULTS fills the slot, the leaf treats it as explicitly configured | LOUD |
| `template_key` is empty/None AND NO SLOT_DEFAULTS cascade fired | Leaf doesn't render — passes input through unchanged (legacy "leaf as raw LLM passthrough") | INTENTIONAL |

**File**: `src/agent_foundation/common/inferencers/templated_inferencer_base.py`,
method `_render_prompt`

**Change**: Add a probe step that distinguishes "found genuine template"
from "got TemplateManager's default fallback".

```python
# ─────────────────────────────────────────────────────────────────────────
# IMPORTANT: This is an ADDITIVE change, not a replacement.
# Current production code in templated_inferencer_base.py:121 implements
# _build_template_feed with: load_variables() resolution, __template_space__
# injection, mode flags (_inject_mode_flags_and_content), output_path
# resolution. We MUST NOT drop any of that. The change is to extend the
# existing method with one new keyword-only parameter (extra_feed) and the
# reserved-key guard. The full ~50-line existing logic stays.
# ─────────────────────────────────────────────────────────────────────────

def _build_template_feed(self, inference_input: str, *, extra_feed: Optional[dict] = None) -> dict:
    """Build the template variable feed dict (extended with extra_feed support).

    PRESERVES all existing functionality from the current implementation:
        1. load_variables() cascade for self.template_variables
        2. self.template_extra_feed merge
        3. __template_space__ injection (when template_root_space set)
        4. self._inject_mode_flags_and_content(feed) (mode flags + content)
        5. {{ input }} binding (sacrosanct, always wins)
        6. output_path resolution (when has_local_access)

    NEW additions (this plan):
        7. Reserved-key guard for extra_feed (Q11)
        8. extra_feed merge between class-level and {{ input }}
    """
    # NEW: Reserved-key guard (Q11) — extra_feed cannot clobber sacrosanct slots
    if extra_feed:
        PROTECTED = {"input", "__template_space__"}
        collisions = PROTECTED & extra_feed.keys()
        if collisions:
            raise ValueError(
                f"extra_feed contains reserved key(s) {sorted(collisions)} which would "
                f"clobber sacrosanct slots. Reserved: {sorted(PROTECTED)}. "
                f"Caller must remove these keys before passing extra_feed."
            )

    # === EXISTING LOGIC (unchanged from production) ===
    feed: dict = {}
    # Step 1: load_variables() cascade for template_variables
    if self.template_variables and self.template_manager:
        if hasattr(self.template_manager, "load_variables"):
            resolved = self.template_manager.load_variables(
                variable_specs=self.template_variables,
                root_space=self.template_root_space,
                default_version=self.template_version or "",
            )
            feed.update(resolved)
        else:
            for var_name, value in self.template_variables.items():
                feed[var_name] = value if value else ""
    # Step 2: class-level extra_feed merge
    feed.update(self.template_extra_feed)
    # NEW Step 2.5: per-call extra_feed merge (between class-level and __template_space__/input)
    if extra_feed:
        feed.update(extra_feed)
    # Step 3: __template_space__ injection
    if self.template_root_space:
        feed["__template_space__"] = self.template_root_space
    # Step 4: mode flags + content injection (preserves enable_<name> etc.)
    self._inject_mode_flags_and_content(feed)
    # Step 5: {{ input }} sacrosanct binding (always last)
    feed["input"] = inference_input
    # Step 6: output_path injection (existing logic — preserved)
    # ... (existing code from production for output_path resolution)
    return feed


# Note on _render_prompt signature (Round-5 fix):
# We do NOT change _render_prompt's signature from (inference_input: Any) to (feed: dict).
# That change would break:
#   - The InferencerBase stub at inferencer_base.py:696 — `def _render_prompt(self, inference_input: Any)`
#   - The override in conversational_inferencer.py:568 — `def _render_prompt(self, current_message: str)`
#   - The call site in _ainfer_single / _infer_single
# Instead, _render_prompt KEEPS its current signature (inference_input as first param)
# and gains an OPTIONAL keyword-only `extra_feed` argument. The internal call to
# _build_template_feed forwards extra_feed through. ConversationalInferencer's
# override is unaffected (it doesn't accept extra_feed and doesn't need to).

def _render_prompt(self, inference_input: Any, *, extra_feed: Optional[dict] = None) -> Any:
    """Render template if configured; otherwise pass input through.

    PRESERVES the production guard order from templated_inferencer_base.py:280:
      Guard 1: template_manager is None → passthrough (leaf without templates)
      Guard 2: template_manager set BUT no key/space → ValueError (misconfig)
      Otherwise → render via template_manager
    """
    # ── Guard 1 (PRODUCTION-PRESERVED) ──
    # template_manager is None → passthrough. Critical for leaves whose
    # template_manager wasn't injected (e.g., constructed in tests, or
    # cascade injection skipped). Without this guard, the get_raw_template()
    # call below would raise AttributeError on NoneType.
    if self.template_manager is None:
        return inference_input

    # ── Guard 2 (PRODUCTION-PRESERVED) ──
    # template_manager set BUT neither key nor space configured → loud
    # failure. This is a leaf that opted INTO templating but didn't say
    # which template. Silent passthrough here would hide bugs.
    if not self.template_root_space and not self.template_key:
        raise ValueError(
            f"{type(self).__name__}: template_manager is set but neither "
            f"template_root_space nor template_key is configured. Either "
            f"set them, or remove template_manager to opt out of templating."
        )

    # ── NEW (Phase 1d): Distinguish genuine template hit from
    #                    silent default_template fallback. ──
    raw = self.template_manager.get_raw_template(
        template_key=self.template_key,
        root_space=self.template_root_space,
    )
    # NOTE: TemplateManager.default_template defaults to "" (empty string),
    # not None — see /CoreProjects/RichPythonUtils/.../template_manager.py:226.
    # We must compare against the actual configured default value AND treat
    # empty-string default as "no real default configured" (which means a
    # missing template returns "" — also a silent failure mode).
    tm_default = getattr(self.template_manager, "default_template", None)
    is_silent_default_fallback = (
        # We asked for a specific template_key but...
        bool(self.template_key)
        and (
            # ...we got nothing back at all (TemplateManager returned None or "")
            raw is None
            or raw == ""
            # ...or we got back the configured fallback (caller didn't ask for default)
            or (tm_default and raw == tm_default)
        )
    )
    if is_silent_default_fallback:
        raise ValueError(
            f"{type(self).__name__}.template_key={self.template_key!r} "
            f"(root_space={self.template_root_space!r}) could not be resolved by the "
            f"configured TemplateManager. The lookup either returned None or "
            f"silently fell through to TemplateManager.default_template, which is "
            f"a configuration error. Either:\n"
            f"  (a) provide a template at "
            f"{self.template_root_space}/{self.template_key}.jinja2 (or your "
            f"TemplateManager's equivalent path), or\n"
            f"  (b) clear ``template_key`` to opt into pure-passthrough mode "
            f"(leaf treats inference_input as the final prompt, no rendering)."
        )

    # ... existing render logic with extra_feed merging ...
```

**Why this is elegant (not hacky)**:
- Single failure mode: ANY misconfiguration → loud ValueError
- Pure-passthrough mode is opt-in via empty `template_key` (clear intent)
- No magical "is_default" detection that depends on string comparison alone
  — we use identity check against TemplateManager's known default_template
- Symmetric with Phase 0's DualInferencer loud-failure semantics
- Leaf-level enforcement means "no silent failure" is now a structural
  invariant of the rendering pipeline, not a per-orchestrator concern

#### 4.3e — Tests (7 methods, ≥1 per concern)

**File**: `test/agent_foundation/common/inferencers/test_dual_inferencer/test_extra_feed.py`

| Test | Asserts |
|---|---|
| `test_extra_feed_merges_into_jinja_feed` | Mock leaf with template `"{{ input }} {{ custom_var }}"`, call `ainfer("hello", extra_feed={"custom_var": "world"})`, expect `"hello world"` |
| `test_extra_feed_precedence_over_template_variables` | Class has `template_variables={"x": "default"}`; call with `extra_feed={"x": "override"}`; expect rendered output uses `"override"` |
| `test_render_only_returns_rendered_prompt_without_llm_call` | Mock leaf; call `ainfer(..., render_only=True)`; assert LLM mock never invoked; assert `result.rendered_prompt` is non-empty and `result.response` is `None` |
| `test_extra_feed_warns_once_for_non_templated_leaf` | Non-templated leaf; first `ainfer(..., extra_feed={...})` triggers `logger.warning`; second call with same leaf class does NOT warn again |
| `test_followup_template_defaults_cascades_to_fixer_via_slot_defaults` | Construct Dual via Hydra-walker with no explicit `template_key` on fixer leaf; assert `dual.fixer_inferencer.template_key == "followup"` |
| `test_template_key_typo_raises_loud_value_error` | Set `template_key="review_typo"` (no such file); call `ainfer(...)`; assert `ValueError` with message naming the bad key + remediation guidance — NO silent fallback to `default_template` |
| `test_empty_template_key_passes_input_through_unchanged` | Set `template_key=""` (intentional passthrough); call `ainfer("hello world")`; assert rendered output is exactly `"hello world"` (no rendering, no error) |

**Acceptance**: All 7 new tests pass (5 functionality + 2 loud-failure); all existing 50 tests continue to pass.

### §4.4 Phase 2 — Migrate `DualInferencer` to Leaf-Side Rendering

**Status**: NOT STARTED
**Estimated effort**: 1.5 days
**Risk**: 🟡 Medium (touches the most-used orchestrator)

**Goal**: Stop the Dual from rendering `review`/`followup` itself. Instead, build feed dicts and let the leaf render via its own TemplateManager.

#### 4.4a — Add `_build_review_feed()` and `_build_followup_feed()` to Dual

```python
def _build_review_feed(
    self, inference_input, proposal, counter_feedback, inference_config,
    *, iteration: int, attempt: int,
) -> dict:
    """Return the feed dict (NOT a rendered string) to be merged into
    the review leaf's template render.

    Plan A note: the existing _build_review_prompt is kept as a thin
    deprecated wrapper for one release cycle; new code uses this method.
    """
    return {
        "main_response": proposal,
        "counter_feedback": counter_feedback,
        # ... all other path-aware feed values from the existing builder ...
        "iteration": iteration,
        "attempt": attempt,
    }
```

Same pattern for `_build_followup_feed()`.

#### 4.4b — Modify `_step_review_impl` and `_step_fix_impl`

```python
# In _step_review_impl:
review_inferencer = self.review_inferencer
if review_inferencer is not None and getattr(review_inferencer, "template_manager", None) is not None:
    # NEW PATH: leaf-side rendering
    feed = self._build_review_feed(...)
    result = await review_inferencer.ainfer(
        inference_input=state["inference_input"],
        extra_feed=feed,
    )
else:
    # LEGACY PATH: orchestrator-side rendering (deprecated)
    rendered = self._render_role_prompt("review", ..., inference_config)
    result = await review_inferencer.ainfer(inference_input=rendered)
```

#### 4.4c — Mark `Dual.review_prompt` / `Dual.followup_prompt` as deprecated

```python
def __attrs_post_init__(self):
    # ... existing code ...
    if self.review_prompt is not None or self.followup_prompt is not None:
        warnings.warn(
            "DualInferencer.review_prompt / followup_prompt are deprecated. "
            "Move to leaf-side rendering: remove these YAML fields and let "
            "SLOT_DEFAULTS cascade `template_key` to the leaf inferencer. "
            "See: _docs/migrations/leaf_owned_template_rendering.md",
            DeprecationWarning,
            stacklevel=2,
        )
```

#### 4.4d — Migrate `breakdown-multiflow-plan.yaml`

The single YAML with `<role>_prompt` fields. Two-line change:
```yaml
# REMOVE:
review_prompt: review
followup_prompt: followup
```

Verify SLOT_DEFAULTS cascade lands `template_key="review"` on `review_inferencer` and `template_key="followup"` on `fixer_inferencer` automatically.

#### 4.4e — Tests (6 methods)

| Test | Asserts |
|---|---|
| `test_build_review_feed_returns_dict_with_expected_keys` | Returns dict with `main_response`, `counter_feedback`, `iteration`, etc. — all keys the review template expects |
| `test_build_followup_feed_returns_dict_with_path_aware_keys` | Returns dict including `prior_output_path`, `main_response`, `reviewer_response` |
| `test_step_review_impl_uses_extra_feed_when_leaf_templated` | Mock templated leaf; verify Dual passes `extra_feed=...` and leaf receives it |
| `test_step_review_impl_falls_back_to_old_path_when_leaf_not_templated` | Mock non-templated leaf; verify Dual renders prompt itself and passes string |
| `test_step_fix_impl_same_two_paths` | Same as above for fix step |
| `test_dual_with_deprecated_field_emits_warning` | Construct Dual with `review_prompt="review"`; assert `DeprecationWarning` fires |
| `test_e2e_migrated_yaml_renders_correctly` | Live render of `breakdown-multiflow-plan.yaml` — assert fixer's prompt has the path-aware block at the leaf level (single render, no nesting) |

**Acceptance**:
- All existing tests pass (with deprecation warnings filtered for legacy tests)
- 7 new tests pass
- E2E test confirms double-rendering is structurally impossible
- Live SOP plan run produces expected fixer prompt

### §4.5 Phase 3 — Migrate `MultiFlowDual`, `MultiFlow`, audit `BTA`/`PTI`

**Status**: NOT STARTED
**Estimated effort**: 2 days
**Risk**: 🟡 Medium (broader surface)

**Scope** (verified from §4.2 audit):

| Orchestrator | Fields to migrate | Notes |
|---|---|---|
| `MultiFlowDualInferencer` | `multi_flow_aggregator_prompt`, `multi_flow_followup_prompt`, `multi_flow_initial_prompt` | Same pattern as Dual |
| `MultiFlowInferencer` | `multiflow_followup_prompt`, `aggregator_prompt` | Same pattern |
| `BreakdownThenAggregateInferencer` | `aggregator_prompt_builder` (Callable, not str) | Audit — different mechanism, may not need same treatment |
| `PlanThenImplementInferencer` | (uses `prompt_builder` callbacks) | Audit — likely out of scope |
| `Dual.initial_prompt` | (deferred from Phase 2) | Decide in Phase 3: migrate or keep |

**Per orchestrator**: same pattern as Phase 2 (add `_build_<role>_feed`, modify step methods, deprecate fields, migrate any YAML).

**Tests**: ~5 per orchestrator × 2-3 orchestrators = ~10-15 new tests.

**Acceptance**: Each migrated orchestrator passes the same criteria as Phase 2; all production YAMLs verified by re-running the Phase 0a audit script.

### §4.6 Phase 4 — Deprecation Period

**Status**: NOT STARTED
**Estimated effort**: 0.5 day (active) + 1 release cycle wait
**Risk**: 🟢 Low

**Deliverables**:

1. Bump `DeprecationWarning` → `FutureWarning` after one release cycle
2. **Migration guide** in `_docs/migrations/leaf_owned_template_rendering.md`:
   - Before/after YAML examples
   - Common pitfalls (forgetting `template_root_space` on leaf; missing `template_key`)
   - How to verify migration via the Phase 0a audit script
3. **Documentation updates**:
   - `templated_inferencer_base.py` module docstring: document `extra_feed` precedence rule
   - `template_defaults.py` module docstring: list all available `*_TEMPLATE_DEFAULTS` bundles + the leaf-side flow
   - Architecture overview doc: "two mechanisms" → "one mechanism" diagram

**Acceptance**: Migration guide reviewed; all docs current.

### §4.7 Phase 5 — Remove Deprecated Fields

**Status**: NOT STARTED
**Estimated effort**: 0.5 day
**Risk**: 🟢 Low (all migrated by this point)

**Pre-conditions**:
- ✅ Phase 0a audit script returns zero `<role>_prompt` usages
- ✅ All tests pass without using deprecated fields
- ✅ Deprecation period elapsed (≥1 release cycle)

**Deliverables**:

1. **Remove from `DualInferencer`**:
   - `review_prompt`, `followup_prompt`, `initial_prompt` (if Phase 3 migrated it) fields
   - `_render_role_prompt()` method
   - `_RoleDisabledError` class (if no longer needed; verify)
   - `_build_review_prompt()`, `_build_followup_prompt()` (deprecated wrappers; the underscored `_feed` versions remain)

2. **Remove from `MultiFlowDualInferencer`, `MultiFlowInferencer`**:
   - All `<role>_prompt` fields
   - `_render_role_prompt()` (or equivalent)

3. **Remove from `constants.py`**:
   - `DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE`
   - `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE`
   - Any other deprecated default-template constants

4. **Re-run audit script** — confirm zero matches across the codebase

5. **YAML schema linter** (if any): reject the removed fields with helpful error pointing at migration guide

**Acceptance**:
- `git grep "review_prompt\|followup_prompt"` returns zero matches in production code (test files may keep historical references in old test cases)
- All tests pass
- Audit script clean

---

## §5 Sequencing & Schedule

| Phase | Status | Duration | Cumulative | Risk |
|---|---|---|---|---|
| **0** ✅ DONE | Path-aware fix + YAML cleanup + tests | — | — | — |
| **0a** | Pre-flight audit script + report | 0.5 d | 0.5 d | 🟢 None |
| **1** | `extra_feed` + `render_only` + `FOLLOWUP_TEMPLATE_DEFAULTS` + **leaf-side loud-failure** + 7 tests | 1 d | 1.5 d | 🟢 Low |
| **2** | Migrate `Dual` + 7 tests + 1-YAML migration | 1.5 d | 3 d | 🟡 Medium |
| **3** | Migrate `MFDual`, `MultiFlow`, audit BTA/PTI | 2 d | 5 d | 🟡 Medium |
| **4** | Migration guide + docs + bump warnings | 0.5 d | 5.5 d | 🟢 Low |
| **(wait)** | Deprecation cycle (1 release) | — | — | — |
| **5** | Remove deprecated fields + verify via audit | 0.5 d | 6 d | 🟢 Low |

**Total active engineering**: **~5.5 days** across 6 active phases (Phase 0a is new vs Plan A's 5.5d estimate; this offsets by ~0 since Phase 2 YAML scope shrunk from "all topologies" to 2 lines in 1 file)

**Wall-clock**: 1 active week + 1 deprecation cycle wait + 0.5 day final cleanup ≈ **6 weeks total**

**Critical path**: 0a → 1 → 2 → 3 must ship in order. Phase 4 is documentation-heavy. Phase 5 requires the deprecation cycle wait.

---

## §6 Risk Assessment

### §6.1 Mitigated Risks

| Risk | Mitigation |
|---|---|
| Breaking external consumers using `<role>_prompt` | Phase 4 deprecation period + migration guide + audit script as automation |
| Breaking tests that mock orchestrator-side rendering | Phase 1's backward-compat fallback (non-templated leaves use old path) + Phase 2's deprecation warnings (don't remove yet) |
| Double-rendering re-emerging during transition | Phase 2's YAML migration removes the only real instance; SLOT_DEFAULTS cascade is the structural fix |
| Loss of orchestrator-side audit logging | Phase 1's `render_only=True` mode preserves the capability |
| Hidden code paths still relying on `_render_role_prompt` | Phase 0a audit + Phase 5 re-audit catches these |

### §6.2 Residual Risks

| Risk | Why residual | Acceptance |
|---|---|---|
| Some external repo (rovo-chat-desktop, etc.) imports `_render_role_prompt` directly | Cross-repo grep is expensive; Phase 4 deprecation warning will surface in their logs | Accepted; communicate via release notes |
| Performance regression: `extra_feed` adds dict-merge overhead | A few-keys merge per inference call — negligible (<1ms) | Accepted as cost of correctness |
| Hydra walker corner cases when `<role>_prompt` is removed mid-migration | Possible if YAMLs have unusual structure | Caught by Phase 2/3 E2E tests; Phase 4 migration guide flags this class |
| `Dual.initial_prompt` migration unclear | Different semantics (one-shot wrap, not workflow loop) | Deferred to Phase 3 with explicit decision point |

### §6.3 Out of Scope

- **PTI's `prompt_builder` mechanism** — different abstraction; audit in Phase 3 to see if it benefits
- **Conversational orchestrators** — already use a different prompt paradigm (per-turn)
- **Inferencer caching strategy** — orthogonal
- **Adding new roles** (`critique`, `refine`, `validate`) — this refactor enables them; specific additions are future work

---

## §7 Acceptance Criteria

The refactor is "done" when ALL of the following are true:

1. ✅ Phase 0a audit script returns zero `<role>_prompt` usages in production code
2. ✅ No production code path renders `review`/`followup`/`aggregator` templates outside the leaf's TemplateManager
3. ✅ No YAML in `OpenStartup/topologies/` or `AgentFoundation/test/yaml_configs/` sets `<role>_prompt` on an orchestrator
4. ✅ The `<role>_prompt` fields and `_render_role_prompt()` method are removed from `DualInferencer`, `MultiFlowDualInferencer`, `MultiFlowInferencer`
5. ✅ `DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE` and `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE` constants are removed
6. ✅ Migration guide published at `_docs/migrations/leaf_owned_template_rendering.md`
7. ✅ Live SOP plan run produces a fixer prompt with the path-aware block fired at the leaf level (single render, no nesting)
8. ✅ All existing test files pass without using deprecated fields
9. ✅ At least 20 new tests added (7 in Phase 1 + 7 in Phase 2 + ~6 in Phase 3) for `extra_feed`, `render_only`, `_build_*_feed`, deprecation warning, **leaf-side loud-failure** (typo'd `template_key` raises ValueError; empty `template_key` is intentional passthrough), and E2E rendering
10. ✅ The architectural principle is documented in `templated_inferencer_base.py` module docstring
11. ✅ **"No silent failure" invariant is preserved end-to-end**: explicitly-set `template_key` that doesn't resolve raises `ValueError` (loud), at BOTH the Dual layer (Phase 0, already shipped) AND the leaf layer (Phase 1d). No code path falls back to a TemplateManager default for an explicitly-named key.
12. ✅ **Non-templated leaf rejection invariant**: After Phase 4, supplying a non-`TemplatedInferencerBase` instance to Dual's `review_inferencer`/`fixer_inferencer` slots raises `TypeError` at construction. Verified by `test_dual_rejects_non_templated_review_inferencer`.
13. ✅ **`extra_feed` reserved-key invariant**: Passing `extra_feed={"input": ...}` to any leaf raises `ValueError` (silent clobber of `{{ input }}` is impossible). Verified by `test_extra_feed_with_input_key_raises_value_error`.
14. ✅ **Deprecation visibility**: Warning fires both at construction AND at first inference call (per leaf class, deduplicated). Verified by `test_deprecation_warning_fires_at_first_inference`.
15. ✅ **Nested-orchestrator isolation**: Each orchestrator builds its own `extra_feed`; outer `extra_feed` does NOT propagate to nested orchestrators' children. Verified by `test_extra_feed_does_not_propagate_across_nested_orchestrators`.
16. ✅ **3 example YAMLs migrated**: `02_nested.yaml`, `04_shorthand.yaml`, `06_deep_nesting.yaml` all updated to demonstrate leaf-side template configuration. The `<role>_prompt` pattern no longer appears in any tutorial/example. Phase 5 audit re-runs and finds zero remaining usages.

---

## §8 Open Questions (Need Decisions Before Implementation)

| # | Question | Suggested decision |
|---|---|---|
| Q1 | Param name: `extra_feed`, `template_feed`, `feed_overrides`, `extra_template_vars`? | **`extra_feed`** — short, clear, distinguishes from `feed["input"]`; matches Plan B's naming |
| Q2 | Precedence of `extra_feed` vs class-level `template_variables`? | **Above** `template_variables` (caller wins over class defaults), **below** `feed["input"]` (LLM input slot is sacrosanct) |
| Q3 | Non-templated leaf gets `extra_feed`: silent ignore, warn, or raise? | **Warn once per leaf class** (not per call). Logger.warning, not exception. |
| Q4 | `render_only=True` on leaf — should it write to leaf's session log? | **No** by default. Add `_log_render_only` class flag for leaves that DO want it logged. |
| Q5 | Provide `Dual.with_legacy_prompts(...)` opt-in escape hatch? | **No** — users who need it can subclass and override. Keeps API surface minimal. |
| Q6 | What about `initial_prompt`? | **Defer** — different semantics. Audit usages in Phase 3, decide migrate-or-keep then. |
| Q7 | `placeholder_proposal`, `placeholder_main_response` constants — keep or remove? | **Keep** — still useful for SLOT_DEFAULTS-style leaf template feed key naming. |
| Q8 (NEW) | Should `extra_feed` be allowed to override `template_extra_feed` (the class-level static dict)? | **Yes** — caller wins. The `extra_feed` is per-call; the static dict is a class default. Document the precedence in §3.2 of `templated_inferencer_base.py` docstring. |
| Q9 (NEW) | If the `audit_role_prompt_usage.py` script finds an unexpected usage, do we block Phase 1? | **Yes** — block and update the plan. The audit must produce a clean baseline before Phase 1 starts. |
| Q10 (NEW) | After Phase 5 (orchestrator-side rendering removed), what happens if someone supplies a NON-templated leaf (e.g. `MockInferencer`, `ClaudeBedrockInferencer`) as `review_inferencer`/`fixer_inferencer`? | **DECISION: hard requirement at end of Phase 4 (NOT Phase 5).** Phase 4 inserts an `__attrs_post_init__` validation: when `review_inferencer`/`fixer_inferencer` is set, it MUST be `TemplatedInferencerBase` subclass — raise `TypeError` otherwise. Phase 5 then safely deletes `_render_role_prompt` because no code path can reach a non-templated leaf. **Phase 3 is reconciled by removing the `hasattr(leaf, "template_manager")` fallback path entirely** — replaced by the validation in Phase 4. Tests using `MockInferencer` migrate to a new `MockTemplatedInferencer` (thin templated mock) added in Phase 1. **Acceptance criterion #12 (NEW)**: TypeError raised when non-templated leaf supplied to Dual review/fixer slot post-Phase-4. |
| Q11 (NEW) | Should `extra_feed` reject the reserved key `"input"` (which would clobber the user's original prompt)? | **YES — raise `ValueError` at the leaf**. Plan B §4 borrowed: explicitly guarded against in `TemplatedInferencerBase._build_template_feed()`. Test: `test_extra_feed_with_input_key_raises_value_error`. Rationale: silent override of `{{ input }}` would be a bug that's invisible until production. |
| Q12 (NEW) | When orchestrators are nested (Dual containing MFDual containing leaves), how does `extra_feed` flow? | **Each orchestrator builds its own `extra_feed` independently. NO forwarding.** The outer Dual's `extra_feed` for its `review_inferencer` does not flow to nested orchestrators' children. This is consistent with how `template_extra_feed` (class-level) works today and prevents cross-orchestrator coupling. Document explicitly in `templated_inferencer_base.py` docstring as Subtlety 6. |
| Q13 (NEW) | The deprecation warning fires only at construction (in `__attrs_post_init__`). How do we ensure it's "noisy enough" to surface old usages? | **Emit at BOTH construction AND first inference call.** Use a class-level `_warned_at_first_inference` set keyed by leaf-class name — emits once per process per class to avoid log spam, but still surfaces during normal operation (not just startup). YAML loaded once at server startup → construction warning only fires once and gets buried. Adding the first-inference-call warning catches usage patterns that monitoring tools track. |
| Q14 (NEW) | After Phase 2, `Dual` stops rendering and instead passes `extra_feed` to the leaf. But `ConsensusIterationRecord.review_input: str` (defined in `agentic_inferencers/common.py:258`) currently stores the rendered review prompt string. What goes into this field after Phase 2? | **Three options, recommend Option A**: **(A) Pre-render via `render_only=True`** — Dual calls `await review_inferencer.ainfer(input, extra_feed=feed, render_only=True)` once to get the rendered prompt string for the record, then a second call without `render_only` to actually run inference. Cost: 2 calls but second one is the real inference. Cleanest, no record-type change. **(B) Change record type** — `review_input: Union[str, dict]` accepts either rendered string (legacy) or feed dict (post-Phase-2). Cost: schema change ripples to checkpoint serialization, debugging tools, etc. **(C) Drop the field** — record only stores `review_output`. Cost: lose audit trail. **Decision: Option A**. This is precisely WHY `render_only` exists in §4.3b (mentioned but underexplained) — to preserve `ConsensusIterationRecord.review_input` semantics without record-schema changes. Update §4.3b to call out this use case explicitly. |
| Q15 (REVISED in Round 7) | `ConversationalInferencer._render_prompt(self, current_message: str) -> str` overrides `TemplatedInferencerBase._render_prompt` with a custom signature/body. Does our extension break it? | **YES, naive call breaks it.** Verified by direct Python test (Round 7): even when `extra_feed=None` (default for all existing callers), Python raises `TypeError: _render_prompt() got an unexpected keyword argument 'extra_feed'` on the override because LSP enforcement is at the override's signature, not the base class. **My Round-5 claim that "subclasses work unchanged" was wrong.** **Decision (Round 7): conditional-pass at the call site** — `_ainfer_single` and `_infer_single` MUST guard the `extra_feed` parameter: ```python\nif extra_feed is not None:\n    inference_input = self._render_prompt(inference_input, extra_feed=extra_feed)\nelse:\n    inference_input = self._render_prompt(inference_input)\n```\nWhen `extra_feed is None` (today's behavior, all existing callers, including ConversationalInferencer), the call is byte-identical to today's code → zero breakage. When orchestrator explicitly passes `extra_feed={...}`, only templated leaves that accept the kwarg receive it. **Phase 0a audit deliverable amended**: enumerate all `_render_prompt` overrides; for any that wouldn't accept `extra_feed`, the conditional-pass guard is the contract. Future overrides MAY accept `extra_feed` to opt into per-call feed injection; absence is fine.|

---

## §9 Provenance

This plan is the integration of:

- **Plan A** (`leaf_owned_template_rendering_refactor_plan.md`) — provided the spine: explicit `render_only` mode, open-questions discipline, residual-risk + out-of-scope sections, quantified test target
- **Plan B** (`_alt_plan_leaf_rendering_splendid_lantern.md`) — provided the edge-case scenarios table (§3.4) and the explicit `FOLLOWUP_TEMPLATE_DEFAULTS` deliverable (§4.3c)
- **Parallel agent verification ROUND 1** (4 agents, 2026-05-09 14:32) — corrected scope (1 YAML not "all"), added Phase 0a Pre-Flight Audit, verified codebase claims
- **Parallel agent verification ROUND 2** (4 agents, 2026-05-09 14:47) — refined scope (4 YAMLs not 1: 1 production + 3 examples), fixed `is_default_fallback` for `default_template=""` empty-string case, reconciled Q10 with Phase 3 (Phase 4 validation, not Phase 5), added Q11/Q12/Q13 (reserved-key guard, nested isolation, noisy deprecation), added 5 new acceptance criteria (#12-16). All four agents independently selected Plan A as the spine.
- **Critical-thinking ROUND 3 audit** (2026-05-09 14:54) — caught a **CRITICAL API BREAKING BUG** in §4.3b's proposed `ainfer` signature: original draft used `*,` keyword-only barrier that would have dropped `inference_config` as a positional parameter, breaking 5 verified positional callers (inferencer_base.py:1006, :1595, :1632; multi_flow_inferencer.py:1129; openclaw_inferencer.py wrapper). Fix: preserved `inference_config` as 2nd positional, added explicit `**_inference_args` forwarding contract, added explicit "leakage prevention" deliverable (consume `extra_feed`/`render_only` in `_ainfer_single`, never forward to `_ainfer()`). Also: promoted Q11 reserved-key guard from spec-only to concrete `_build_template_feed()` code with `PROTECTED = {"input", "__template_space__"}` collision check. Added 2 backward-compat regression tests.
- **Critical-thinking ROUND 7 audit** (2026-05-09 15:21) — caught a **GENUINE PYTHON LSP BUG in Q15's recommendation**: my Round-5 claim that "subclasses that don't accept `extra_feed` work unchanged because the parameter is optional with default None" was **wrong**. Verified by direct Python test: passing `extra_feed=None` to a subclass override that doesn't declare `extra_feed` raises `TypeError: got an unexpected keyword argument 'extra_feed'` because LSP enforcement is at the override's signature, not the base class. Concrete victim: `ConversationalInferencer._render_prompt(self, current_message: str) -> str` at conversational_inferencer.py:568 — every existing call that hits ConversationalInferencer would crash post-Phase-1. Fix: amended Q15 to acknowledge the mistake, added "Conditional `_render_prompt` call" deliverable to §4.3b that guards the call site with `if extra_feed is not None: ... else: ...` pattern. When `extra_feed is None` (today's behavior, all existing callers), the call is byte-identical to today → zero breakage. When `extra_feed={...}`, it's only passed to overrides that accept it. Architectural lesson learned: **always test Python behavioral claims with a 3-line repro, never trust your intuition about LSP**.
- **Critical-thinking ROUND 6 audit** (2026-05-09 15:14) — verified 3 feedback claims; **2 valid bugs fixed in §4.3d code sketch + 1 false positive rejected (Phase 1c stale text was already fixed in Round 5 — agent re-flagged stale state)**: (Bug 1) `_render_prompt` sketch had `return feed.get("input", "")` referencing undefined `feed` variable (left over from earlier draft when `_render_prompt` took `feed` as first param). Fixed: replaced with `return inference_input` matching production passthrough at templated_inferencer_base.py:282. (Bug 2) `_render_prompt` sketch was missing the production `template_manager is None → passthrough` guard. Without it, leaves whose `template_key` was set via SLOT_DEFAULTS cascade but whose `template_manager` wasn't injected (test mode, cascade skipped) would crash with `AttributeError` on `self.template_manager.get_raw_template(...)`. Fixed: restored both production guards (Guard 1: template_manager None → passthrough; Guard 2: manager set but no key/space → ValueError) above the new "silent default_template fallback" detection. The fix preserves the documented production guard order while adding the Phase 1d invariant.
- **Critical-thinking ROUND 5 audit** (2026-05-09 15:09) — verified 8 feedback claims; **5 valid bugs fixed, 1 false positive rejected (Phase 1c was already fixed in Round 4 — agent re-flagged stale state), 1 style preference declined (`render_only` API surface — kept; serves Q14's pre-render use case)**: (1) `_build_template_feed` code sketch was a REPLACEMENT not an EXTENSION — would have silently dropped `load_variables()`, `_inject_mode_flags_and_content()`, `__template_space__`, and `output_path` injection. Fixed: rewrote sketch as ADDITIVE extension preserving all 6 existing steps from production code (line 121 of templated_inferencer_base.py). (2) `_render_prompt` signature was changed from `(inference_input: Any)` to `(feed: dict)` — would have broken `ConversationalInferencer._render_prompt(self, current_message: str)` override at conversational_inferencer.py:568 + `_ainfer_single` call site. Fixed: kept first-param signature, added optional keyword-only `extra_feed` parameter. (3) `extra_feed` was redundantly on both `_build_template_feed` and `_render_prompt` — collapsed to single entry point. (4) `ConsensusIterationRecord.review_input: str` (common.py:258) post-Phase-2 transition not addressed — added Q14 with Option A recommendation: use `render_only=True` to pre-render for record (this is precisely WHY `render_only` exists, now explicitly documented). (5) `ConversationalInferencer._render_prompt` override existence not audited — added Q15 + Phase 0a audit deliverable to enumerate all `_render_prompt` overrides and verify each tolerates the new optional `extra_feed` kwarg. Stale Phase 1c reference at line 167 also corrected to "Phase 2 wiring".
- **Critical-thinking ROUND 4 audit** (2026-05-09 15:03) — caught a **PHASE-ORDERING REGRESSION** in §4.3c: original draft wired `FOLLOWUP_TEMPLATE_DEFAULTS` into `Dual.SLOT_DEFAULTS` in Phase 1, BEFORE Phase 2 makes Dual stop rendering its own `followup.jinja2`. Verified via `apply_to()` semantics in `template_defaults.py` — `apply_to()` only fills `template_key` if not already present, so wiring SLOT_DEFAULTS auto-cascades `template_key="followup"` onto the fixer leaf. With Dual still rendering followup.jinja2 in Phase 1, this would re-introduce the exact double-rendering bug we manually fixed at the YAML level (Phase 0). Fix: split §4.3c into "Phase 1 = define constant only" + "Phase 2 = wire into SLOT_DEFAULTS atomically with Dual stop-rendering". Added test `test_phase2_no_double_rendering_after_slot_defaults_wiring` to enforce. The transactional invariant (BEFORE/AFTER table) is now documented in §4.3c.
- **Pre-existing infrastructure** in:
  - `templated_inferencer_base.py` (the leaf-side renderer this plan consummates)
  - `template_defaults.py` (the SLOT_DEFAULTS bundles this plan extends)
  - `dual_inferencer.py` (the orchestrator this plan migrates)

The proposal does NOT introduce new abstractions — it completes a half-done migration that was already in flight via `SLOT_DEFAULTS`.
