# Leaf-Owned Template Rendering — Architectural Refactor Plan

**Status**: DRAFT — design proposal pending discussion
**Author**: Co-designed with @tchen7
**Created**: 2026-05-09
**Scope**: AgentFoundation orchestrator inferencers + companion YAMLs in OpenStartup
**Risk**: Medium-to-high (touches widely-used base classes)
**Estimated effort**: 4–6 days across 5 phases (with safety buffers)

---

## §1 Background — The Architectural Tension

### §1.1 What's There Today

Today, **two coexisting template-rendering mechanisms** live in the codebase:

| Mechanism | Owner | Where it renders | When it was added |
|---|---|---|---|
| **Orchestrator-side rendering** | `DualInferencer.review_prompt`, `DualInferencer.followup_prompt`, `MultiFlowDual.multi_flow_followup_prompt`, etc. | Inside the orchestrator class via `_render_role_prompt()` | Older — pre-dated leaf TemplateManager |
| **Leaf-side rendering** | `TemplatedInferencerBase.template_manager` + `template_root_space` + `template_key` | Inside the leaf via the leaf's own `TemplateManager` | Newer — added when leaves got built-in templating |

### §1.2 Concrete Manifestation in YAML

In `breakdown-multiflow-plan.yaml` (and many other topology YAMLs) the user must currently spell out:

```yaml
_target_: Dual
_template_root_space: plan
review_prompt: review            # ← orchestrator-side reference to plan/main/review.jinja2
followup_prompt: followup        # ← orchestrator-side reference to plan/main/followup.jinja2

review_inferencer:
  _target_: ${_params.default_inferencer}
  # template_root_space + template_key cascaded by the SLOT_DEFAULTS mechanism
  # via ``REVIEW_TEMPLATE_DEFAULTS`` (see template_defaults.py:289)

fixer_inferencer:
  _target_: ${_params.default_inferencer}
  # Used to also have ``template_root_space: plan`` + ``template_key: followup``
  # but those were removed in 2026-05 because they caused double-rendering
  # (Dual already renders ``plan/main/followup.jinja2`` → leaf rendering it
  # again on top of Dual's already-rendered output created nested envelope tags
  # and an empty path-aware block on the outer layer).
```

**Both mechanisms claim ownership of the same templates.** Today's rule
(empirically reverse-engineered) is "the orchestrator renders, the leaf consumes
the rendered string as input". But the leaf could ALSO render via its own
`SLOT_DEFAULTS`-cascaded `template_key` — leading to the double-render bug
we just fixed.

### §1.3 Why It's a Problem

| Symptom | Root cause |
|---|---|
| **Bug class — double rendering**: same template invoked at both layers, producing nested envelope tags and empty slots in the outer layer | Two owners for the same artifact |
| **YAML duplication**: `review_prompt: review` on Dual + `template_key: review` cascaded onto leaf both reference `plan/main/review.jinja2` | Two declarations for the same thing |
| **Future fragility**: adding a new orchestrator phase (`critique`, `refine`, `validate`, ...) requires a new `<phase>_prompt` field on the orchestrator class + wiring through `_render_role_prompt` + YAML docs | Phases are tied to orchestrator class hierarchy, not data |
| **Cognitive overhead**: developers must understand which mechanism wins when (today's empirical answer is "orchestrator wins for `review`/`followup`, leaf for everything else") | Two mechanisms with overlapping scope |
| **Debugging difficulty**: when the rendered prompt looks wrong, you must check both Dual's `_render_role_prompt` AND the leaf's `_render_prompt` to find which one produced the artifact | Scattered render path |

### §1.4 What's Already In Place That Helps

The codebase **already has the foundations** for leaf-owned rendering:

| Building block | Status | Where |
|---|---|---|
| Leaf-side TemplateManager + `template_root_space` + `template_key` | ✅ Production | `templated_inferencer_base.py` |
| `SLOT_DEFAULTS` cascading mechanism that auto-fills role-typical template fields onto child slots | ✅ Production | `template_defaults.py`, used by Dual/BTA/MFDual |
| Reusable role-default bundles (`REVIEW_TEMPLATE_DEFAULTS`, `BREAKDOWN_TEMPLATE_DEFAULTS`, `AGGREGATION_TEMPLATE_DEFAULTS`, `FOLLOWUP_AGGREGATION_DEFAULTS`) | ✅ Production | `template_defaults.py` |
| Hydra walker that applies SLOT_DEFAULTS before construction | ✅ Production | `rich_python_utils.config_utils._instantiate` |

**Critical insight**: 50% of the refactor's destination is already shipped.
The work is to **fully migrate from Mechanism A (orchestrator-side) to
Mechanism B (leaf-side, SLOT_DEFAULTS-cascaded)** rather than to invent
new infrastructure.

---

## §2 Proposed Direction

### §2.1 The Architectural Principle

```
ORCHESTRATOR responsibilities:
  • Workflow control (when to invoke review, when to invoke fix, etc.)
  • Feed-dict assembly (compute prior_output_path, gather main_response,
    reviewer_response, output_path, iteration counters, ...)
  • Pass feed dict + role hint to the leaf
  • Capture leaf's response, advance state machine

LEAF responsibilities:
  • Receive feed dict (via inference_input + extra-feed channel)
  • Render its OWN template (using its own TemplateManager) for its
    declared role
  • Execute the LLM call with the rendered prompt
  • Return raw response

OUTCOME:
  • Single source of truth for "which template renders" (the leaf)
  • Single source of truth for "when/why does it render" (the orchestrator)
  • No duplication, no double-rendering bug class possible
```

### §2.2 What Changes

| Field/method | Today | Proposed |
|---|---|---|
| `Dual.review_prompt: Optional[str]` | Renders `plan/main/review.jinja2` via Dual's `_render_role_prompt` | ❌ Removed (deprecated through Phase 4) |
| `Dual.followup_prompt: Optional[str]` | Same for followup | ❌ Removed |
| `Dual.initial_prompt: Optional[str]` | Same for initial-leaf wrapping | ❌ Removed (used much less; verify usages first) |
| `MultiFlowDual.multi_flow_followup_prompt`, `.multi_flow_initial_prompt`, `.multi_flow_aggregator_prompt` | Same pattern | ❌ Removed |
| `MultiFlow.multiflow_followup_prompt`, `.aggregator_prompt` | Same | ❌ Removed |
| `Dual._render_role_prompt(role, feed, config)` | Orchestrator-side render | ❌ Removed |
| `Dual._build_review_prompt(...)` returns rendered string | — | ✅ Renamed to `_build_review_feed(...)` returning a dict |
| `Dual._build_followup_prompt(...)` returns rendered string | — | ✅ Renamed to `_build_followup_feed(...)` returning a dict |
| `LeafInferencer.ainfer(input=str, ...)` | Takes raw input string; if leaf has TemplateManager, treats input as `{{ input }}` slot | ✅ Extended with `extra_feed: Optional[dict]` to receive orchestrator-computed values for the leaf's template feed dict (e.g. `prior_output_path`, `main_response`, `reviewer_response`) |
| `Dual` calls leaf | `leaf.ainfer(input=rendered_str)` | `leaf.ainfer(input=USER_REQUEST_or_similar, extra_feed={"prior_output_path": ..., "main_response": ..., "reviewer_response": ...})` |
| YAML on Dual: `review_prompt: review` | Required if want non-default review template | ❌ Removed — leaf's `template_key` (auto-cascaded by SLOT_DEFAULTS) is the single source of truth |

### §2.3 Why This Is Right (And Where It's Subtle)

#### ✅ Pro 1 — Eliminates the double-render bug class structurally
Only one renderer (the leaf). The bug we fixed in 2026-05 cannot recur.

#### ✅ Pro 2 — Eliminates YAML duplication
`review_prompt: review` and `followup_prompt: followup` lines disappear
from every Dual YAML. A new orchestrator phase requires zero `<phase>_prompt`
fields on the orchestrator — just route the role hint to the right leaf.

#### ✅ Pro 3 — Aligns with existing SLOT_DEFAULTS architecture
This is the direction the codebase was already drifting toward via
`SLOT_DEFAULTS` + `REVIEW_TEMPLATE_DEFAULTS`. We complete the migration
rather than invent something new.

#### ✅ Pro 4 — Future-proofs for new orchestrator phases
Adding a `critique_inferencer` slot becomes: declare a new `CRITIQUE_TEMPLATE_DEFAULTS`,
add it to `SLOT_DEFAULTS`, write `critique.jinja2`. Zero changes to the orchestrator's
attribute surface. Zero new `<phase>_prompt` field. Pure data.

#### ⚠️ Subtlety 1 — Orchestrator sometimes needs the rendered prompt
Today the Dual sometimes uses the rendered prompt for:
- Audit logging (`logger.info("review prompt: %s", rendered)`)
- Caching keys
- Checkpoint serialization

**Mitigation**: The leaf already exposes the rendered prompt internally
(via `_render_prompt()` → produces `inference_input` for the LLM call).
Add a `render_only=True` mode to the leaf so the orchestrator can request
the rendered prompt without invoking the LLM. The leaf returns
`InferenceResult(rendered_prompt=..., response=None)` in that mode.

For audit logging specifically: the leaf already logs its rendered
`inference_input` to its own session log. The orchestrator's "review
input" log can simply point at the leaf's session log path instead of
duplicating the data.

#### ⚠️ Subtlety 2 — Some workflow state must flow into the leaf's feed
Things like `iteration_count`, `attempt_num`, `consensus_state` are
workflow state owned by the orchestrator. The leaf's template (e.g.
`plan/main/review.jinja2`) sometimes wants `{{ iteration_count }}` so
the LLM knows "this is the 3rd attempt".

**Mitigation**: This is fine. The leaf treats them as opaque variables.
The orchestrator passes them via the new `extra_feed` channel. The
leaf's template substitutes them; the leaf doesn't *interpret* them.
Semantic ownership stays with the orchestrator; mechanical substitution
is the leaf's job.

#### ⚠️ Subtlety 3 — Backward compatibility with non-templated leaves
Some leaves (`MockInferencer`, `ClaudeBedrockInferencer`, etc.) don't
inherit from `TemplatedInferencerBase` and have no TemplateManager.

**Mitigation**: For non-templated leaves, the orchestrator falls back
to the old behavior (orchestrator renders and passes string as input).
The migration is opt-in per leaf type. Most production paths use
TemplateManager-equipped leaves anyway (e.g. RovoDevCli, Claude API
with templates, etc.).

#### ⚠️ Subtlety 4 — Some templates legitimately need orchestrator-only context
The `<OriginalUserRequest>` envelope in `plan/main/followup.jinja2`
references `{{ task_preamble }}` and `{{ input }}`. Those are
mechanically the user's original request — orchestrator-computed and
passed via the existing `inference_input` channel. The proposal does
not change how `{{ input }}` works; it adds the `extra_feed` channel
in parallel for orchestrator-computed *additional* feed variables
(`prior_output_path`, `main_response`, `reviewer_response`).

#### ⚠️ Subtlety 5 — `_template_root_space` cascade
Today, `_template_root_space: plan` on the Dual cascades to children
via `SLOT_DEFAULTS`. This is **already the leaf-side mechanism** — no
change needed. The orchestrator's only responsibility is to set the
cascade root; leaves discover their own templates from there.

---

## §3 Phased Plan

The refactor is large enough that a single PR is unsafe. We sequence
into 5 phases, each independently shippable, each adding value without
breaking the next-phase preconditions.

### §3.1 Phase 0 — Already Done (this conversation)

**Status**: ✅ COMPLETE

- Path-aware fix shipped (`prior_output_path` plumbing in Dual + 4 templates)
- Double-rendering bug fixed in `breakdown-multiflow-plan.yaml`
- 50/50 unit tests passing
- "No silent failure" semantic: implicit-default lookup with disable-on-missing
  for `review_prompt`/`followup_prompt`

This phase establishes the safety net (tests + path-aware infrastructure)
for the bigger refactor that follows.

### §3.2 Phase 1 — Add `extra_feed` Channel to Leaf API (Additive Only)

**Goal**: Give leaves a way to receive orchestrator-computed feed
variables in their template feed dict, without changing existing
behavior.

**Deliverables**:

1. **`TemplatedInferencerBase._render_prompt()` extension** (file:
   `templated_inferencer_base.py`):
   - Accept `extra_feed: Optional[dict] = None` parameter
   - Merge `extra_feed` into the rendering feed dict AFTER `template_variables`
     resolution but BEFORE Jinja render
   - Document the precedence rule: `template_variables` < `extra_feed` <
     `feed["input"]` (latter wins for slot collisions)

2. **`InferencerBase.ainfer()` extension** (file: `inferencer_base.py`):
   - Accept `extra_feed: Optional[dict] = None` keyword
   - Forward to `_render_prompt(..., extra_feed=extra_feed)` when leaf is
     templated; ignore (with one-time warning) when leaf is not templated

3. **`InferencerBase.ainfer()` `render_only` mode**:
   - Accept `render_only: bool = False`
   - When True, perform template render and return
     `InferenceResult(rendered_prompt=..., response=None)` without
     invoking the LLM
   - Used by orchestrators that need to log/cache the prompt

4. **Tests**:
   - F1: `extra_feed` correctly merges into Jinja feed dict (mock leaf,
     mock template, verify substitutions)
   - F2: `extra_feed` precedence vs `template_variables` (extra_feed wins)
   - F3: `render_only=True` returns rendered prompt without invoking LLM
   - F4: Backward compat — calling `ainfer(input=...)` without `extra_feed`
     produces identical output to current behavior (regression baseline)

5. **No production code change yet** — orchestrators continue to call
   leaves the old way. This phase is purely additive infrastructure.

**Risk**: 🟢 Low — additive only, no callers depend on the new params.

**Estimated effort**: 1 day (helper + 4 tests + verification)

**Acceptance**: All existing tests pass + 4 new tests pass + at least one
integration test confirming `render_only` works end-to-end.

### §3.3 Phase 2 — Migrate `DualInferencer` to Leaf-Side Rendering

**Goal**: Stop the Dual from rendering `review_prompt`/`followup_prompt`
itself. Instead, build the feed dict and let the leaf render via its own
TemplateManager.

**Deliverables**:

1. **Add `_build_review_feed()` and `_build_followup_feed()` to Dual**:
   - Same signature as today's `_build_review_prompt` / `_build_followup_prompt`
   - Returns a dict (not a rendered string) containing all the
     orchestrator-computed values: `main_response`, `reviewer_response`,
     `prior_output_path`, `output_path`, `iteration`, `attempt_num`, etc.
   - Old `_build_review_prompt` / `_build_followup_prompt` kept as thin
     deprecated wrappers (call `_build_*_feed` then `_render_role_prompt`)

2. **Modify `_step_review_impl` and `_step_fix_impl`**:
   - When `review_inferencer` has a TemplateManager (the new path):
     ```python
     feed = self._build_review_feed(...)
     result = await self.review_inferencer.ainfer(
         input=state["inference_input"],
         extra_feed=feed,
     )
     ```
   - When `review_inferencer` does NOT have a TemplateManager (legacy
     leaves):
     - Fall back to old path (`_render_role_prompt` + ainfer with rendered
       string)
   - Detection: `hasattr(leaf, "template_manager") and leaf.template_manager is not None`

3. **Mark `Dual.review_prompt` / `Dual.followup_prompt` as deprecated**:
   - Add `DeprecationWarning` in `__attrs_post_init__` if either field
     is set explicitly (non-None)
   - Don't remove yet — let YAMLs migrate gradually
   - Add a comment in the field's docstring pointing to the
     leaf-template path

4. **Migrate `breakdown-multiflow-plan.yaml`**:
   - Remove the `review_prompt: review` / `followup_prompt: followup`
     lines
   - Verify the leaf's `template_root_space` + `template_key` cascade
     correctly via `SLOT_DEFAULTS`. Add `template_key: followup` back
     to `fixer_inferencer` (it was removed in Phase 0 to avoid
     double-rendering, but in Phase 2 it becomes the canonical
     mechanism — Dual no longer renders, so the leaf's render is the
     single source of truth).
   - Verify path-aware works end-to-end via the live SOP plan run

5. **Tests**:
   - G1: `_build_review_feed` returns dict with all expected keys
   - G2: `_build_followup_feed` returns dict with all expected keys
     (including `prior_output_path`)
   - G3: Modified `_step_review_impl` calls leaf with `extra_feed` when
     leaf is templated
   - G4: Modified `_step_fix_impl` same
   - G5: Backward compat — leaves WITHOUT TemplateManager still render
     via old path (regression check on tests using MockInferencer)
   - G6: E2E — Dual + templated leaf produces correct rendered prompt
     with `prior_output_path` and all path-aware blocks firing

6. **Update `breakdown-multiflow-plan.yaml`'s comment block**:
   - Document that `template_key: followup` on the leaf is now the
     canonical (and only) renderer; no double-render concern
   - Remove the no-longer-accurate comment about Dual rendering it

**Risk**: 🟡 Medium — touches the most-used orchestrator. Mitigated by
the deprecated-but-still-functional fallback path for legacy leaves.

**Estimated effort**: 1.5 days (code + tests + YAML migration + live verification)

**Acceptance**:
- All existing tests pass
- 6 new tests pass
- Live SOP plan run shows path-aware block in fixer's rendered prompt at
  the leaf level
- Deprecation warnings fire when YAMLs use the old `<role>_prompt` field

### §3.4 Phase 3 — Migrate `MultiFlowDual`, `MultiFlow`, and Other Orchestrators

**Goal**: Apply the same pattern to the other orchestrators that have
`<role>_prompt` fields.

**Scope**:

| Orchestrator | Fields to migrate |
|---|---|
| `MultiFlowDualInferencer` | `multi_flow_aggregator_prompt`, `multi_flow_followup_prompt`, `multi_flow_initial_prompt` |
| `MultiFlowInferencer` | `multiflow_followup_prompt`, `aggregator_prompt` |
| `BreakdownThenAggregateInferencer` | (already partly leaf-side via `aggregator_prompt_builder`; verify and clean up) |
| `PlanThenImplementInferencer` | (audit — uses different mechanism via `prompt_builder` callbacks) |

**Deliverables**:

1. For each orchestrator:
   - Add `_build_<role>_feed()` methods returning dicts
   - Modify step methods to use `extra_feed` channel when leaf is templated
   - Mark `<role>_prompt` fields as deprecated (with warning)
   - Migrate the orchestrator's SLOT_DEFAULTS to include any newly-leaf-
     owned roles (some are already there; check coverage)

2. For each topology YAML using these orchestrators:
   - Audit and remove `<role>_prompt: <key>` lines that are now redundant
   - Ensure leaf children have correct `template_key` (cascaded via
     SLOT_DEFAULTS where possible)

3. **Tests**: Same pattern as Phase 2 — feed-dict tests + step-method
   tests + E2E + backward-compat tests, per orchestrator.

**Risk**: 🟡 Medium — broader surface area, but Phase 2's pattern is
mechanically applicable per orchestrator.

**Estimated effort**: 2 days (3-4 orchestrators × ~0.5 day each)

**Acceptance**:
- Each migrated orchestrator: same acceptance criteria as Phase 2
- All YAMLs in `OpenStartup/topologies/` and `AgentFoundation/test/yaml_configs/`
  audited and updated (or explicitly opted-out)

### §3.5 Phase 4 — Deprecation Period

**Goal**: Give external consumers (other repositories, tests, downstream
notebooks) time to migrate.

**Deliverables**:

1. **Deprecation warnings** are already in place (Phase 2 + 3). Now make
   them louder:
   - Bump from `DeprecationWarning` to `PendingDeprecationWarning` →
     `DeprecationWarning` → `FutureWarning` over time
   - Add a one-time startup banner if any deprecated field is used:
     `"DualInferencer.review_prompt is deprecated; migrate to leaf-side
     rendering. See docs at <link>."`

2. **Migration guide** in `_docs/migrations/`:
   - "How to migrate from `<role>_prompt` to leaf-side rendering"
   - Before/after YAML examples
   - Common pitfalls (e.g. forgetting to set `template_root_space` on the
     leaf; missing `template_key`)

3. **Documentation updates**:
   - `templated_inferencer_base.py` docstring: mention `extra_feed`
   - `template_defaults.py` docstring: mention the new flow
   - Architecture overview doc: update the "two mechanisms" → "one
     mechanism" diagram

**Risk**: 🟢 Low — pure documentation + warning bumps.

**Estimated effort**: 0.5 day

**Acceptance**: Migration guide reviewed; all docs current.

### §3.6 Phase 5 — Remove Deprecated Fields

**Goal**: Final cleanup — delete the now-unused `<role>_prompt` fields
and methods.

**Pre-conditions** (must be true before starting):
- ✅ All known YAMLs migrated
- ✅ All tests pass without using the deprecated fields
- ✅ Deprecation period (Phase 4) elapsed (suggest: 1 release cycle)

**Deliverables**:

1. **Remove from `DualInferencer`**:
   - `review_prompt: Optional[str]` field
   - `followup_prompt: Optional[str]` field
   - `initial_prompt: Optional[str]` field (verify usages first)
   - `_render_role_prompt()` method
   - `_build_review_prompt()` (was a thin wrapper post-Phase 2)
   - `_build_followup_prompt()` (same)
   - `_RoleDisabledError` if no longer needed
   - `DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE` / `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE`
     constants from `constants.py`

2. **Same for `MultiFlowDual`, `MultiFlow`** (Phase 3 orchestrators)

3. **YAML schema updates**:
   - Update YAML linter / Hydra schema (if any) to reject the removed
     fields with a helpful error message

4. **Tests**: Remove the deprecated-field tests; verify no remaining
   imports of the removed constants.

**Risk**: 🟢 Low (since by this point everything's been migrated and
warned for a release cycle).

**Estimated effort**: 0.5 day

**Acceptance**:
- No grep matches for the removed field names in production code
- All tests pass
- YAML configs without the deprecated fields work end-to-end

---

## §4 Implementation Sequencing & Schedule

| Week | Phase | Days | Cumulative |
|---|---|---|---|
| 1 | Phase 1 (extra_feed channel) | 1 | 1 |
| 1 | Phase 2 (Dual migration) | 1.5 | 2.5 |
| 2 | Phase 3 (MFDual + MF migration) | 2 | 4.5 |
| 3 | (Wait for deprecation period) | — | — |
| 4 | Phase 4 (docs + bumped warnings) | 0.5 | 5 |
| 5 | (Wait for deprecation period to expire) | — | — |
| 6 | Phase 5 (final removal) | 0.5 | 5.5 |

**Critical path**: Phases 1 → 2 → 3 must ship in order. Phases 4 → 5
require a deprecation cycle wait between them; how long depends on
release cadence.

**Total active engineering**: ~5.5 days
**Total wall-clock with deprecation periods**: ~6 weeks

---

## §5 Risk Assessment

### §5.1 Mitigated Risks

| Risk | Mitigation |
|---|---|
| Breaking external consumers using `<role>_prompt` | Phase 4 deprecation period + migration guide |
| Breaking tests that mock orchestrator-side rendering | Phase 1's backward-compat fallback (leaves without TemplateManager use old path) |
| Double-rendering re-emerging during transition | Phase 2's YAML migration removes the `<role>_prompt` lines that triggered the bug |
| Loss of orchestrator-side audit logging | Phase 1's `render_only=True` mode preserves the capability |

### §5.2 Residual Risks

| Risk | Why it's residual | Acceptance |
|---|---|---|
| Some orchestrator codepath we haven't audited still relies on the rendered string post-leaf-call | We can grep but can't perfectly verify without exhaustive integration tests | Mitigated by Phase 4 deprecation warnings — anything still relying on the old path will warn |
| Performance regression: `extra_feed` adds dict-merge overhead | Negligible (a few-keys merge per inference call) | Accepted as cost of correctness |
| Hydra walker corner cases when `<role>_prompt` is removed but cascade hasn't reached the leaf yet | Possible if YAMLs have unusual structure | Caught by Phase 2/3 E2E tests; Phase 4's migration guide flags this class of issue |
| Some non-AgentFoundation downstream uses `_render_role_prompt` directly | Unknown without grepping the universe | Phase 4's loud deprecation warning surfaces these in user logs |

### §5.3 Out of Scope

- **PTI's `prompt_builder` mechanism** — different abstraction, not directly
  affected. Audit in Phase 3 to see if it benefits from similar treatment;
  but no direct migration planned.
- **Conversational orchestrators** — already use a different prompt
  paradigm (per-turn). Not in scope.
- **Inferencer caching strategy** — orthogonal concern; this refactor
  doesn't change cache behavior.
- **Adding new roles (critique, refine, validate)** — this refactor enables
  them to be added cleanly, but adding any specific new role is its own
  task.

---

## §6 Acceptance Criteria

The refactor is "done" when ALL of the following are true:

1. ✅ No production code path renders `review`/`followup`/`aggregator`
   templates outside the leaf's TemplateManager
2. ✅ No YAML in `OpenStartup/topologies/` or `AgentFoundation/test/yaml_configs/`
   sets `<role>_prompt` on an orchestrator
3. ✅ The `<role>_prompt` fields and `_render_role_prompt()` method are
   removed from `DualInferencer`, `MultiFlowDualInferencer`, and
   `MultiFlowInferencer`
4. ✅ The `DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE` and
   `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE` constants are removed
5. ✅ The migration guide is published
6. ✅ Live SOP plan run produces a fixer prompt with the path-aware
   block fired at the leaf level (single render, no nesting)
7. ✅ All existing test files pass
8. ✅ At least 15 new tests added across phases (5 per phase 1/2/3) for
   `extra_feed`, `render_only`, `_build_*_feed`, and E2E rendering
9. ✅ The architectural principle is documented in
   `templated_inferencer_base.py` module docstring as the canonical
   source of truth for "leaves render, orchestrators orchestrate"

---

## §7 Open Questions

These need answers before Phase 1 implementation begins:

| # | Question | Suggested decision |
|---|---|---|
| Q1 | What's the precise key name for the new param? `extra_feed`, `template_feed`, `feed_overrides`, `extra_template_vars`? | `extra_feed` — short, clear, distinguishes from `feed["input"]` |
| Q2 | Where does `extra_feed` slot into the precedence order? Above or below `template_variables`? | **Above** `template_variables` (caller wins over class-level defaults), but **below** `feed["input"]` (the LLM's actual input slot is sacrosanct) |
| Q3 | When a leaf is non-templated, should the orchestrator silently fall back to its old render path, or warn? | **Silently fall back** in Phase 2; **warn** starting Phase 3 to encourage migration; **raise** in Phase 5 if anyone tries to set `<role>_prompt` after removal |
| Q4 | Should `render_only=True` write to the leaf's session log? | **No** — it's intended for orchestrator-side audit/cache, not real inference. Add a `_log_render_only=True` flag on the leaf for orchestrators that DO want it logged. |
| Q5 | Should we provide a `Dual.with_legacy_prompts(...)` opt-in escape hatch for users who absolutely need the old render path? | **No** — adds API surface. If someone needs it, they can subclass and override. |
| Q6 | What about `initial_prompt`? It's used to wrap the user's first input. Migrate now, defer, or keep? | **Defer** — `initial_prompt` has different semantics (not part of a workflow loop, just a one-shot initial wrap). Audit its usages in Phase 3 and decide. |
| Q7 | The `placeholder_proposal`, `placeholder_main_response`, etc. constants — keep or remove after migration? | **Keep for now** — they're still useful for SLOT_DEFAULTS-style leaf template feed key naming. |

---

## §8 Provenance

This plan synthesizes:

- The architectural debate in this conversation (2026-05-09 13:51-13:58)
- The path-aware fix already shipped in Phase 0 (this conversation, 2026-05-08 → 2026-05-09)
- Existing infrastructure documented in:
  - `agent_foundation/common/inferencers/templated_inferencer_base.py`
  - `agent_foundation/common/inferencers/template_defaults.py`
  - `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`
- The bug-class analysis from the integrated path-aware plan
  (`dual_inferencer_path_aware_followup_INTEGRATED_plan.md`)

The proposal does NOT introduce new abstractions — it completes a
half-done migration that was already in flight via `SLOT_DEFAULTS`.
