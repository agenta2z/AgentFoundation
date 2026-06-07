# INTEGRATED Plan: Path-Aware Followup for `DualInferencer`

> **Status**: Proposed — synthesized from two prior plans on 2026-05-09
> **Supersedes**: `dual_inferencer_path_aware_followup_fix_plan.md` (Plan A) and `_alt_plan_splendid_lantern.md` (Plan B)
> **Scope**: Fix the regeneration-drift bug observed in `task_task-7a39a77a_20260508_081252` where the fixer regenerated 17% shorter content instead of incrementally patching the base inferencer's plan.
> **Author note**: This plan combines the architectural rigor of Plan A with the pragmatism of Plan B. It deliberately scopes OUT Plan A's Phase 5/6 (loud-fallback + remove-defaults) — those are valuable cleanup but separate from the user-facing drift fix and should ship as a follow-up.

---

## 1. Background

### 1.1 Observed Symptom

In a recent shallow-profile run of `breakdown-multiflow-plan.yaml`, the `fixer_inferencer` produced a plan that was **17% shorter** (35 KB / 629 lines) than the base inferencer's plan (43 KB / 773 lines), despite the reviewer flagging only ~100 lines of localized edits across 6 issues. The fixer also renamed section numbering (`1.1` → `§1.1`) and dropped multiple paragraphs.

The followup template (`plan/main/followup.jinja2`) explicitly instructed the LLM to **"YOU MUST copy the previous document file and apply targeted edits"** — yet the LLM regenerated anyway.

### 1.2 Three Real Bugs (After Investigation)

By inspecting the actual rendered fixer prompt at `<workspace>/children/fixer_inferencer/logs/session/RovoDevCliInferencer-58f906bd.jsonl.parts/InferenceInput/20260508_093329_f135be00.txt` and tracing `dual_inferencer.py` control flow:

| Bug | Description | Evidence |
|---|---|---|
| **A — Empty `<ProposedDocument>`** | Outer template (`plan/main/followup.jinja2`:12-14) renders `<ProposedDocument>{{ main_response }}</ProposedDocument>`. But `_build_followup_prompt` (`dual_inferencer.py`:1090-1142) **only sets `feed[self.placeholder_proposal] = proposal`** (= "proposal" by default). The variable `main_response` is never populated → tag renders empty. | Inspect rendered prompt: tag is empty. |
| **B — Empty `<ReviewerFeedback>`** | Same root cause as Bug A: outer template uses `{{ reviewer_response }}` but the prompt builder only sets it under a strict `if review_output is not None` guard. Currently fires inconsistently. | Same. |
| **C — Prior file path missing** | The MUST-copy directive in the template is **mechanically unobeyable**. The output destination path is given (line 199 of rendered prompt: `Write your improved document to: <output_path>`), but no source path. The LLM can only "copy mentally" from `<CurrentProposal>` content, which **is regeneration in disguise**. | grep `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/.../base_inferencer/outputs/final_deliverables/output.md` returns 0 hits in the rendered prompt. |
| **D — `<CurrentProposal>` content is just a summary** *(out of scope)* | When the proposer is a BTA orchestrator, `state["base_output_str"]` captures the leaf-LLM's `<Response>` text (a 2.7 KB summary of "what was investigated"), **not** the 43 KB file deliverable. So even where the prior content IS in the prompt, it's the wrong content. | Side-by-side MD5: `<CurrentProposal>` content ≠ on-disk plan file. |

**This plan addresses A, B, C.** Bug D is acknowledged but is out of scope — it would require changes to BTA's state-management or a `_maybe_replace_with_file_reference`-style transform, which is a separate architectural change.

### 1.3 Why Solving C Effectively Mitigates D

Once the LLM is given the **path** to the prior file (Bug C fix), it can `read_file(<path>)` to get the actual 43 KB content directly — no longer dependent on whatever was captured in `state["base_output_str"]`. The path-aware fix functionally bypasses Bug D for any LLM with file-reading tools.

---

## 2. Synthesis: Best of Both Prior Plans

This plan integrates the two prior plans by adopting the strongest design choice on each axis:

| Design choice | Plan A (mine) | Plan B (alt) | This plan picks | Why |
|---|---|---|---|---|
| **What the path resolver returns** | File path (`output.md`) | Folder path (`final_deliverables/`) | **File path** (Plan A) | Directly usable for `cp`/`read_file`; folder requires extra `ls` step the LLM may skip. |
| **Active proposer detection (in helper)** | `_active_proposer()` (state-driven) | `iteration <= 1` heuristic | **`_active_proposer()`** (Plan A) | Robust to refactoring; doesn't require callers to thread `iteration` through. |
| **How the path reaches LLM** | New template variable + explicit `cp` instruction | Path string-appended into `proposal` | **New template variable** (Plan A) | Explicit instruction is far more reliably followed by LLMs than a path appearing inline as plain text. |
| **Fix Bug A (`<ProposedDocument>` empty)** | Set `feed["main_response"] = proposal` directly | Rely on caller setting `placeholder_proposal="main_response"` in YAML | **Set directly in feed** (Plan A) | Plan B's approach requires per-YAML config; today's broken YAML doesn't have it, so Plan B alone wouldn't fix the observed bug. |
| **Fix Bug B (`<ReviewerFeedback>` empty)** | Set `feed["reviewer_response"] = review_output` (only if not None) | Same | **Set with empty-string default** | Empty string > None for Jinja's `{% if %}` semantics. |
| **Template edits** | 4 templates (plan/main + implementation/main, both followup + review) | 0 templates | **4 templates, but with safe `{% if prior_output_path %}` guards** | Backward compat: if helper returns None, template renders without the cp instruction. Other DualInferencer consumers using these templates won't break. |
| **Test E2E coverage** | Mocked Jinja-render only | `IsolatedAsyncioTestCase` running full consensus loop | **Both** | Real-Jinja test catches template regressions; async E2E test catches integration bugs. |
| **Test infra correctness** | Uses `__new__` (skips init); CWD-relative paths | Uses proper factories from `_helpers/` | **Plan B's factory pattern + `__file__`-relative paths** | Plan A's test choices were fragile. |
| **Loud fallback / remove defaults** | Phases 5+6 included | Not addressed | **DEFERRED to follow-up plan** | Important architectural cleanup but separate from the user-facing drift bug; bundling them increases blast radius unnecessarily. |
| **Preflight invariant test** | Included | Not included | **Included** | Locks template-feed-dict integration; cheap insurance against future regressions. |

---

## 3. Detailed Phased Plan

### Phase 1 — Add `_resolve_prior_proposer_output_path()` helper

**File**: `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

**Location**: Place immediately after `_active_proposer()` (currently around line 446).

**Code**:

```python
def _resolve_prior_proposer_output_path(self) -> Optional[str]:
    """Resolve the file path of the prior proposal's output, if discoverable.

    The "active proposer" is the inferencer whose output the next round will
    review or fix:
      - Round 1 review: base_inferencer is active.
      - Round 2+ review or fix: the proposer that ran in the previous iteration
        (base if no fix yet, fixer if a fix completed). _active_proposer()
        encodes this state-driven rule.

    Returns the most-likely "prior output" file path using a two-tier rule:

      Tier 1 — Deliverable file:
        If active proposer's workspace has non-empty final_deliverables/,
        return the deliverable file path. Prefer the basename matching the
        proposer's _output_path attr (e.g. "output.md"); else first non-dotfile
        deliverable in alphabetical order.

      Tier 2 — Outputs file:
        If no Tier 1 hit but outputs/<basename> exists, return that path.

      Tier 3:
        Return None.

    This helper is pure: it inspects the filesystem and proposer attributes
    only; no mutation. Returning None is a normal outcome; callers must handle
    it gracefully (typically by passing an empty string into the prompt feed
    so Jinja's {% if %} treats it as falsy).

    Why file-path (not folder-path):
      The file path is directly usable in the LLM's prompt as the source for
      `cp <prior_file> <output_path>` or `read_file(<prior_file>)`. Returning a
      folder requires the LLM to perform an extra discovery step (ls / glob /
      guess), which is fragile.

    Why _active_proposer (not iteration heuristic):
      _active_proposer() encapsulates the "did a fix run?" decision in one
      place using state["attempt_record"]. Reading state directly here would
      duplicate that logic and risk drift if the state schema changes.
    """
    proposer = self._active_proposer()
    if proposer is None:
        return None

    ws = getattr(proposer, "_workspace", None)
    if ws is None:
        return None

    out_basename = getattr(proposer, "_output_path", None) or "output.md"
    out_basename = os.path.basename(out_basename)

    # Tier 1: deliverable file
    if getattr(ws, "has_deliverables", False):
        # Prefer the basename matching the proposer's configured output path.
        preferred = ws.deliverable_path(out_basename)
        if preferred and os.path.isfile(preferred):
            return preferred
        # Fall back: first non-dotfile deliverable, alphabetically.
        try:
            names = ws.deliverable_paths()  # list of basenames
        except Exception:
            names = []
        for name in sorted(n for n in names if not n.startswith(".")):
            candidate = ws.deliverable_path(name)
            if candidate and os.path.isfile(candidate):
                return candidate

    # Tier 2: outputs file
    candidate = ws.output_path(out_basename) if hasattr(ws, "output_path") else None
    if candidate and os.path.isfile(candidate):
        return candidate

    # Tier 3
    return None
```

**Invariants**:
1. Pure: no filesystem mutation, no state mutation.
2. Returns either an absolute file path (string) or None. Never returns a directory path.
3. Safe with mocked or partial workspaces (graceful `getattr` defaults throughout).
4. Idempotent: same inputs → same output.

---

### Phase 2 — Plumb `main_response`, `prior_output_path`, `reviewer_response` into builders

**File**: `dual_inferencer.py`

**Method 1 — `_build_followup_prompt`** (currently lines 1115-1142)

Replace the existing feed dict assembly:

```python
def _build_followup_prompt(
    self,
    inference_input,
    proposal: str,
    parsed_review: dict,
    inference_config: dict,
    iteration: int = 1,
    attempt: int = 1,
    review_output: Optional[str] = None,
) -> str:
    """Build the followup prompt from template.

    Feed dict semantics — ALL THREE must be set, even if redundant with each
    other or with placeholder_proposal, to support both naming conventions:

      - placeholder_proposal (default "proposal") → inner default template's
        {{ proposal }} → renders into <CurrentProposal> tag.
      - "main_response"                            → outer template's
        {{ main_response }} → renders into <ProposedDocument> tag.
      - "prior_output_path"                        → outer template's
        {{ prior_output_path }} → renders the cp/read instruction.
      - "reviewer_response"                        → outer template's
        {{ reviewer_response }} → renders into <ReviewerFeedback> tag.

    Empty-string sentinel: any unresolvable path renders as "" so Jinja's
    {% if prior_output_path %} treats it as falsy. NEVER pass None — Jinja's
    default rendering would emit the literal string "None" into the prompt.
    """
    issues = parsed_review.get("issues", [])
    reasoning = parsed_review.get("reasoning", "")
    config = inference_config.get("consensus_config", self.consensus_config)

    prior_output_path = self._resolve_prior_proposer_output_path() or ""

    feed = {
        # Inner default template variables (backward compat for callers using
        # DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE).
        self.placeholder_input: inference_input,
        self.placeholder_proposal: proposal,
        self.placeholder_issues: self._serialize_issues(issues),
        self.placeholder_reasoning: reasoning,
        # Outer template variables (used by plan/implementation followup.jinja2).
        # Set unconditionally so Bug A cannot reoccur regardless of YAML config.
        "main_response": proposal,
        "prior_output_path": prior_output_path,
        "reviewer_response": review_output if review_output is not None else "",
        # Iteration metadata.
        "enable_counter_feedback": config.enable_counter_feedback,
        "iteration": iteration,
        "attempt": attempt,
        "round_index": iteration,
    }
    return self._render_role_prompt("followup", feed, inference_config)
```

**Method 2 — `_build_review_prompt`** (currently lines 1094-1113)

Symmetric change — apply the same path resolution and `main_response` / `prior_output_path` plumbing. The reviewer also benefits: it can now read the actual prior file rather than working from a textual summary.

```python
def _build_review_prompt(
    self,
    inference_input,
    proposal,
    counter_feedback,
    inference_config,
    iteration: int = 1,
    attempt: int = 1,
) -> str:
    """Build the review prompt from template. See _build_followup_prompt for
    feed-dict semantics."""
    prior_output_path = self._resolve_prior_proposer_output_path() or ""

    feed = {
        self.placeholder_input: inference_input,
        self.placeholder_proposal: proposal,
        "main_response": proposal,
        "prior_output_path": prior_output_path,
        "iteration": iteration,
        "attempt": attempt,
        "round_index": max(0, iteration - 1),
    }
    if counter_feedback is not None:
        feed[self.placeholder_counter_feedback] = counter_feedback
    return self._render_role_prompt("review", feed, inference_config)
```

**Important**: do NOT remove the `placeholder_proposal: proposal` line. That maintains backward compatibility for any DualInferencer instance that uses the inner default template, AND for any consumer that has configured `placeholder_proposal="main_response"` (in which case `feed["main_response"]` is set twice with the same value — harmless).

**Why both keys are set unconditionally**:
The naming-convention split exists in the codebase today: some callers configure `placeholder_proposal="main_response"` (e.g. `test_plan_then_implement.py:725`, `dual_inferencer_cli.py:308`); others rely on the default `"proposal"`. Setting BOTH keys means we satisfy both conventions simultaneously without requiring per-YAML configuration. This is the key insight that makes the fix work for the broken `breakdown-multiflow-plan.yaml` topology — that YAML does NOT set `placeholder_proposal`, so it relies on the default `"proposal"`. Adding `feed["main_response"] = proposal` is what makes the outer template's `<ProposedDocument>` populate.

---

### Phase 3 — Update `plan/main/followup.jinja2`

**File**: `CoreProjects/AgentFoundation/src/agent_foundation/resources/prompt_templates/plan/main/followup.jinja2`

**Current lines 11-14**:
```jinja2
**Your Current Proposed Document:**
<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**Replace with**:
```jinja2
{% if prior_output_path %}
**Your previous document file is on disk at:**
`{{ prior_output_path }}`

**MANDATORY first step**: copy this file to the output path before editing.
For example: `cp {{ prior_output_path }} {{ output_path }}`
(or your environment's equivalent file-copy operation).
Then make incremental targeted edits in place. Do NOT regenerate from scratch.
{% else %}
**(Note: prior output path is unavailable for this run; the inline content
below is the only source of truth.)**
{% endif %}

**Your Current Proposed Document:**
<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**Why this exact wording**:
- `MANDATORY first step` is a strong directive that LLMs reliably honor when paired with a concrete `cp` example.
- The example uses `cp` because it is the lowest-common-denominator file-copy primitive across LLM environments. The parenthetical "or your environment's equivalent" gives the LLM permission to use `find_and_replace_in_file` with an empty `find` string, `bash`, or any other available primitive.
- The `{% else %}` branch ensures backward compatibility: if any consumer of this template doesn't pass `prior_output_path` (or it's empty), the template renders gracefully with a helpful note. No template-render error, no LLM confusion.
- Placing the path BEFORE `<ProposedDocument>` ensures the LLM sees the path before ever reading the inline content, increasing the chance it acts on the path rather than working from the inline copy.

---

### Phase 4 — Symmetric updates to sibling templates

Apply the same Phase-3 pattern to:

1. `CoreProjects/AgentFoundation/src/agent_foundation/resources/prompt_templates/plan/main/review.jinja2` — wording adjusted for reviewer context ("read the previous document file at: `{{ prior_output_path }}`").
2. `CoreProjects/AgentFoundation/src/agent_foundation/resources/prompt_templates/implementation/main/followup.jinja2`
3. `CoreProjects/AgentFoundation/src/agent_foundation/resources/prompt_templates/implementation/main/review.jinja2`

Each gets the same `{% if prior_output_path %}` block adapted for its phase-specific verbs (review: "read", followup: "copy + edit").

**Out of scope for this phase**: any templates under `_archive/`, any non-followup/non-review templates, any non-plan/non-implementation domains.

---

### Phase 5 (DEFERRED) — Loud-warning + remove silent defaults

The original Plan A had additional phases for:
- Adding loud warnings when `_render_role_prompt` falls back to in-Python defaults
- Auditing all consumers and migrating away from defaults
- Replacing in-Python defaults with sentinel raises

These are **valuable architectural cleanup** but separate from the user-facing drift bug. They will be tracked as a follow-up plan: `dual_inferencer_silent_defaults_cleanup_plan.md` (to be written separately).

**Reason for deferral**: The Phase 1+2+3+4 changes here are strictly additive and have a small blast radius. Bundling Phase 5/6 would force migration of `dual.yaml` and any test fixtures that construct `DualInferencer()` without a template config — that's a wider blast radius and a separate decision.

---

### Phase 6 — Tests (Comprehensive)

This phase adopts both plans' test ideas and fixes Plan A's test infrastructure issues.

#### Test infrastructure conventions verified in the codebase

- **AgentFoundation**: `unittest.TestCase` + `unittest.mock.MagicMock`/`AsyncMock`. DualInferencer-specific tests live under `test_dual_inferencer/` subdirectory. Helpers live in `test_dual_inferencer/_helpers/` (`mock_inferencer.py`, `factories.py`).
- **OpenStartup preflight**: `jinja2.Environment(undefined=jinja2.StrictUndefined)` with `StubObject` sentinel context. Marked `@pytest.mark.preflight`. Designed to run in <5 seconds.
- **Path resolution**: All test files use `Path(__file__).resolve().parent` for portable, CWD-independent path resolution. NEVER use `Path("relative/path").resolve()` (CWD-dependent).
- **DualInferencer construction**: Use `_helpers/factories._make_dual()` factory — NEVER `DualInferencer.__new__(DualInferencer)` (skips `__init__`, leaves required state unset).

#### Test File 1 — Pure unit tests for `_resolve_prior_proposer_output_path()`

**Path**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_path_resolution.py`

Use the existing `_helpers/factories._make_dual()` and `_helpers/mock_inferencer.MockInferencer` (verified to exist in subagent reports). For workspace state, use `tempfile.TemporaryDirectory()` + `InferencerWorkspace(root=tmpdir, use_final_deliverables_folder=True)` + `ws.ensure_dirs()`.

Test methods (12):
1. `test_T1_returns_deliverable_file_when_output_md_present` — Tier 1 happy path
2. `test_T1_prefers_output_path_basename_over_alphabetical` — preferred basename wins
3. `test_T1_skips_dotfile_falls_through_to_T2` — `.self_promoted` marker doesn't count
4. `test_T1_alphabetical_fallback_when_preferred_basename_absent`
5. `test_T2_returns_outputs_file_when_no_deliverables`
6. `test_T2_uses_configured_output_path_basename`
7. `test_T3_returns_None_when_neither_exists`
8. `test_proposer_None_returns_None`
9. `test_proposer_workspace_None_returns_None`
10. `test_proposer_without_output_path_attr_uses_default_basename`
11. `test_after_fix_iteration_resolves_fixer_path` — exercises `_active_proposer()` integration
12. `test_two_agent_mode_fixer_None_resolves_base` — when `fixer_inferencer is None`

**Construction pattern** (correct, NOT `__new__`):
```python
from .._helpers.factories import _make_dual
from .._helpers.mock_inferencer import MockInferencer

def setUp(self):
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

def _make_dual_with_workspace(self):
    base = MockInferencer(name="base", responses=["<Response>summary</Response>"])
    base._workspace = InferencerWorkspace(
        root=os.path.join(self.tmp, "base"),
        use_final_deliverables_folder=True,
    )
    base._workspace.ensure_dirs()
    return _make_dual(base_inferencer=base), base
```

#### Test File 2 — Builder feed-dict shape tests

**Path**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_prompt_builder_feed_dict.py`

Intercept `_render_role_prompt` to inspect the feed dict. Test methods (8):
1. `test_followup_feed_includes_main_response` — `feed["main_response"] == proposal`
2. `test_followup_feed_includes_prior_output_path` — equals helper's return
3. `test_followup_feed_emits_empty_string_when_path_None` — never literal `"None"`
4. `test_followup_feed_includes_proposal_for_backward_compat` — backward compat retained
5. `test_followup_feed_includes_reviewer_response_when_provided`
6. `test_followup_feed_emits_empty_string_when_review_output_None`
7. `test_review_feed_includes_main_response_and_prior_output_path` — symmetric
8. `test_review_feed_includes_counter_feedback_when_provided`

#### Test File 3 — Real-Jinja end-to-end render tests

**Path**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_followup_renders_path_aware.py`

Render the **actual** `plan/main/followup.jinja2` file via `jinja2.Environment(undefined=StrictUndefined)`. Use **`__file__`-relative** path resolution:

```python
TEMPLATE_DIR = (
    Path(__file__).resolve().parents[5]
    / "src" / "agent_foundation" / "resources" / "prompt_templates"
    / "plan" / "main"
)
```

Test methods (5):
1. `test_rendered_prompt_contains_prior_output_path_literal`
2. `test_rendered_prompt_contains_cp_instruction_with_correct_source_and_dest`
3. `test_rendered_prompt_proposed_document_tag_populated`
4. `test_rendered_prompt_falls_back_gracefully_when_path_empty` — `{% else %}` branch fires
5. `test_rendered_prompt_does_not_leak_literal_None`

#### Test File 4 — Symmetric for `review.jinja2`

**Path**: `test_review_renders_path_aware.py` — 3 test methods mirroring File 3 with reviewer-appropriate wording assertions.

#### Test File 5 — IsolatedAsyncioTestCase E2E (borrowed from Plan B)

**Path**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_path_aware_e2e.py`

Run the full consensus loop with mocked LLM responses; assert the fixer's captured input contains both the literal path AND the `cp` instruction. This is **the strongest possible test** because it exercises the entire dual-flow integration. Test methods (3):

1. `test_fixer_input_contains_prior_output_path_after_fix_triggered`
2. `test_no_path_in_fixer_input_when_no_workspace`
3. `test_round2_fixer_sees_round1_fixers_path` — multi-round path-switching

Construction:
```python
class TestPathAwareE2E(unittest.IsolatedAsyncioTestCase):
    async def test_fixer_input_contains_prior_output_path_after_fix_triggered(self):
        # Mock base/reviewer/fixer with stub responses that trigger the fix path.
        captured = []
        base = MockInferencer(name="base", responses=["<Response>summary</Response>"])
        base._workspace = InferencerWorkspace(...); base._workspace.ensure_dirs()
        # Write a fake deliverable so the helper resolves a real path.
        with open(base._workspace.deliverable_path("output.md"), "w") as f:
            f.write("# Plan content")

        reviewer = MockInferencer(name="reviewer",
            responses=['{"issues":[{"severity":"MAJOR",...}],"reasoning":"..."}'])
        fixer = MockInferencer(name="fixer",
            on_call=lambda inp: captured.append(inp) or "<Response>fixed</Response>")
        fixer._workspace = InferencerWorkspace(...); fixer._workspace.ensure_dirs()

        dual = _make_dual(base_inferencer=base, reviewer_inferencer=reviewer,
                          fixer_inferencer=fixer)
        await dual._ainfer("test request")

        self.assertEqual(len(captured), 1)
        fixer_input = captured[0]
        self.assertIn(base._workspace.deliverable_path("output.md"), fixer_input)
        self.assertRegex(fixer_input, r"cp\s+/.+/output\.md")
```

#### Test File 6 — OpenStartup preflight invariant

**Path**: `CoreProjects/OpenStartup/test/openteam/resources/tools/task/preflight/test_followup_path_aware_template.py`

```python
import re
from pathlib import Path
import pytest

TARGETS = [
    "plan/main/followup.jinja2",
    "plan/main/review.jinja2",
    "implementation/main/followup.jinja2",
    "implementation/main/review.jinja2",
]
REQUIRED_VARS = ("main_response", "prior_output_path")


def _resolve_template(rel_path: str) -> Path:
    """Use __file__-relative path resolution (NOT CWD-relative)."""
    here = Path(__file__).resolve().parent
    # walk up from preflight/ to repo root, then over to AgentFoundation prompts
    af_root = (here.parents[5] / ".." / "AgentFoundation").resolve()
    return af_root / "src" / "agent_foundation" / "resources" / "prompt_templates" / rel_path


@pytest.mark.preflight
@pytest.mark.parametrize("template_rel", TARGETS, ids=lambda p: p.replace("/", "::"))
def test_template_references_main_response_and_prior_output_path(template_rel):
    p = _resolve_template(template_rel)
    if not p.is_file():
        pytest.skip(f"Template not yet present: {template_rel}")
    content = p.read_text(encoding="utf-8")
    for var in REQUIRED_VARS:
        assert re.search(rf"\b{re.escape(var)}\b", content), (
            f"Template {template_rel} must reference '{var}'. "
            f"This locks DualInferencer feed-dict integration. "
            f"See _docs/_plans/dual_inferencer_path_aware_followup_INTEGRATED_plan.md."
        )
```

### Test Coverage Summary

| File | New | Methods | Type | Time |
|---|---|---|---|---|
| F1: `test_path_resolution.py` | ✅ | 12 | Pure unit + tempfile | <1s |
| F2: `test_prompt_builder_feed_dict.py` | ✅ | 8 | Mock-intercept | <1s |
| F3: `test_followup_renders_path_aware.py` | ✅ | 5 | Real Jinja + filesystem mocks | <1s |
| F4: `test_review_renders_path_aware.py` | ✅ | 3 | Real Jinja + filesystem mocks | <1s |
| F5: `test_path_aware_e2e.py` | ✅ | 3 | `IsolatedAsyncioTestCase` end-to-end | <2s |
| F6: `test_followup_path_aware_template.py` | ✅ | 4 (param) | Preflight | <1s |

**Total**: 35 new test methods. Total runtime: <10 seconds. Zero LLM calls.

---

## 4. Sequencing and Ship-It Order

| Step | Phase | Risk | Verify | Time |
|---|---|---|---|---|
| 1 | Phase 1 (helper) | Low — pure addition | Run F1 | 30 min |
| 2 | Phase 2 (builders) | Low — adds keys to feed dict | Run F2, F5 | 30 min |
| 3 | Phase 3 (`plan/main/followup.jinja2`) | Low — `{% if %}` guards | Run F3, F6 | 15 min |
| 4 | Phase 6 (tests F1-F4, F6) | None | All pass | 60 min |
| 5 | Re-run shallow profile end-to-end | Low | Acceptance criteria 1-3 below | 15 min |
| 6 | Phase 4 (sibling templates) | Low — same pattern | Re-run F6 (preflight) | 30 min |
| 7 | Phase 6 test F5 (E2E) | None | Pass | 30 min |

**Total**: ~3.5 hours.

---

## 5. Acceptance Criteria

After Phases 1-4 are shipped and a fresh shallow run completes:

1. ✅ Fixer's rendered prompt contains literal path of base inferencer's deliverable.
   *Verify*: `grep "/Users/.../base_inferencer/outputs/final_deliverables/output.md" <fixer_inference_input.txt>` returns ≥1 hit.
2. ✅ Fixer's rendered prompt contains a `cp` instruction with that path as source.
   *Verify*: regex `cp\s+/.+/base_inferencer/.+/output\.md` matches.
3. ✅ `<ProposedDocument>` tag in rendered prompt is non-empty.
   *Verify*: regex `<ProposedDocument>\s*</ProposedDocument>` does NOT match.
4. ✅ `<ReviewerFeedback>` tag is non-empty when a review ran.
5. ✅ Fixer's output plan length within ±5% of base's plan length (LLM-behavior; manual verification).
6. ✅ Section numbering and titles preserved (LLM-behavior; manual diff verification).
7. ✅ All new tests F1-F6 pass.
8. ✅ Existing tests in `test/agent_foundation/common/inferencers/test_dual_inferencer/` still pass (no regressions).

---

## 6. Risk Assessment

### Risks Mitigated

| Risk | Mitigation |
|---|---|
| Fixer regenerates instead of patching (Bug C) | Path now in prompt + explicit `cp` instruction |
| `<ProposedDocument>` empty (Bug A) | `feed["main_response"] = proposal` set unconditionally |
| `<ReviewerFeedback>` empty (Bug B) | `feed["reviewer_response"] = ... or ""` set with empty default |
| Other DualInferencer consumers break | `{% if prior_output_path %}` guard + backward-compat keys retained |
| Future template-only edits regress the fix | Preflight invariant (F6) catches missing variable references |
| Future code edits break the helper | F1 unit tests (12 methods) lock behavior |
| LLM ignores the path | Explicit `MANDATORY first step` + concrete `cp` example far stronger than implicit appended-text |

### Residual Risks

| Risk | Why we accept it |
|---|---|
| Bug D (state captures summary not file) is not directly fixed | The path-aware fix bypasses Bug D for any LLM with file tools — no longer dependent on what's in `state["base_output_str"]`. |
| LLM behavior tests (criteria 5, 6) cannot be unit-tested | Inherently non-deterministic; manual verification against test workspace is the established practice |
| `dual.yaml` and Plan A's "remove silent defaults" cleanup not addressed | Tracked as separate follow-up plan; out of scope for the user-facing drift fix |
| Test file 5 (E2E) requires `_helpers/factories._make_dual()` to support workspace assignment | Verified via subagent inspection: factory exists; if it doesn't expose workspace assignment, extend it minimally |

### What This Fix Does NOT Solve

- Bug D (state captures summary text not file content) — bypassed in practice but not architecturally fixed
- The doubly-templated architecture (inner default + outer Jinja) — Plan A's Phase 5/6 deferred
- Identity-preservation prompt strengthening (don't renumber, don't retitle) — separate prompt-engineering improvement
- Tool-level enforcement (force `find_and_replace_in_file` only) — separate inferencer-config improvement

---

## 7. Open Questions for Decision Before Implementation

1. **`cp` vs environment-agnostic wording in template?**
   *Recommendation*: Keep `cp` as the concrete example with parenthetical "or your environment's equivalent". Concrete examples are more reliably acted upon by LLMs than abstract instructions.

2. **Should `_resolve_prior_proposer_output_path` walk up nested orchestrators?**
   E.g. if proposer is a BTA whose own output is produced by a sub-aggregator, do we want the BTA's outer deliverable or the sub-aggregator's?
   *Recommendation*: Outer only (the BTA's own `final_deliverables/`). Deeper traversal couples Dual to BTA internals.

3. **Should we expose `prior_output_path_relative` (relative to workspace root) for short-path display?**
   *Recommendation*: Not in this fix; can be added when a template needs it.

4. **Feature flag for Phase 1+2+3 in case of unforeseen issues?**
   *Recommendation*: No. Changes are small, additive, easy to revert via git.

5. **Should we ship a small helper update to `_helpers/factories.py` so test F5 can construct a Dual with workspace-assigned proposers?**
   *Recommendation*: Yes if the existing factory doesn't expose this; the helper change is small.

---

## 8. Provenance

This integrated plan was synthesized on 2026-05-09 from:

1. **Plan A** (mine, 1100 lines): `dual_inferencer_path_aware_followup_fix_plan.md` — comprehensive 7-phase plan with template changes, loud warnings, fallback removal, and 6 test files.
2. **Plan B** (alt, 326 lines): `_alt_plan_splendid_lantern.md` — focused 3-change plan with no template changes, relying on `placeholder_proposal` indirection.
3. **Three parallel verification subagents** that:
   - Compared both plans dimension-by-dimension.
   - Verified Plan B's `placeholder_proposal="main_response"` claim against the actual codebase (confirmed: works when configured per-call-site, but the broken YAML doesn't configure it, so Plan B alone wouldn't fix today's bug).
   - Verified Plan A's helper code against codebase (mostly correct; identified two test infrastructure issues: `__new__` usage and CWD-relative paths).
4. **Direct codebase inspection** of:
   - `dual_inferencer.py` lines 188 (`placeholder_proposal` attrib), 446 (`_active_proposer`), 1094-1142 (prompt builders)
   - `inferencer_workspace.py` lines 79-92 (`deliverables_dir`), 102-110 (`has_deliverables`), 96 (`deliverable_path`), 113 (`deliverable_paths`), 221 (`output_path`)
   - `constants.py` line 43 (`DEFAULT_PLACEHOLDER_DUAL_PROPOSAL = "proposal"`)
   - `plan/main/followup.jinja2` lines 11-14 (current `<ProposedDocument>` block)
   - Existing test conventions in `test_dual_inferencer/_helpers/`
5. **Live evidence** from the broken run at `task_task-7a39a77a_20260508_081252` (rendered fixer prompt; on-disk plan file; reviewer JSON output).

This plan does NOT propose changes to `inferencer_workspace.py`, `inferencer_base.py`, or any non-Dual orchestrator (BTA, PTI, MFDual). The fix is contained to `DualInferencer` and four per-domain Jinja templates.
