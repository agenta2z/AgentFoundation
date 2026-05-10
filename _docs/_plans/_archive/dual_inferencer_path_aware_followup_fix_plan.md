# Plan: Path-Aware Followup for `DualInferencer` (and Cleanup of Silent Default Templates)

> **Status**: Proposed (not yet implemented)
> **Author**: Codified from rovodev session, 2026-05-09
> **Scope**: `agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer` + per-domain templates under `prompt_templates/{plan,implementation}/main/`
> **Estimated effort**: 90–120 min for Phase 1+2; ~1 day for the full Phase 1–6 cleanup
> **Risk**: Low (Phase 1+2 are additive, backward-compatible). Medium (Phase 4–6 touch consumers).

---

## 1. Background and Problem Statement

### 1.1 Observed Symptom

In a recent shallow-profile run of `breakdown-multiflow-plan.yaml` (workspace `task_task-7a39a77a_20260508_081252`), the `fixer_inferencer` produced a **35 KB plan that was 17% shorter than the base inferencer's 43 KB plan**, despite the reviewer flagging only 6 localized issues (~100 lines of edits across a 773-line document). The fixer also:

- Renamed section numbering (`1.1` → `§1.1` throughout)
- Re-titled the document
- Compressed/dropped content from sections the reviewer did NOT flag

The followup template (`plan/main/followup.jinja2`) explicitly told the LLM to **"copy the previous document file and apply targeted edits"** — first as a soft "try to" suggestion, then later strengthened to "**YOU MUST** copy". Neither stopped the regeneration.

### 1.2 Root Cause — Three Layered Bugs

After deep investigation of the rendered fixer prompt and the Dual control flow (`dual_inferencer.py:_step_fix_impl` line 970, `_build_followup_prompt` line 1090, `_active_proposer` line 440), three real bugs were identified:

#### Bug A — `<ProposedDocument>` is empty in the rendered prompt
`plan/main/followup.jinja2` lines 12–14 expect `{{ main_response }}` to be the prior proposal. But `_build_followup_prompt` in `dual_inferencer.py` only sets `feed[self.placeholder_proposal] = proposal` (the inner default's placeholder) — **it never sets `main_response`**. The outer template's slot silently renders to empty string.

#### Bug B — `<ReviewerFeedback>` is also empty in the rendered prompt
Same root cause as A: `_build_followup_prompt` only sets `feed["reviewer_response"] = review_output` *if* `review_output is not None`. In the observed run, the value was either None or omitted, so the outer template's `<ReviewerFeedback>` rendered empty.

#### Bug C — Prior proposal **file path** is never given to the LLM
The MUST-copy directive in `plan/main/followup.jinja2` is **mechanically unobeyable**: nowhere in the prompt is the LLM told *where* the previous document lives on disk. The destination path (`output_path`, line 199 of the rendered prompt) is provided, but no source path. The LLM's only fallback is to read the prior content from `<CurrentProposal>` (the inner default's tag, populated via `{{ proposal }}` plumbing) and "copy mentally" — which is regeneration in disguise.

#### Bug D (worse-than-A) — Even `<CurrentProposal>` has the wrong content
The 2,669 byte content the fixer actually saw inside `<CurrentProposal>` was a **meta-summary** of what the BTA produced (`"## What Was Investigated… A 773-line consolidated engineering plan…"`), not the 43 KB plan itself. This is because `state["base_output_str"]` (which feeds `proposal`) captures the leaf-LLM's `<Response>` text, not the file deliverable on disk. When the proposer is a BTA/PTI orchestrator that produces a file deliverable + summary text, the summary lands in state but the file content does not.

### 1.3 Architectural Layering Discovered

The Dual ↔ leaf-fixer interaction is **doubly templated** in a way that was not designed but emerged accidentally:

| Layer | Template | Tags |
|---|---|---|
| Inner (Python constant) | `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE` in `constants.py:110` | `<OriginalRequest>`, `<CurrentProposal>`, `<ReviewIssues>`, `<ReviewerReasoning>`, `<ImprovedProposal>` |
| Outer (Jinja file) | `plan/main/followup.jinja2` | `<OriginalUserRequest>`, `<ProposedDocument>`, `<ReviewerFeedback>` |

The Dual renders the inner template internally to produce a `followup_prompt` string. That string is passed to `await self.fixer_inferencer.ainfer(followup_prompt, ...)` (line 998). The leaf fixer then renders the outer Jinja template, **stuffing the inner-rendered string as `{{ input }}`**. Result: `<OriginalUserRequest>` (outer) wraps `<OriginalRequest>` + `<CurrentProposal>` + `<ReviewIssues>` + … (inner). The two layers carry **overlapping but inconsistent** instructions to the LLM:

- Inner says: "produce your improved proposal inside `<ImprovedProposal>` tags"
- Outer says: "wrap a structured `<Response>` JSON"
- Outer NOTES says: "MUST copy the previous file"

The LLM has to reconcile three competing directives. Some confusion is structurally guaranteed.

### 1.4 Why This Is Not Plan-Specific

The same bugs affect `implementation/main/{followup,review}.jinja2`, `_archive/plan/main/`, and `_archive/implementation/main/` — all 8 production + archive templates use `{{ main_response }}` and would benefit from `prior_output_path`. The fix must be **domain-agnostic** because `DualInferencer` is a generic library primitive used across plan, implementation, evaluation, and any future use case.

---

## 2. Proposed Fix — Two-Tier Path Resolution + Wiring + Loud Fallback

### 2.1 Core Idea

Add a single, well-defined source-of-truth helper on `DualInferencer` that resolves the active proposer's on-disk output path using a deterministic two-tier rule:

```
Rule (in priority order):
  1. If proposer's workspace has non-empty final_deliverables/ →
     prior_output_path = the deliverable file path
     (preferred for orchestrators with deliverables: BTA, PTI, MFDual)
  2. Else if proposer's workspace has output_path written →
     prior_output_path = path to outputs/output.md (or configured filename)
     (canonical for leaf inferencers)
  3. Else → prior_output_path = None
     (template renders graceful fallback message)
```

Plumb the resolved path AND the `main_response` / `reviewer_response` data into the leaf fixer's render context so the per-domain Jinja templates' slots fill correctly. Update the templates to use the new `prior_output_path` variable, instructing the LLM to perform a literal `cp` as its first tool action.

### 2.2 Why Path-Only Is Sufficient

The LLM has file-reading and file-editing tools. Once given a path, it can:
- `read_file({{ prior_output_path }})` to inspect the prior content (no need to inline 43 KB into the prompt)
- `cp {{ prior_output_path }} {{ output_path }}` to start from a byte-identical copy
- `find_and_replace_in_file({{ output_path }}, find=..., replace=...)` to apply targeted edits

This converts the MUST-copy directive from "abstract goal" to "executable instruction with concrete first tool call" — which is what was missing today.

---

## 3. Detailed Phased Plan

### Phase 1 — Add `_resolve_prior_proposer_output_path()` helper on `DualInferencer`

**File**: `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

**Location**: Insert near `_active_proposer()` (around line 440).

**Code**:

```python
def _resolve_prior_proposer_output_path(self) -> Optional[str]:
    """Return the on-disk path of the active proposer's output, if any.

    Two-tier resolution rule (deterministic, domain-agnostic):
      1. Tier 1 — Deliverable file (preferred for orchestrators):
         If the active proposer's workspace has non-empty
         final_deliverables/, return the canonical deliverable file path.
         Preference within deliverables_dir:
           a. The file matching the proposer's configured _output_path
              basename (typically "output.md").
           b. Else the first non-dotfile alphabetically.
      2. Tier 2 — Outputs file (canonical for leaf inferencers):
         Else if the proposer's outputs/<basename> exists on disk,
         return that path.
      3. Tier 3 — None: no usable file. Caller should render a
         graceful "previous version unavailable" fallback in the
         template.

    Used by _build_followup_prompt and _build_review_prompt to populate
    the per-domain Jinja templates' {{ prior_output_path }} variable so
    the LLM can perform a literal `cp` of the prior proposal.

    Returns:
        str | None: Absolute path or None.
    """
    proposer = self._active_proposer()
    if proposer is None:
        return None
    ws = getattr(proposer, "_workspace", None)
    if ws is None:
        return None

    # Tier 1: deliverable file (preferred when present)
    if ws.has_deliverables:
        candidate = None
        try:
            preferred_basename = os.path.basename(
                getattr(proposer, "_output_path", None) or "output.md"
            )
            preferred_path = ws.deliverable_path(preferred_basename)
            if preferred_path and os.path.isfile(preferred_path):
                candidate = preferred_path
        except Exception:
            candidate = None
        if not candidate:
            paths = [
                p for p in ws.deliverable_paths()
                if not os.path.basename(p).startswith(".")
            ]
            if paths:
                candidate = os.path.join(ws.deliverables_dir, paths[0])
        if candidate and os.path.isfile(candidate):
            return candidate

    # Tier 2: outputs/<basename> for leaf-style proposers
    out_basename = getattr(proposer, "_output_path", None) or "output.md"
    out_basename = os.path.basename(out_basename)
    out_path = ws.output_path(out_basename)
    if os.path.isfile(out_path):
        return out_path

    # Tier 3: nothing usable
    return None
```

**Invariants**:
- Pure function: filesystem read only, no mutation.
- Returns absolute path (matches `_workspace.output_path` semantics).
- Returns `None` (never raises) on any failure mode (no proposer, no workspace, no files, etc.).
- Tier 1 dotfile filter prevents accidentally returning `.self_promoted` markers.
- Tier 1 preferred-basename selection is deterministic (no `glob` ordering surprises).

**Test cases** (to be added in Phase 6):
1. Proposer with non-empty `final_deliverables/` containing `output.md` → returns the deliverable file path.
2. Proposer with non-empty `final_deliverables/` containing only a dotfile marker → falls through to Tier 2.
3. Proposer with no `final_deliverables/` but with `outputs/output.md` → returns the outputs path.
4. Proposer with neither → returns `None`.
5. Proposer with no `_workspace` → returns `None`.
6. Proposer is `None` (degenerate) → returns `None`.
7. After a fix iteration, `_active_proposer()` returns the fixer; helper resolves the fixer's path.

---

### Phase 2 — Plumb `prior_output_path` and `main_response` into prompt builders

**File**: same as Phase 1.

**Location 2a**: `_build_followup_prompt` (currently lines 1117–1142). Add to the `feed` dict:

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
    # === NEW (path-aware followup) ===
    # Per-domain templates expect these named variables.
    # `main_response` is the prior proposal's textual response (the
    # response_parser-extracted content of state["base_output_str"]).
    # `prior_output_path` is the on-disk location of the prior proposal's
    # canonical output file (deliverable preferred, else outputs/output.md;
    # None if neither exists).
    "main_response": proposal,
    "prior_output_path": self._resolve_prior_proposer_output_path() or "",
}
if review_output is not None:
    feed["reviewer_response"] = review_output
return self._render_role_prompt("followup", feed, inference_config)
```

**Location 2b**: `_build_review_prompt` (currently lines 1095–1113). Add to its `feed` dict:

```python
feed = {
    self.placeholder_input: inference_input,
    self.placeholder_proposal: proposal,
    "iteration": iteration,
    "attempt": attempt,
    "round_index": iteration - 1,
    # === NEW (path-aware review) ===
    "main_response": proposal,
    "prior_output_path": self._resolve_prior_proposer_output_path() or "",
}
if counter_feedback is not None:
    feed[self.placeholder_counter_feedback] = counter_feedback
return self._render_role_prompt("review", feed, inference_config)
```

**Why also set `main_response`**: It is the same string as `proposal`, so this is zero-cost data duplication that fixes Bug A (empty `<ProposedDocument>`). Some templates may eventually use only `main_response` (after the inner-template deprecation in Phase 5+), but until then both names point to the same data without harm.

**Why `or ""` and not raw `None`**: Jinja `{% if prior_output_path %}` correctly treats empty string as falsy, but `None` rendered to template via `{{ prior_output_path }}` would emit the string `"None"` if the if-block is omitted. Empty string is the safest sentinel for "missing".

**Subtle bug to avoid**: Bug B (`reviewer_response` empty in followup) deserves separate investigation. The conditional `if review_output is not None: feed["reviewer_response"] = review_output` is structurally correct; the bug is upstream — `_step_fix_impl` (line 985) passes `review_output=state.get("review_output_str")` which can be `None` if `state["review_output_str"]` was not set. Track this as a follow-up; not blocking.

---

### Phase 3 — Update `plan/main/followup.jinja2` to leverage `prior_output_path`

**File**: `CoreProjects/AgentFoundation/src/agent_foundation/resources/prompt_templates/plan/main/followup.jinja2`

**Location**: Lines 11–14 (the `<ProposedDocument>` block).

**Replace**:

```jinja2
**Your Current Proposed Document:**
<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**With**:

```jinja2
**Your Current Proposed Document:**
{%- if prior_output_path %}
The previous version is saved on disk at:
  `{{ prior_output_path }}`

To obey the MUST-copy directive in §2 / NOTES, your FIRST tool action MUST be:
    cp {{ prior_output_path }} {{ output_path }}

Then apply targeted in-place edits to `{{ output_path }}` (e.g. via
find_and_replace_in_file). DO NOT retype the document from inline content —
copy the file first and edit it byte-for-byte except for the targeted fixes.
{%- else %}
(The previous version path is unavailable in this round — fall back to using
the inline content below as a reference, but be aware that retyping risks
content loss. Try to preserve title, section numbering, and overall length.)
{%- endif %}

<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

**Backward compatibility**: If `prior_output_path` is undefined or empty, the `else` branch renders a graceful fallback message and the `<ProposedDocument>` tag still exists with its prior content (now actually populated thanks to Phase 2's `main_response` plumbing).

---

### Phase 4 — Symmetric updates to sibling templates

Apply the same `{% if prior_output_path %}` block (with appropriate per-template wording) to:

- `plan/main/review.jinja2` (lines 11–14)
- `implementation/main/followup.jinja2` (lines 11–14)
- `implementation/main/review.jinja2` (lines 11–14)

For the **review** templates, the wording should be slightly different — the reviewer's job is to read+critique, not edit. Suggested wording:

```jinja2
**Proposal Under Review:**
{%- if prior_output_path %}
The full proposal artifact is on disk at:
  `{{ prior_output_path }}`
You may inspect it directly via read_file if needed; the inline content
below is for quick context.
{%- endif %}

<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

This gives reviewers the path so they can verify file-only details (line counts, exact whitespace, code in fenced blocks, etc.) without relying solely on the inline content.

**Out of scope**: `_archive/` templates do not need updating — they're explicitly archived.

---

### Phase 5 — Loud-warning when `_render_role_prompt` falls back to in-Python defaults

**File**: `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

**Location**: `_render_role_prompt` method (search for the lines that select between file-based and in-Python templates).

**Add a one-shot warning** (per role, per inferencer instance) when the in-Python `DEFAULT_DUAL_*_PROMPT_TEMPLATE` is used:

```python
def _render_role_prompt(self, role, feed, inference_config):
    template_root_space = ... # existing extraction
    template_key = ...        # existing extraction
    if template_root_space and template_key:
        # File-based template — render via Jinja
        ...
    else:
        # In-Python fallback. Warn LOUDLY (one-shot per role).
        if not getattr(self, f"_warned_default_{role}", False):
            self.log_info(
                (f"DualInferencer at {getattr(self._workspace, 'root', '<no_workspace>')} "
                 f"rendered '{role}' prompt using the in-Python fallback "
                 f"DEFAULT_DUAL_{role.upper()}_PROMPT_TEMPLATE. "
                 f"This is a deprecated path — configure template_root_space "
                 f"and template_key on this inferencer for consistent path-aware "
                 f"prompts and full identity preservation guarantees. "
                 f"See _docs/_plans/dual_inferencer_path_aware_followup_fix_plan.md."),
                "DeprecatedFallbackPrompt",
            )
            setattr(self, f"_warned_default_{role}", True)
        # Render the in-Python default
        ...
```

**Rationale**: Don't break anything yet — just emit a clear, single-warning signal for every consumer that hasn't migrated. This will surface during Phase 6's audit.

---

### Phase 6 — Audit and migrate consumers, then remove the fallback

**6a. Audit**: Run all integration + unit tests with the Phase 5 warning in place. Collect a list of every `DualInferencer` instance that triggers the warning. Source candidates to inspect:

- `CoreProjects/OpenStartup/src/openteam/server/resources/tools/task/topologies/dual.yaml` (already known: 0 references to `template_root_space`)
- All test fixtures that construct `DualInferencer()` directly without specifying templates
- Any external `agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer.DualInferencer(...)` instantiation in client code

**6b. Migration** (per consumer):
- For YAML configs missing templates: add `template_root_space` and `template_key` referencing an appropriate Jinja template (or create a new domain-agnostic one in `prompt_templates/_dual/main/` if needed).
- For test fixtures: either (a) explicitly pass `template_root_space`/`template_key`, or (b) accept the warning if the test is intentionally exercising the fallback.

**6c. Removal** — once all warnings are eliminated:
- Replace `DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE` and `DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE` in `constants.py` with sentinel strings that raise `NotImplementedError` if accidentally used at render time:

  ```python
  DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE = (
      "[ERROR: This in-Python fallback was deprecated in 2026-05. "
      "All DualInferencer instances must configure template_root_space "
      "and template_key. See _docs/_plans/dual_inferencer_path_aware_followup_fix_plan.md.]"
  )
  ```

- In `_render_role_prompt`, replace the fallback path with `raise MissingTemplateError("DualInferencer at <ws> has no template_root_space/template_key configured")`.
- Move the original templates to a docstring or `_archive/` for historical reference.

---

### Phase 7 — Tests (Comprehensive)

This phase has been significantly expanded after auditing existing test conventions:

- **AgentFoundation conventions** (from `test/agent_foundation/common/inferencers/`):
  - Tests use `unittest.TestCase` + `unittest.mock.MagicMock`/`AsyncMock` for fixtures
  - DualInferencer-specific tests live under `test_dual_inferencer/` subdirectory
  - Existing `test_supports_prompt_rendering.py` shows the established pattern for testing `_render_role_prompt`-adjacent behavior — extends `unittest.TestCase`, uses MagicMock proposers, runs async via a `_run_async` helper
- **OpenStartup preflight conventions** (from `test/openteam/resources/tools/task/preflight/`):
  - Pure-Jinja template tests use `jinja2.Environment(undefined=jinja2.StrictUndefined)` with `StubObject` sentinel context (see `test_jinja_render_all_templates.py`)
  - Marked `@pytest.mark.preflight` and parametrized over discovered templates
  - Designed to run in <5 seconds without LLM calls

The plan adds **5 new test files** plus **1 modification to an existing file**:

---

#### Test File 1 — Pure unit tests for `_resolve_prior_proposer_output_path()`

**Location**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_path_resolution.py`

**Convention**: `unittest.TestCase` + `MagicMock` + `tempfile.TemporaryDirectory` for filesystem fixtures.

**Why a separate file**: Path resolution has its own surface area (filesystem state, tier rules, edge cases) and deserves isolation from prompt-builder tests.

**Test class outline**:

```python
"""Unit tests for DualInferencer._resolve_prior_proposer_output_path().

Covers the two-tier rule + edge cases. Pure unit tests — no LLM calls,
no real workspaces, just MagicMock proposers and tempfile-backed dirs.
"""
from __future__ import annotations
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


class _MockWorkspace:
    """Minimal workspace stub matching the API _resolve_prior_proposer_output_path uses."""

    def __init__(self, root: str, deliverables_subdir: str | None = "final_deliverables"):
        self.root = root
        self.outputs_dir = os.path.join(root, "outputs")
        self.deliverables_dir = (
            os.path.join(self.outputs_dir, deliverables_subdir)
            if deliverables_subdir else None
        )
        os.makedirs(self.outputs_dir, exist_ok=True)
        if self.deliverables_dir:
            os.makedirs(self.deliverables_dir, exist_ok=True)

    @property
    def has_deliverables(self):
        d = self.deliverables_dir
        return bool(d and os.path.isdir(d) and os.listdir(d))

    def deliverable_path(self, relative):
        return os.path.join(self.deliverables_dir, relative) if self.deliverables_dir else None

    def deliverable_paths(self):
        if not self.deliverables_dir or not os.path.isdir(self.deliverables_dir):
            return []
        return sorted(os.listdir(self.deliverables_dir))

    def output_path(self, relative):
        return os.path.join(self.outputs_dir, relative)


def _make_dual_with_proposer(proposer):
    """Build a minimal DualInferencer instance with a mocked _active_proposer."""
    dual = DualInferencer.__new__(DualInferencer)  # bypass __init__ side effects
    dual.base_inferencer = proposer
    dual.fixer_inferencer = None
    dual._state = {}  # so _active_proposer() returns base_inferencer (no iterations)
    return dual


class TestResolvePriorProposerOutputPath(unittest.TestCase):
    """Cover all 3 tiers + 4 edge cases."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    # ---- TIER 1: deliverable file ----
    def test_T1_returns_deliverable_when_output_md_present(self):
        """Tier 1: proposer.deliverables_dir contains output.md → returns it."""
        ws = _MockWorkspace(self.tmp)
        deliv = os.path.join(ws.deliverables_dir, "output.md")
        with open(deliv, "w") as f: f.write("# plan content")
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), deliv)

    def test_T1_prefers_output_path_basename_over_alphabetical(self):
        """If multiple deliverables exist, the proposer's _output_path wins."""
        ws = _MockWorkspace(self.tmp)
        for fn in ("a_first_alphabetically.md", "output.md", "z_last.md"):
            with open(os.path.join(ws.deliverables_dir, fn), "w") as f: f.write("x")
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.deliverables_dir, "output.md"),
        )

    def test_T1_skips_dotfile_and_falls_through_when_only_dotfile(self):
        """Dotfile-only deliverables_dir should NOT count as a Tier-1 hit."""
        ws = _MockWorkspace(self.tmp)
        # Only the .self_promoted marker — no real deliverable
        with open(os.path.join(ws.deliverables_dir, ".self_promoted"), "w") as f: f.write("")
        # But there IS a Tier-2 fallback file in outputs/
        with open(os.path.join(ws.outputs_dir, "output.md"), "w") as f: f.write("x")
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.outputs_dir, "output.md"),
        )

    def test_T1_alphabetical_fallback_when_preferred_basename_absent(self):
        """If preferred basename is missing but other deliverables exist."""
        ws = _MockWorkspace(self.tmp)
        with open(os.path.join(ws.deliverables_dir, "report.md"), "w") as f: f.write("x")
        with open(os.path.join(ws.deliverables_dir, "summary.json"), "w") as f: f.write("{}")
        proposer = MagicMock(_workspace=ws, _output_path="not_present.md")
        dual = _make_dual_with_proposer(proposer)
        # report.md is alphabetically first non-dotfile
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.deliverables_dir, "report.md"),
        )

    # ---- TIER 2: outputs file ----
    def test_T2_returns_outputs_md_when_no_deliverables(self):
        """Tier 2: no deliverables but outputs/output.md exists."""
        ws = _MockWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "output.md")
        with open(out, "w") as f: f.write("# plan")
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    def test_T2_uses_configured_output_path_basename(self):
        """If proposer's _output_path is custom (e.g. 'my_report.md'), use it."""
        ws = _MockWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "my_report.md")
        with open(out, "w") as f: f.write("x")
        proposer = MagicMock(_workspace=ws, _output_path="my_report.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    # ---- TIER 3: None ----
    def test_T3_returns_None_when_neither_exists(self):
        """No deliverables, no outputs file → None."""
        ws = _MockWorkspace(self.tmp, deliverables_subdir=None)
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual = _make_dual_with_proposer(proposer)
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    # ---- EDGE CASES ----
    def test_proposer_None_returns_None(self):
        dual = _make_dual_with_proposer(None)
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_without_workspace_returns_None(self):
        proposer = MagicMock(spec=[])  # no _workspace attribute
        dual = _make_dual_with_proposer(proposer)
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_workspace_None_returns_None(self):
        proposer = MagicMock(_workspace=None)
        dual = _make_dual_with_proposer(proposer)
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_without_output_path_attr_uses_default_basename(self):
        """No _output_path → defaults to 'output.md'."""
        ws = _MockWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "output.md")
        with open(out, "w") as f: f.write("x")
        proposer = MagicMock(spec=["_workspace"])  # no _output_path
        proposer._workspace = ws
        dual = _make_dual_with_proposer(proposer)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    # ---- ACTIVE PROPOSER SWITCHING ----
    def test_after_fix_iteration_resolves_fixer_path(self):
        """After a fix iteration, _active_proposer() returns fixer; helper resolves fixer's path."""
        # Setup: base + fixer, both with own workspaces, fix already ran
        base_ws = _MockWorkspace(os.path.join(self.tmp, "base"))
        fixer_ws = _MockWorkspace(os.path.join(self.tmp, "fixer"))
        # Both have deliverable files
        base_deliv = os.path.join(base_ws.deliverables_dir, "output.md")
        fixer_deliv = os.path.join(fixer_ws.deliverables_dir, "output.md")
        with open(base_deliv, "w") as f: f.write("base plan")
        with open(fixer_deliv, "w") as f: f.write("fixer plan")
        base = MagicMock(_workspace=base_ws, _output_path="output.md")
        fixer = MagicMock(_workspace=fixer_ws, _output_path="output.md")

        dual = DualInferencer.__new__(DualInferencer)
        dual.base_inferencer = base
        dual.fixer_inferencer = fixer
        # Simulate a state where last iteration had counter_feedback (fix ran)
        dual._state = {
            "attempt_record": {
                "iterations": [
                    {"counter_feedback": "needs revision"},
                ]
            }
        }
        # Helper should now return fixer's deliverable path (not base's)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), fixer_deliv)
```

**Coverage**: 12 test methods covering all 3 tiers + 4 edge cases + the active-proposer switching guarantee.

---

#### Test File 2 — Unit tests for `_build_followup_prompt` and `_build_review_prompt` feed dicts

**Location**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_prompt_builder_feed_dict.py`

**Convention**: `unittest.TestCase` + `MagicMock` to intercept `_render_role_prompt` and assert what feed dict it was called with.

**Why intercepting `_render_role_prompt`**: We don't want to test the actual Jinja rendering here (that's File 4 below) — we want to test that the prompt builders **construct the right feed dict** to pass downstream.

**Test class outline**:

```python
"""Unit tests asserting _build_followup_prompt / _build_review_prompt
construct the correct feed dict (containing main_response, prior_output_path, etc.)
"""
from unittest.mock import MagicMock, patch
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


class TestBuildFollowupPromptFeed(unittest.TestCase):

    def _make_dual(self, prior_path=None):
        dual = DualInferencer.__new__(DualInferencer)
        dual.base_inferencer = MagicMock(_workspace=None)
        dual.fixer_inferencer = None
        dual._state = {}
        # Stub out the placeholder names + serializer + helper
        dual.placeholder_input = "input"
        dual.placeholder_proposal = "proposal"
        dual.placeholder_issues = "issues"
        dual.placeholder_reasoning = "reasoning"
        dual.placeholder_counter_feedback = "counter_feedback"
        dual.consensus_config = MagicMock(enable_counter_feedback=False)
        dual._serialize_issues = lambda issues: "ISSUES_STR"
        dual._resolve_prior_proposer_output_path = MagicMock(return_value=prior_path)
        dual._render_role_prompt = MagicMock(return_value="RENDERED_PROMPT")
        return dual

    def test_followup_feed_includes_main_response_and_prior_output_path(self):
        dual = self._make_dual(prior_path="/tmp/x/output.md")
        dual._build_followup_prompt(
            inference_input="INPUT",
            proposal="PROPOSAL_TEXT",
            parsed_review={"issues": [], "reasoning": "ok"},
            inference_config={},
            iteration=1,
            attempt=1,
            review_output="REVIEW_RAW",
        )
        called_feed = dual._render_role_prompt.call_args[0][1]
        self.assertEqual(called_feed["main_response"], "PROPOSAL_TEXT")
        self.assertEqual(called_feed["prior_output_path"], "/tmp/x/output.md")
        self.assertEqual(called_feed["reviewer_response"], "REVIEW_RAW")
        # Backward-compat: inner placeholder_proposal is also still set
        self.assertEqual(called_feed["proposal"], "PROPOSAL_TEXT")

    def test_followup_feed_emits_empty_string_when_path_none(self):
        """prior_output_path=None should render as '' so {% if %} is falsy."""
        dual = self._make_dual(prior_path=None)
        dual._build_followup_prompt(
            inference_input="i", proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
        )
        called_feed = dual._render_role_prompt.call_args[0][1]
        self.assertEqual(called_feed["prior_output_path"], "")
        self.assertNotEqual(called_feed["prior_output_path"], None)
        # Importantly, NOT the string "None"
        self.assertNotEqual(called_feed["prior_output_path"], "None")

    def test_followup_omits_reviewer_response_when_review_output_none(self):
        """reviewer_response should NOT be in feed if review_output is None."""
        dual = self._make_dual()
        dual._build_followup_prompt(
            inference_input="i", proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
            review_output=None,
        )
        called_feed = dual._render_role_prompt.call_args[0][1]
        self.assertNotIn("reviewer_response", called_feed)


class TestBuildReviewPromptFeed(unittest.TestCase):
    """Symmetric tests for review prompt builder."""

    def _make_dual(self, prior_path=None):
        # … same as above
        pass

    def test_review_feed_includes_main_response_and_prior_output_path(self):
        ...

    def test_review_feed_omits_counter_feedback_when_none(self):
        ...
```

**Coverage**: 5–6 test methods asserting feed-dict shape, empty-string sentinel for missing paths, and non-leakage of None.

---

#### Test File 3 — Mock end-to-end test demonstrating rendered prompt is correct

**Location**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_followup_renders_path_aware.py`

**Convention**: Combines real Jinja rendering (using actual `plan/main/followup.jinja2`) with mocked DualInferencer + mocked filesystem-backed proposer workspace. This is the **MOST IMPORTANT** new test — it demonstrates that the full chain (helper → feed-dict → Jinja render) produces a prompt with the expected substrings.

**Why this test matters most**: It catches the original bug as a regression. If anyone ever again removes `prior_output_path` from the feed dict, or removes the `{% if prior_output_path %}` block from the template, this test fails LOUDLY.

**Test outline**:

```python
"""End-to-end mock test: prove the rendered followup prompt contains
the prior plan's path, the cp instruction, and the populated <ProposedDocument>
tag. This is the regression test for the empty-tag + missing-path bug.

Uses real Jinja templates + real DualInferencer code; only the LLM and the
proposer workspace state are mocked.
"""
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import jinja2

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)

# Real template path
TEMPLATE_DIR = Path(
    "agent_foundation/resources/prompt_templates/plan/main"
).resolve()
FOLLOWUP_TEMPLATE = TEMPLATE_DIR / "followup.jinja2"


class TestFollowupRendersPathAware(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        # Set up a fake base_inferencer workspace with a deliverable file
        self.base_ws_root = os.path.join(self.tmp, "base_inferencer")
        self.base_outputs = os.path.join(self.base_ws_root, "outputs")
        self.base_deliv_dir = os.path.join(self.base_outputs, "final_deliverables")
        os.makedirs(self.base_deliv_dir)
        self.base_deliverable = os.path.join(self.base_deliv_dir, "output.md")
        with open(self.base_deliverable, "w") as f:
            f.write("# Plan title\n\n## 1. Section\nSubstantial 50-line plan content...")

    def tearDown(self):
        self._td.cleanup()

    def _build_real_dual(self):
        """Build a Dual with real prompt-rendering wiring + mocked LLM proposer."""
        dual = DualInferencer.__new__(DualInferencer)
        # Mock proposer with real workspace
        ws = MagicMock(
            root=self.base_ws_root,
            outputs_dir=self.base_outputs,
            deliverables_dir=self.base_deliv_dir,
            has_deliverables=True,
        )
        ws.deliverable_path = lambda rel: os.path.join(self.base_deliv_dir, rel)
        ws.deliverable_paths = lambda: ["output.md"]
        ws.output_path = lambda rel: os.path.join(self.base_outputs, rel)
        proposer = MagicMock(_workspace=ws, _output_path="output.md")
        dual.base_inferencer = proposer
        dual.fixer_inferencer = None
        dual._state = {}
        dual.placeholder_input = "input"
        dual.placeholder_proposal = "proposal"
        dual.placeholder_issues = "issues"
        dual.placeholder_reasoning = "reasoning"
        dual.placeholder_counter_feedback = "counter_feedback"
        dual.consensus_config = MagicMock(enable_counter_feedback=False)
        dual._serialize_issues = lambda issues: "{stub_issues}"
        return dual, proposer

    def _render_followup_template(self, feed):
        """Render the actual plan/main/followup.jinja2 file with the given feed."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
            undefined=jinja2.StrictUndefined,
        )
        # Add any defaults the template needs that we're not providing in feed
        feed.setdefault("task_preamble", "")
        feed.setdefault("output_path", "/tmp/fixer_output.md")
        feed.setdefault("enable_elegant_mode", False)
        feed.setdefault("instructions", MagicMock())
        return env.get_template("followup.jinja2").render(**feed)

    def test_rendered_prompt_contains_prior_output_path(self):
        """The exact path to the base deliverable must appear in the rendered prompt."""
        dual, proposer = self._build_real_dual()
        # Capture the feed by intercepting _render_role_prompt
        with patch.object(dual, "_render_role_prompt", side_effect=lambda role, feed, cfg: feed):
            feed = dual._build_followup_prompt(
                inference_input="USER REQUEST",
                proposal="PRIOR PROPOSAL TEXT",
                parsed_review={"issues": [], "reasoning": "ok"},
                inference_config={},
                review_output="REVIEW RAW",
            )
        self.assertEqual(feed["prior_output_path"], self.base_deliverable)

        # Now render with the actual template
        rendered = self._render_followup_template(feed)

        # MUST contain the literal path
        self.assertIn(self.base_deliverable, rendered,
                      "Rendered prompt must contain the prior deliverable path")
        # MUST contain a cp instruction
        cp_pattern = rf"cp\s+{re.escape(self.base_deliverable)}"
        self.assertRegex(rendered, cp_pattern,
                         "Rendered prompt must contain `cp <prior_path> ...`")
        # <ProposedDocument> tag is populated, not empty
        proposed_doc = re.search(
            r"<ProposedDocument>\s*(.*?)\s*</ProposedDocument>",
            rendered, re.DOTALL,
        )
        self.assertIsNotNone(proposed_doc)
        self.assertIn("PRIOR PROPOSAL TEXT", proposed_doc.group(1),
                      "<ProposedDocument> must be populated with main_response")
        # <ReviewerFeedback> tag is populated, not empty
        reviewer_fb = re.search(
            r"<ReviewerFeedback>\s*(.*?)\s*</ReviewerFeedback>",
            rendered, re.DOTALL,
        )
        self.assertIsNotNone(reviewer_fb)
        self.assertIn("REVIEW RAW", reviewer_fb.group(1),
                      "<ReviewerFeedback> must be populated with reviewer_response")

    def test_rendered_prompt_renders_fallback_when_path_unavailable(self):
        """If prior_output_path is empty, template uses graceful fallback (no `cp` line)."""
        dual, proposer = self._build_real_dual()
        # Force the resolver to return None
        proposer._workspace.has_deliverables = False
        # And no outputs/output.md exists either
        with patch.object(dual, "_render_role_prompt", side_effect=lambda role, feed, cfg: feed):
            feed = dual._build_followup_prompt(
                inference_input="USER REQUEST",
                proposal="PRIOR PROPOSAL TEXT",
                parsed_review={"issues": [], "reasoning": "ok"},
                inference_config={},
            )
        self.assertEqual(feed["prior_output_path"], "")
        rendered = self._render_followup_template(feed)
        self.assertNotIn("cp ", rendered)  # no cp instruction
        self.assertIn("path is unavailable", rendered.lower())  # fallback rendered
        # Inline content still rendered as fallback
        self.assertIn("PRIOR PROPOSAL TEXT", rendered)

    def test_rendered_prompt_does_not_contain_literal_None(self):
        """Sanity: the empty-string sentinel prevents 'None' from ever leaking into prompt."""
        dual, proposer = self._build_real_dual()
        proposer._workspace.has_deliverables = False
        with patch.object(dual, "_render_role_prompt", side_effect=lambda role, feed, cfg: feed):
            feed = dual._build_followup_prompt(
                inference_input="x", proposal="y",
                parsed_review={"issues": [], "reasoning": ""},
                inference_config={},
            )
        rendered = self._render_followup_template(feed)
        self.assertNotIn("None", rendered.split("<ProposedDocument>")[0])  # at least in the non-content sections

    def test_acceptance_criteria_E1_to_E3(self):
        """Rolled-up assertion of acceptance criteria 1, 2, 3 from §6 of the plan."""
        dual, proposer = self._build_real_dual()
        with patch.object(dual, "_render_role_prompt", side_effect=lambda role, feed, cfg: feed):
            feed = dual._build_followup_prompt(
                inference_input="REQ", proposal="PRIOR",
                parsed_review={"issues": [], "reasoning": ""},
                inference_config={},
                review_output="REVIEW",
            )
        rendered = self._render_followup_template(feed)
        # E1: contains literal path
        self.assertIn(self.base_deliverable, rendered)
        # E2: contains cp instruction with that path as source
        self.assertRegex(rendered, rf"cp\s+{re.escape(self.base_deliverable)}\s+/tmp/fixer_output\.md")
        # E3: <ProposedDocument> populated
        self.assertNotRegex(rendered, r"<ProposedDocument>\s*</ProposedDocument>")
```

**Coverage**: 4 test methods that prove the rendering is appropriate end-to-end:
- T1: full success path — prior_output_path present → cp instruction + populated tags
- T2: graceful fallback when path unavailable → no cp line, fallback wording, inline content used
- T3: no literal `"None"` leaks into prompt
- T4: rolled-up acceptance criteria E1–E3

---

#### Test File 4 — Symmetric mock test for `review.jinja2`

**Location**: `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_review_renders_path_aware.py`

Same pattern as File 3 but for `plan/main/review.jinja2`. Coverage: 2–3 methods (review-template wording differs slightly from followup, so we explicitly assert the reviewer-appropriate phrasing).

---

#### Test File 5 — OpenStartup preflight: assert template variable presence

**Location**: `CoreProjects/OpenStartup/test/openteam/resources/tools/task/preflight/test_followup_path_aware_template.py`

**Convention**: Follows `test_yaml_deliverable_flags_set.py` and `test_jinja_render_all_templates.py` patterns. Marked `@pytest.mark.preflight`.

**Why preflight**: This is exactly the kind of "constraint that must hold across the codebase" check that preflight is for. Locks the integration so future template-only edits can't silently regress.

```python
"""PREFLIGHT: every plan/implementation followup+review template must reference
both {{ main_response }} and {{ prior_output_path }}.

This locks the integration with DualInferencer's prompt builder feed dict
(which provides both variables). Removing either reference re-introduces the
empty-<ProposedDocument> bug or the unobeyable MUST-copy-without-path bug.
"""
from pathlib import Path
import re
import pytest

# Auto-discover all 4 templates that should be path-aware
TARGETS = [
    "plan/main/followup.jinja2",
    "plan/main/review.jinja2",
    "implementation/main/followup.jinja2",
    "implementation/main/review.jinja2",
]

REQUIRED_VARS = ("main_response", "prior_output_path")


def _resolve_template(rel_path: str) -> Path:
    here = Path(__file__).resolve().parent
    # walk up to repo root then into AgentFoundation prompts
    af_prompts = (
        here.parents[5] / ".." / "AgentFoundation" / "src" / "agent_foundation"
        / "resources" / "prompt_templates"
    ).resolve()
    return af_prompts / rel_path


@pytest.mark.preflight
@pytest.mark.parametrize("template_rel", TARGETS, ids=lambda p: p.replace("/", "::"))
def test_template_references_main_response_and_prior_output_path(template_rel):
    p = _resolve_template(template_rel)
    if not p.is_file():
        pytest.skip(f"Template not present (deferred): {template_rel}")
    content = p.read_text(encoding="utf-8")
    for var in REQUIRED_VARS:
        # Look for {{ var }} or {% ... var ... %} reference
        pattern = rf"\b{re.escape(var)}\b"
        assert re.search(pattern, content), (
            f"Template {template_rel} must reference '{var}'. "
            f"This locks DualInferencer feed-dict integration. "
            f"See _docs/_plans/dual_inferencer_path_aware_followup_fix_plan.md."
        )
```

---

#### Test File 6 — MODIFICATION to existing `test_jinja_render_all_templates.py`

**Action**: Extend `_create_sentinel_context` to include `prior_output_path` and `main_response` in its known-variable set, OR rely on auto-discovery (the existing helper already extracts them automatically via regex — so no modification needed if regex captures them, but verify).

**Verification step**: Run the existing preflight after adding template changes. If `prior_output_path` is captured by `_get_variable_names_from_template`, no change needed. If not, add it explicitly.

---

### Test Coverage Summary

| Test file | New | Tests | What it proves |
|---|---|---|---|
| F1: `test_path_resolution.py` | ✅ | 12 | Helper resolves correct path for all 3 tiers + edge cases + active proposer switching |
| F2: `test_prompt_builder_feed_dict.py` | ✅ | 5–6 | Feed dict has correct keys + empty-string sentinel + no None leakage |
| F3: `test_followup_renders_path_aware.py` | ✅ | 4 | End-to-end Jinja render shows path + cp + populated tags |
| F4: `test_review_renders_path_aware.py` | ✅ | 2–3 | Same for review template |
| F5: `test_followup_path_aware_template.py` | ✅ | 4 (1 per template) | Preflight asserts variable references in templates |
| F6: `test_jinja_render_all_templates.py` | ⚙️ Verify | (existing) | Existing preflight still passes with new variables |

**Total**: 27–29 new test methods + 1 preflight verification.

**Run cost**: All under 1 second each (pure Jinja + filesystem mocks); preflight tests run in <5s total. No LLM calls anywhere.

---

## 4. Sequencing and Ship-It Order

To minimize risk and maximize incremental verifiability, ship in this order:

| # | Phase | Risk | Why this order |
|---|---|---|---|
| 1 | **Phase 1+2** (helper + plumbing) | Low — additive only | Highest user-facing impact. Fixes the empty `<ProposedDocument>` bug and adds `prior_output_path` data to the feed dict. Breaks no consumers. |
| 2 | **Phase 3** (update plan/main/followup.jinja2) | Low — backward-compat via if/else | Activates the LLM-facing improvement for the ONE template we know is bitten. |
| 3 | **Re-run shallow profile**, verify fixer's prompt now contains the path + populated `<ProposedDocument>` | — | Empirical validation of Phases 1–3. |
| 4 | **Phase 7** (unit + preflight tests) | Low | Lock in correctness. |
| 5 | **Phase 4** (sibling templates: review.jinja2, implementation/) | Low — same backward-compat pattern | Symmetric application. |
| 6 | **Phase 5** (loud warning) | Low — non-fatal | Surface remaining fallback consumers. |
| 7 | **Phase 6** (audit + migrate + remove) | Medium | Touches multiple consumers; deferred until warning data is available. |

**Phase 1+2+3+7 = the "MVP fix"**. Roughly 90 min. Solves the regeneration bug.

**Full Phase 1–7 = the "clean architecture"**. Roughly 1 day. Eliminates the silent fallback entirely.

---

## 5. Risk Assessment

### 5.1 Risks Mitigated

| Risk | Mitigation |
|---|---|
| Helper returns wrong file when multiple deliverables exist | Tier 1 prefers proposer's configured `_output_path` basename; falls back to first non-dotfile alphabetically; tested |
| Template renders raw `None` text when path unavailable | Plumbing uses `or ""`; template uses `{% if prior_output_path %}` |
| Backward-compat with templates that don't reference `prior_output_path` | Variable is just unused — Jinja silently ignores. Phase 2's `main_response` addition also fixes a pre-existing empty-tag bug for those templates |
| LLM ignores the new `cp` instruction and still regenerates | This is the next escalation: combine with stronger NOTES + (later) tool-only enforcement. Current fix is necessary even if not 100% sufficient |
| `_resolve_prior_proposer_output_path` slow (filesystem hits) | 2× `os.path.isfile` + 1× `os.listdir`. Negligible. Called once per fix iteration |
| Phase 6 removes a fallback some test depends on | Phase 5's warning data drives the audit; Phase 6 only proceeds when warning count is 0 |

### 5.2 Residual Risks

| Risk | Why we accept it (for now) |
|---|---|
| `<CurrentProposal>` (inner) and `<ProposedDocument>` (outer) both populated → LLM sees the prior content twice | Suboptimal but not actively harmful. Removing the inner-template duplication is gated by Phase 6 (which removes the inner default entirely). |
| Bug B (`reviewer_response` empty in followup) not fully addressed by this plan | Tracked as a separate follow-up — root cause is upstream in `_step_fix_impl`'s state setup, not in `_build_followup_prompt`'s wiring |
| Bug D (`base_output_str` captures summary text not file content) not fully addressed | Mitigated indirectly: with `prior_output_path` provided, the LLM no longer needs the file content inlined. The summary text is now positioned as a complement, not a substitute. A deeper fix (capturing file content into state when proposer is BTA/PTI) is a future enhancement |

### 5.3 What This Fix Does NOT Solve

- **Identity-preservation guards** in the template (don't renumber sections, don't retitle): valuable next addition, but separate from the path-aware fix.
- **Volume-preservation guards** (e.g. "if output >15% shorter than prior, stop and re-check"): same — separate template-side enhancement.
- **Self-audit JSON in `<Response>`** (preservation_audit fields): orthogonal observability improvement.

These can be added on top of this plan in subsequent iterations.

---

## 6. Acceptance Criteria

After Phase 1+2+3 are shipped and a fresh shallow run completes:

1. ✅ The fixer's rendered prompt contains the literal path of the base inferencer's deliverable (verifiable via `grep "/Users/.../base_inferencer/outputs/final_deliverables/output.md" <fixer_inference_input.txt>`).
2. ✅ The fixer's rendered prompt contains a `cp` instruction with that path as source and the destination as `output_path`.
3. ✅ The fixer's `<ProposedDocument>` tag is populated (non-empty) with the prior proposal text.
4. ✅ The fixer's resulting `output.md` is within ±5% of the base inferencer's `output.md` line count, indicating the LLM honored the copy-then-edit instruction.
5. ✅ The fixer's `output.md` preserves the base inferencer's section numbering scheme (e.g. `1.1` stays `1.1`, not renamed to `§1.1`).
6. ✅ The fixer's `output.md` preserves the base inferencer's title.
7. ✅ All existing DualInferencer tests pass.
8. ✅ The new unit + preflight tests (Phase 7) pass.

After Phase 5 is shipped:
9. ✅ Every `DualInferencer` instance using the in-Python fallback emits exactly one log warning per role per instance.

After Phase 6 is shipped:
10. ✅ The in-Python `DEFAULT_DUAL_*_PROMPT_TEMPLATE` constants raise `NotImplementedError` if accidentally invoked.
11. ✅ Every `DualInferencer` consumer (YAML, code, tests) explicitly configures `template_root_space` and `template_key`.
12. ✅ Re-running the integration suite produces zero `DeprecatedFallbackPrompt` warnings.

---

## 7. Open Questions for Decision Before Implementation

1. **Should the `cp` instruction in the template be hard-coded or use a tool-name placeholder?** Different LLM environments may have different copy primitives (some use `find_and_replace_in_file` with full-file replacement, others have explicit `cp` tools, others use `bash`). Recommendation: keep `cp` as the example but add a parenthetical "or your environment's equivalent file-copy operation".

2. **Should `_resolve_prior_proposer_output_path` walk up nested orchestrators?** E.g. if the proposer is a BTA whose own deliverable is itself produced by a sub-aggregator, do we want the BTA's outer deliverable or the sub-aggregator's? Recommendation: outer only (the BTA's own `final_deliverables/`) — this is the contract the BTA exposes; deeper traversal couples Dual to BTA internals.

3. **Should we also expose `prior_output_path_relative` (relative to `_workspace.root`)?** Useful for templates that want to print short paths. Recommendation: not in this fix; can be added later when a template needs it.

4. **Should Phase 6 happen at all, or is the loud warning enough?** Removing the fallback eliminates an entire class of silent-failure bugs at the cost of forcing every consumer to migrate. Recommendation: yes, eventually — the warning-only state is a half-measure that leaves the door open for new silent-fallback consumers.

5. **Do we need a feature flag** for Phase 1+2+3 in case re-runs reveal an unforeseen issue? Recommendation: no — the changes are small, additive, and easy to revert via git.

---

## 8. Provenance

This plan was derived from the rovodev session investigating `task_task-7a39a77a_20260508_081252` and the accompanying code base on 2026-05-09. Key evidence:

- **Rendered fixer prompt** at `<workspace>/children/fixer_inferencer/logs/session/RovoDevCliInferencer-58f906bd.jsonl.parts/InferenceInput/20260508_093329_f135be00.txt` — confirmed empty `<ProposedDocument>` and 2.7 KB summary in `<CurrentProposal>` instead of the 43 KB plan.
- **Base inferencer's deliverable** at `<workspace>/children/base_inferencer/outputs/final_deliverables/output.md` — 43,298 bytes, 773 lines, MD5 `ee5e4f0e60a00a159a3a9994f50131a7`.
- **Code references** in `dual_inferencer.py`: `_step_fix_impl` (line 970), `_build_followup_prompt` (line 1117), `_build_review_prompt` (line 1095), `_active_proposer` (line 440), `_resolve_*` placeholder for new helper.
- **Template references**: `plan/main/followup.jinja2`, `plan/main/review.jinja2`, `implementation/main/followup.jinja2`, `implementation/main/review.jinja2`.
- **Existing infrastructure leveraged**: `inferencer_workspace.has_deliverables` (line 102), `deliverable_path` (line 96), `deliverable_paths` (line 113), `output_path` (line 221), `outputs_dir` (line 55).

This plan does NOT propose changes to `inferencer_workspace.py`, `inferencer_base.py`, or any non-Dual orchestrator (BTA, PTI, MFDual). The fix is contained to `DualInferencer` and the per-domain Jinja templates.
