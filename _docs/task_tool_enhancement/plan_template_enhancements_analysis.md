# Plan Template Enhancements — Analysis & Recommendations

**Original analysis date:** Earlier session (May 2026)
**Verified & recorded:** 2026-05-16 08:35
**Source codebase verified against:** `CoreProjects/atlassian-packages/rankevolve/src/resources/prompt_templates/`

> **Caveat about code drift.** Some prompt template code has changed since the original test run that motivated this analysis. The *problems* (silent review degradation, weak decisiveness, vague risk discipline, breakdown vs. synthesis mismatch) all remain valid in the current codebase. Where partial implementation of a recommendation has already landed, this document calls that out explicitly with the `[PARTIALLY IMPLEMENTED]` marker; the rest of each recommendation still applies.

---

## 1. Context

The original analysis was triggered by an A/B/C plan-quality comparison where:
- **Plan A** (BreakdownThenAggregate workflow with 4 worker subtasks) produced a 1,026-line plan that was weak on risk analysis, had no severity ratings, had a wart of a "Conflicts & Tensions" section, and lacked acceptance criteria.
- **Plans B and C** (solo plans by the same model) produced shorter, more decisive, more risk-aware outputs (657 and 870 lines).

The investigation read:
- All three plan templates at `prompt_templates/plan/main/{initial,review,followup}.jinja2`
- `breakdown_result.json` showing the 4 worker subtask decomposition
- Plan A's review output (`review/outputs/output.md`)
- Round01 fix summary (8 issues, 8 accepted, 0 rejected)

---

## 2. Smoking-gun diagnosis (still valid 2026-05-16)

Three root causes, in priority order:

### Root cause 1 — Reviewer silently degraded when sandboxed in wrong workspace

The review document confirms:
> "I was unable to locate the requested files and classes within the accessible workspace. The paths you specified (`/Users/tchen7/MyProjects/rovoteam/OpenTeam/src/...`) are outside the current workspace directory… Recommendation: To investigate this properly, I would need access to the actual OpenTeam codebase…"

Three review subagents (Verify SOP Pipeline, Verify Tool/Skill Patterns, Verify State & Persistence) each exhausted iterations asking "would you like to copy the OpenTeam codebase into the workspace?" before producing only cosmetic feedback. The review template commands "YOU MUST also look into the actual codebase to verify the plan's assumptions" but provides no fallback when the codebase is unreachable, no instruction to flag "unverifiable" issues, and no requirement to surface the verification gap as an issue itself.

**Consequence:** the 8 issues that surfaced were mostly cosmetic typos and naming nits (e.g., `RichPythonUtils → PythonUtils`, a fabricated `task_queue: list[dict]` field, renumbering notes). None of the deep engineering risks that Plan B caught (the `_sop` JSON-serializability bug, the cache-invalidation problem, the SOP-key-prefix persistence problem) ever surfaced — the reviewer never even got to evaluate them.

**Verification 2026-05-16:** `review.jinja2:105` now mentions "Flag as MAJOR" for verification-related cases, but the trigger is "user-suggested approach dismissed without verification" — it does NOT catch "the reviewer itself couldn't access the workspace." The core silent-degradation problem persists.

### Root cause 2 — Breakdown subtasks frame work as "investigate and design," not "decide and commit"

All 4 worker subtasks were framed as comprehensive investigation/design exercises with todo items like:
- "Document all fields…"
- "Analyze… how X works"
- "Identify all limitations of…"
- "Design WorkflowDefinition… Fields: id, name, description…"

None of them demanded:
- Ranked trade-off comparisons (option A vs B vs C with explicit recommendation)
- Severity-rated risk registers
- Acceptance criteria for each phase
- Pasteable code snippets for the most novel parts
- An explicit "What we are NOT doing" non-goals section

Workers produced encyclopedic findings; the aggregator concatenated them; the final plan was broad-but-shallow — exactly what was observed (1,026 lines, weak risk analysis, no severity ratings, no acceptance criteria per phase).

**Verification 2026-05-16:** `task_breakdown/main/initial.jinja2` does NOT have any `deliverable_type` schema for emitted subtasks. The breakdown still treats every subtask as a fungible investigation unit.

### Root cause 3 — `initial.jinja2` rubric is too soft on decisiveness and risk

The original rubric (lines 12–18 of the version analyzed) listed 5 plan sections: High-Level Approach, Files to Create/Modify, Key Implementation Steps, Potential Risks and Mitigations (present but unweighted), Testing Strategy. This rubric:
- Has no severity scale for risks
- Has no acceptance criteria per phase
- Has no explicit "decisions made vs decisions open" separation
- Has no "Non-Goals" requirement
- Has no self-review / adversarial-challenge step

`followup.jinja2:39` further amplifies the problem by telling the fixer to "Prefer Incremental Edits" — efficient but structurally biases the system against deep restructuring even when the original plan has architectural weaknesses.

**Verification 2026-05-16:** `initial.jinja2` is now 97 lines, but grep for `risk register | acceptance criteria | non.?goals? | self.?review` returns **zero matches**. The rubric is still soft on all five dimensions.

---

## 3. Recommendations (with verification status as of 2026-05-16)

### 🥇 P0 — Highest-leverage, low-risk

#### Enhancement 1: Add "Verification Coverage" requirement to `review.jinja2`
**Status:** ⚠️ PARTIALLY IMPLEMENTED — `review.jinja2:105` has *some* MAJOR-flag language but does not cover the "reviewer cannot access the workspace at all" case. **Core problem persists.**

**Change** — add a new section between the existing `4.` and `5.` items:

```
5. **Verification Coverage** (REQUIRED):
   * At the start of your review, identify each codebase claim in the plan that
     needs verification (paths, function signatures, line numbers, behavior assertions).
   * For each, attempt verification. If a referenced path is outside your accessible
     workspace OR a tool returns "Paths must be within root" / "Invalid path", you
     MUST raise this as a **MAJOR `verification_gap` issue** in your JSON output —
     do NOT silently skip it.
   * Format the issue as:
     `{"severity": "MAJOR", "description": "Cannot verify <claim> — <reason>",
       "suggestion": "Plan author must inline the relevant code excerpt or
       relocate files into the workspace before next review round."}`
   * NEVER end a review with "I could not investigate"; always convert that into
     a structured issue.
```

And update the approval clause:
```
- You approve the plan only if there are only one or two trivial COSMETIC issues.
+ You approve the plan only if:
+   (a) there are no CRITICAL/MAJOR issues, including no `verification_gap` issues, AND
+   (b) there are at most 2 trivial COSMETIC issues.
+ If you raised any `verification_gap` issues, you MUST NOT approve.
```

**Why this is the single highest-leverage fix:** the entire collapse of Plan A's review quality came from silent degradation. Forcing the reviewer to surface the gap as a MAJOR issue means the fix step (or a re-run with broader workspace scope) can address it.

**Cost:** ~15 lines. **Risk:** near-zero.

#### Enhancement 2: Add "Decision Commitment" requirement to `initial.jinja2`
**Status:** ❌ NOT IMPLEMENTED (verified: grep for risk register / acceptance criteria / non-goals / self-review returns zero matches).

**Change** — replace the 5-item rubric with:

```
Create a comprehensive, implementable plan that covers:

1. **TL;DR / Mental Model** (≤1 page): A diagram or before/after table that lets
   a reviewer understand the architectural shift in 60 seconds.
2. **Goals & Non-Goals**: Bullet what is in scope, then explicitly bullet what is
   OUT of scope (so future scope creep can be referenced back). Minimum 2 non-goals.
3. **High-Level Approach**: Overall strategy and architecture decisions.
4. **Files to Create/Modify**: Three subsections — NEW / MODIFIED / DEPRECATED.
   Each file gets a one-line purpose.
5. **Decisions Made vs. Decisions Open**: Two clearly separated tables.
   * "Decisions Made" lists each design choice + one-sentence rationale.
   * "Decisions Open" lists questions still requiring user input + a tentative answer.
   Every open question MUST have a tentative answer; do NOT leave "TBD".
6. **Implementation Phases**: Dependency-ordered. Each phase MUST include:
   - Objective (1 line)
   - Files to touch
   - Acceptance criteria (concrete, testable, e.g. "pytest path/to/test.py is green",
     "render snapshot test produces N lines with M sections")
7. **Risk Register** (REQUIRED): A table of risks with columns:
   `# | Risk | Severity (🔴 High / 🟡 Med / 🟢 Low) | Likelihood | Mitigation`.
   You MUST identify at least 5 distinct technical risks. "Maybe X breaks" is
   not enough — name the specific failure mode (e.g. "Persisting `_sop` object
   into prior_context is not JSON-serializable; will crash session save").
8. **Testing Strategy**: Unit / Integration / E2E / Snapshot categories explicit.
9. **Critical-Thinking Self-Review** (REQUIRED): Before submitting, list 5–10
   plausible reviewer objections to your plan and respond to each in 1–3 lines.
   This is your final adversarial pass.
```

**Why:** this single replacement gives the proposer the same structural discipline that made Plan C win. The risk register requirement alone would have forced Plan A to surface the `_sop` JSON-serializability issue, the cache invalidation problem, and the SOP-key persistence problem that Plan B caught.

**Cost:** ~30 lines. **Risk:** low — proposer may produce a slightly more rigid document, but rigid+complete is exactly what an implementer wants.

---

### 🥈 P1 — High-value, moderate effort

#### Enhancement 3: Add "Decisiveness Audit" to `review.jinja2`
**Status:** ❌ NOT IMPLEMENTED.

**Change** — add a sub-bullet under item (3) Fair:

```
3. **Fair AND Decisive**:
   * Is there hacky implementation when there is more elegant approach?
   * Is it over-engineering when a simpler approach is equally effective?
   * **Does the plan commit to opinionated choices?** Flag as MAJOR if more
     than 30% of design choices are left "open" or use vague language like
     "we could either…", "TBD", or "options include…" without a recommended
     selection. An execution-oriented plan must commit; open questions
     belong in a separate dedicated section with tentative answers.
   * **Does the plan have an explicit Risk Register with severity ratings?**
     Flag as MAJOR if missing or if it has fewer than 5 specific technical risks.
   * **Does the plan have Non-Goals?** Flag as MINOR if missing.
   * **Does each implementation phase have testable Acceptance Criteria?**
     Flag as MAJOR if more than half the phases lack them.
```

**Why:** makes the reviewer enforce the rubric the proposer was given (Enhancement 2). Creates pressure on the proposer to comply.

**Cost:** ~12 lines. **Risk:** low.

#### Enhancement 4: Strengthen the breakdown subtask schema to demand "produce decisions, not findings"
**Status:** ❌ NOT IMPLEMENTED (verified: `task_breakdown/initial.jinja2` has no `deliverable_type` schema).

**Change** — wherever the breakdown prompt instructs the breakdown agent to emit subtasks JSON, add:

```
For EACH subtask in your breakdown, you MUST include:

  - "deliverable_type": one of:
      "investigation"  — produces findings only
      "design_decision"— produces a chosen architecture with named alternatives rejected
      "risk_analysis"  — produces a severity-rated risk register
      "acceptance_criteria"— produces concrete, testable done-conditions per phase

  - "must_commit_to": (only required if deliverable_type == "design_decision")
      An array of decision points this subtask must DECIDE (not just explore).
      Example: ["lifecycle states: active|suspended|completed|aborted",
                "tool count: 6 control tools",
                "manifest format: YAML or JSON"]

For any user request that involves an architectural change, you MUST include
AT LEAST ONE subtask each of type "design_decision", "risk_analysis", and
"acceptance_criteria". Pure "investigation" subtasks alone produce vague plans.
```

**Why:** the four subtasks in `breakdown_result.json` were all investigation/design hybrids. None was a dedicated risk-analysis or acceptance-criteria task. Adding these as required deliverable types forces the diamond workgraph to produce decision-making outputs, not just findings.

**Cost:** ~15 lines. **Risk:** moderate — may need to update schema-handler to recognize new fields, but at minimum the LLM produces them and the aggregator can use them.

#### Enhancement 5: Replace default concatenation with a synthesis-focused aggregator prompt
**Status:** ⚠️ PARTIALLY IMPLEMENTED — `breakdown_then_aggregate_inferencer.py` now offers `make_upstream_injecting_aggregator_prompt_builder()` (line 166) and a conflict-detection factory (line 197). But the **default** (when no custom builder is wired) is still raw concatenation: `f"### Result {idx+1}\n{res}"` at line 226. The synthesis-aggregator template I recommended below would be the natural backing for these factories.

**Change** — create file `plan/main/aggregator.jinja2`:

```
You are the synthesis aggregator. {{ num_workers }} workers each investigated
a slice of the user's request:

**Original request:**
<UserRequest>{{ original_query }}</UserRequest>

**Worker outputs:**
{% for r in worker_results %}
---
## Worker {{ loop.index }} — {{ r.subtask_description | truncate(120) }}
{{ r.output }}
{% endfor %}

---

# Your task: produce ONE cohesive plan, NOT a concatenation.

Hard rules:
1. **Synthesize, do not stitch.** If two workers disagree on a design choice,
   PICK ONE and justify it in ≤2 lines. Do NOT include both options as a
   "Conflicts & Tensions" section in the final plan.
2. **Compress duplicates.** If 3 workers all describe the same current-state
   fact, state it ONCE.
3. **Hold to the rubric** in `initial.jinja2` (sections 1–9 including
   Risk Register and Self-Review).
4. **The final plan is one author's voice.** No "Worker 1 said…" attributions
   in the body. You are the author; the workers are your sources.
5. **Page budget:** Aim for ≤ 800 lines total. If your draft exceeds this,
   compress — every additional 200 lines is a tax on the implementer.

Write the synthesized plan to `{{ output_path }}` and emit `<Response>` tags.
```

**Why:** this single template kills the largest source of Plan A's verbosity. Plan A was 1,026 lines vs. Plan B's 870 and Plan C's 657 for substantively similar coverage. The "Conflicts & Tensions" section in Plan A — which exists only because of poor synthesis — would never appear.

**Cost:** one new template (~30 lines) + wiring (~5 lines). **Risk:** low — worst case is fallback to current behavior.

---

### 🥉 P2 — Worthwhile, smaller impact

#### Enhancement 6: Tell `followup.jinja2` to push back when the plan has structural weaknesses
**Status:** ⚠️ PARTIALLY IMPLEMENTED — `followup.jinja2:30` says "Reject with specific evidence if the reviewer is mistaken" (right direction). But there's no guidance to add MISSING sections that the reviewer didn't request. Round01's "8 accepted, 0 rejected" outcome would still happen today.

**Change** — add after the "Prefer Incremental Edits" section:

```
## When Incremental Edits Are NOT Enough

If, after accepting reviewer feedback, you notice that the plan still lacks:
- a severity-rated **Risk Register** with ≥5 specific risks,
- **Acceptance Criteria** per implementation phase,
- explicit **Non-Goals**,
- a **Decisions Made vs. Decisions Open** separation, or
- a **Self-Review** section anticipating reviewer objections,

then you MUST add or rebuild those sections, even if the reviewer didn't request
it. Plan rubric compliance (per `initial.jinja2`) is a baseline obligation —
the reviewer's silence on a missing section is NOT permission to omit it.

Record this as an `auto_added_sections` entry in your JSON response:
  { "auto_added_sections": ["risk_register", "non_goals"], "reason": "..." }
```

**Why:** today the fix step is structurally conservative. This adds counter-pressure to rubric compliance.

**Cost:** ~12 lines. **Risk:** low.

#### Enhancement 7: Fix breakdown subtask `key_files` paths to be workspace-relative
**Status:** ❌ NOT FULLY IMPLEMENTED. `task_breakdown/initial.jinja2` has path guidance only for "avoid hanging finds on fbsource" (lines 117–119) — not the workspace-mismatch correctness issue.

**Change** — add to the breakdown prompt's note section:

```
# NOTES on `key_files` paths in subtask emission:
- The `key_files` paths in subtask args MUST be reachable from the worker's
  workspace root: `{{ workspace_root }}`.
- Before emitting a path, mentally verify it is INSIDE `{{ workspace_root }}`,
  NOT in a sibling repository or external clone.
- If a referenced file is OUTSIDE the workspace, you MUST either:
  (a) ask the orchestrator to mount/copy it (raise an
      `unreachable_dependency` flag in the breakdown output), OR
  (b) inline the relevant excerpt as a `code_excerpt` field on the subtask
      so the worker doesn't need filesystem access.
- NEVER emit paths starting with `/Users/.../rovoteam/` if the workspace is
  `/Users/.../CoreProjects/OpenStartup/` — these will silently fail in workers.
```

**Why:** the most direct fix for the underlying environmental failure. Combined with Enhancement 1, it would have prevented the entire degradation cascade.

**Cost:** ~10 lines. **Risk:** low.

---

### 🟡 P3 — Optional polish

#### Enhancement 8: Add a "Token Budget" sanity check to `initial.jinja2`
```
# Plan Length Discipline

Target ≤ 800 lines for the final plan. Every 200 additional lines is a tax on
the implementer's attention. Density beats length. If your draft exceeds 800
lines, compress: collapse three sentences to one, replace prose tables with
markdown tables, delete restated background.
```
**Cost:** 5 lines. **Risk:** trivial.

#### Enhancement 9: Bound the `<Response>` summary tag
In `initial.jinja2` and `followup.jinja2`, the `<Response>` tag wraps a "concise natural language summary" with unbounded length. Either keep `<Response>` but require it to be ≤ 5 lines, OR move it to a separate `summary.md`.
**Cost:** 3 lines. **Risk:** trivial.

---

## 4. Prioritized implementation order

| # | Enhancement | LOC | Risk | Status (2026-05-16) | Expected Impact |
|---|---|---|---|---|---|
| 1 | Verification Coverage gating | ~15 | very low | ⚠️ partial | 🟢🟢🟢 catches silent review degradation |
| 2 | Decision Commitment rubric in `initial.jinja2` | ~30 | low | ❌ missing | 🟢🟢🟢 forces risk register, AC, non-goals |
| 3 | Decisiveness Audit in `review.jinja2` | ~12 | low | ❌ missing | 🟢🟢 reinforces #2 from review side |
| 4 | `deliverable_type` in breakdown subtask schema | ~15 | moderate | ❌ missing | 🟢🟢 forces breakdown to allocate dedicated risk/decision/AC subtasks |
| 5 | Synthesis aggregator template | ~30 + wiring | low | ⚠️ partial (factories exist, template missing) | 🟢🟢🟢 eliminates concatenation; kills "Conflicts & Tensions" wart |
| 6 | Fix-step push-back guidance | ~12 | low | ⚠️ partial | 🟡 works in conjunction with #2/#3 |
| 7 | Workspace-relative `key_files` paths | ~10 | low | ❌ missing | 🟢 prevents workspace-mismatch failures |
| 8 | Token budget note | ~5 | trivial | ❌ missing | 🟡 cosmetic compression pressure |
| 9 | Trim `<Response>` summary | ~3 | trivial | ❌ missing | 🟡 cosmetic |

**Recommended sequencing:**
- **If you only do one thing:** Enhancement 1 (Verification Coverage gating in `review.jinja2`).
- **If you do three things:** add Enhancement 2 (Decision Commitment rubric) and Enhancement 5 (Synthesis aggregator template).
- **If you do five things:** add Enhancements 3 (Decisiveness Audit) and 4 (`deliverable_type` schema).
- **The remaining four (#6–#9) are polish.**

---

## 5. Explicitly NOT recommended (and why)

- ❌ **Don't restructure the `DualInferencer` phase machinery itself** — that's a code change, not a prompt change, and prompts are where the leverage is.
- ❌ **Don't force the breakdown into a fixed number of subtasks** — the diamond pattern is fine; the issue is what subtasks ask for.
- ❌ **Don't add a "be more opinionated" tone instruction without the structural rubric to back it up** — vague tone instructions don't move LLM outputs reliably. The rubric (#2) is the lever.
- ❌ **Don't demand the reviewer produce a numeric quality score** — would be theatrical; severity-rated issues + verification-gap flag are more actionable.
- ❌ **Don't add an extra propose-review-fix round by default** — adds latency without clear win; the better fix is to make round 1 more substantive.

---

## 6. One-line summary

The single highest-leverage change is **Enhancement 1**: make the reviewer required to flag unverifiable areas as a MAJOR `verification_gap` issue, which prevents silent review degradation. Add the **Risk Register + Acceptance Criteria + Self-Review** rubric to `initial.jinja2` (Enhancement 2) and a **synthesis aggregator template** (Enhancement 5) for the next two highest-leverage wins.

---

## 7. Verification appendix — what changed since the original analysis

| Original claim | Status 2026-05-16 | Notes |
|---|---|---|
| Plan templates at `prompt_templates/plan/main/{initial,review,followup}.jinja2` | ✅ Still there | Plus `_archive/260322/` subdirectory shows updates have happened |
| `breakdown_then_aggregate_inferencer.py` default aggregator concatenates with `### Result N` headers | ✅ Still true (line 226) | But `make_upstream_injecting_aggregator_prompt_builder()` (line 166) and conflict-detection factory (line 197) now exist — partial implementation of Enhancement 5 |
| `review.jinja2` has no verification-gap handling | ⚠️ Partial (line 105 has *some* MAJOR-flag language for user-suggested-approach dismissal) | The "reviewer cannot access workspace at all" case is NOT covered |
| `initial.jinja2` lacks Risk Register / AC / Non-Goals / Self-Review | ✅ Still true (97-line file; grep returns zero matches for any of these terms) |  |
| `task_breakdown/initial.jinja2` lacks `deliverable_type` schema | ✅ Still true | Has path guidance for avoiding hung `find` on fbsource (lines 117–119), but no schema work |
| `followup.jinja2` lacks "push-back when sections are missing" | ⚠️ Partial — `followup.jinja2:30` says "Reject with specific evidence" but no guidance to ADD missing sections | Round01's "8 accepted, 0 rejected" pattern would recur |

---

*End of analysis. The 9 enhancements remain implementable as-described; partial implementations noted above can be extended rather than redone.*
