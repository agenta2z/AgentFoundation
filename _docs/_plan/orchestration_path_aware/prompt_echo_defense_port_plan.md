# Prompt-Echo Defense — Integrated Fix Plan v3 (AgentFoundation ← RankEvolve)

**Status:** Proposed — INTEGRATED v3 (supersedes v1 and v2)
**Severity:** HIGH — silent artifact corruption + latent NoneType crashes in hot path
**Scope:** `DualInferencer`, `MultiFlowDualInferencer`, `flow_parsers.parse_finalplan_tag`, three downstream parsers, 12 live prompt templates (14 example blocks).
**Owners / Reviewers:** TBD (DualInferencer + prompt-templates owners)

> **Integration provenance.** This plan integrates the best parts of two prior drafts and reconciles them against verified ground truth from a fresh codebase inspection (session 2026-05-19, 21:13):
> - **Draft A (`_docs/_plan/prompt_echo_defense_port_plan.md` v2):** comprehensive phasing, test strategy, risk & rollback, DoD checklist, design principles.
> - **Draft B (`~/.claude/plans/take-a-look-into-hidden-cloud.md` updated):** sibling `MultiFlowDualInferencer` discovery, exact template line numbers, crisp out-of-scope statement, inline-reimpl (acknowledging delegation bug).
>
> The drafts have **converged** on the inline reimplementation; they now disagree mainly on **scope** (MultiFlow), **template count** (12 vs 13), and **organization**.
> See **Appendix A** for the side-by-side comparison, the bugs/inaccuracies found in each, and the rationale for every integration decision.

---

## 0. INDEX

1. Problem Statement
2. Verified Ground Truth (read before any edits)
3. Design Principles
4. Fix Plan
    - 4.1 Phase 1 — CRITICAL (live vulnerability)
    - 4.2 Phase 2 — IMPORTANT (downstream `None`-safety)
    - 4.3 Phase 3 — NICE-TO-HAVE (additional hardening)
5. Migration Order & PR Shape
6. Test Strategy
7. Risk & Rollback
8. Definition of Done
9. Cross-references
- Appendix A: Reconciliation of the prior drafts (including draft-by-draft defects and the "if you pick one" answer)

---

## 1. Problem Statement

Several AgentFoundation prompt templates contain a literal `<Response>...</Response>`
example block as part of their instructions. For instance,
`src/agent_foundation/resources/prompt_templates/implementation/main/initial.jinja2:37-41`:

```jinja2
**IMPORTANT: You MUST wrap your response in `<Response>` tags**: The system depends on
these tags to extract your summary. If you omit them, the entire raw output will be
used as the proposal text, and may cause downstream failures.

<Response>
[Your concise natural language summary of what you implemented, key results, and any concerns.]
</Response>
```

A naïve `<Response>...</Response>` regex will match the **prompt's own example
block** if the model fails to emit its own block. The model fails to emit its
own block whenever:

1. A CLI inferencer (Devmate / Metamate / Kiro / OpenClaw / RovoDev) echoes the
   rendered prompt back into its raw stream.
2. The model is truncated at `max_tokens` *after* invoking a tool but *before*
   producing its closing `</Response>` tag.
3. The model deliberately skips the block after a tool call.

The symptom in RankEvolve was a 607-byte round file containing the literal
prompt template instead of the model's answer. RankEvolve catalogued this as
`forensics_round_template_echo_bug` and deployed a 3-layer defense.

**Root cause in AgentFoundation today:** the leaf parser `extract_delimited`
has the defense (markers, reverse-iteration, `None`-on-all-echo), but
`DualInferencer._default_response_parser` does **not delegate to it** and the
three step callsites do **not handle the `None` failure signal** even if the
parser produced one. The defense was **half-ported**.

A **sibling vulnerability** lives in `MultiFlowDualInferencer`, whose default
`multi_flow_response_parser = parse_finalplan_tag` (`flow_parsers.py:35,
111-117`) uses a naïve `re.compile(r"<FinalPlan>...</FinalPlan>")`. Same
failure mode, different tag.

---

## 2. Verified Ground Truth (read this before any edits)

All facts below were verified by direct file inspection at 2026-05-19 21:13.
Re-verify before each PR — line numbers shift.

### 2.1 Existing infrastructure (do not re-implement)

| Asset | Location | Notes |
|---|---|---|
| `extract_delimited(raw, open_tag="<Response>", close_tag="</Response>") -> Optional[str]` | `src/agent_foundation/common/response_parsers/delimiter_parser.py:16-53` | Already has `_PROMPT_ECHO_MARKERS = ("**IMPORTANT", "wrap your response")` at module scope (line 13). Reverse-iterates matches. Returns `None` only when ALL matches are echoes; returns `raw` unchanged when there are **zero** matches. |
| `InferencerExecutionError(tool, return_code, stderr, error)` | `src/agent_foundation/common/inferencers/agentic_inferencers/common.py:342-357` | `RuntimeError` subclass. Already in active use by Devmate retry plumbing (`devmate/common.py:90`). |
| `extract_response_text(result)` | `agentic_inferencers/common.py:377` | Type-dispatch wrapper for dict-returning leaves. Used by Phase 3 (F7). |
| `DualInferencer.phase: str = attrib(default="")` | `dual_inferencer.py:244` | Phase attribute exists; default is `""` — always wrap as `self.phase or "DualInferencer"`. |
| `DualInferencer._default_response_parser` (CURRENT, BROKEN) | `dual_inferencer.py:1795-1802` | Iterates `("Response", "ImprovedProposal")`, returns first `re.search` match unconditionally; falls back to raw. **No echo detection, no reverse iteration, no `None`.** |
| `DualInferencer._default_extract_proposal` | `dual_inferencer.py:1854-1861` | Separate `<ImprovedProposal>` extractor (NOT `_default_response_parser`). Keep behavior identical. |
| `parse_finalplan_tag` | `flow_parsers.py:111-117` | Module-level `_FINALPLAN_RE = re.compile(r"<FinalPlan>([\s\S]*?)</FinalPlan>", re.IGNORECASE)` at line 35; returns `s` (raw) on no-match. **No echo defense.** |
| `MultiFlowDualInferencer.multi_flow_response_parser` | `multi_flow_dual_inferencer.py:123, 161, 340` | Defaults to `parse_finalplan_tag`; passed as `response_parser=` to a `DualInferencer` at line 340. The `MultiFlow` layer is therefore vulnerable in the same way DualInferencer is. |

### 2.2 Vulnerable callsites in `dual_inferencer.py`

| Phase | Line | Current code |
|---|---|---|
| propose | 1070 | `base_output_str = self.response_parser(_raw_base)` |
| review | 1248 | `review_output_str = self.response_parser(_raw_review)` |
| fix | 1420 | `fix_output_str = self.response_parser(_raw_fix)` |

Downstream code uses these values immediately; none guard against `None`.

### 2.3 Downstream `None`-crash sites (confirmed)

| File | Line | Issue |
|---|---|---|
| `inferencer_base.py` | 962 | `cleaned = extract_delimited(str(response))` then `f.write(cleaned)` — crashes `TypeError` on `None`. |
| `plan_then_implement_inferencer.py` | 969-991 | `try: cleaned = extract_delimited(...); except ...` — `None` return is not an exception, falls through to `re.search(..., cleaned)` → `TypeError`. |
| `breakdown_then_aggregate_inferencer.py` | 740-748 | `response_text = extract_delimited(str(raw_output))`; no `None` guard. |

### 2.4 Live prompt templates with literal `<Response>...</Response>` example blocks

**Verified count: 12 unique files, 14 example blocks.** (Draft B's table said
"13 files" but counted `deep_research/main/initial.jinja2` twice — that file
has two example blocks but is one file.)

```bash
# Authoritative enumeration (run before F3):
grep -rln '<Response>' src/agent_foundation/resources/prompt_templates/ \
  | grep -v '/_archive/' | grep '\.jinja2$' | sort
# → 12 files (confirmed at 2026-05-19 21:13)
```

| # | File (under `src/agent_foundation/resources/prompt_templates/`) | Example block line(s) | Notes |
|---|---|---|---|
| 1 | `implementation/main/initial.jinja2` | 39-41 | |
| 2 | `implementation/main/followup.jinja2` | 71-87 | |
| 3 | `implementation/main/review.jinja2` | 89-108 | |
| 4 | `plan/main/initial.jinja2` | 44-49 | |
| 5 | `plan/main/followup.jinja2` | 69-85 | |
| 6 | `plan/main/review.jinja2` | 80-99 | |
| 7 | `analysis/main/initial.jinja2` | 62-71 | |
| 8 | `analysis/main/followup.jinja2` | 67-82 | |
| 9 | `analysis/main/review.jinja2` | 68-87 | |
| 10 | `deep_research/main/initial.jinja2` | 38-40 **AND** 48-50 | **TWO example blocks in one file** |
| 11 | `task_breakdown/main/initial.jinja2` | 81-113 | Large (32-line JSON example) |
| 12 | `conversation/main/initial.jinja2` | 93-101 | **Special case — see §2.5.** |

`_archive/**` files are out of scope (not loaded at runtime).

### 2.5 Special case: `conversation/main/initial.jinja2`

This template uses `<Response>` heavily as the **runtime delimiter for the
conversation agent's output**, not just as an instructional example. Lines
71, 72, 77, 78, 80, 105, 106 reference the tag in prose. Only lines **93-101**
form the example block.

**Implication for F3:** rename ONLY the lines 93-101 example block to
`<ResponseSchema>`. Leave all prose references intact (they are load-bearing
echo markers AND runtime contract). Reviewer must double-check this file by
diffing lines outside 93-101 are unchanged.

### 2.6 Out of scope but worth noting

- `task_breakdown/main/initial.jinja2:124` contains a prose reference
  `"YOU MUST output <Response> ... </Response> delimited output"`. This is
  prose, not an example block — keep as-is.
- `unified_proposal/main/*.jinja2`, `individual_proposal/main/*.jinja2`,
  `recovery/*.jinja2` contain NO literal `<Response>` tags. Verified clean.
- `_archive/**` paths are unreachable at runtime; do not edit.

---

## 3. Design Principles (govern every fix below)

1. **Single source of truth for the echo defense.** `_PROMPT_ECHO_MARKERS` lives in **one** place: `delimiter_parser.py`. Code at the consumer (DualInferencer) does NOT re-declare them. (Draft A's instinct to duplicate was rejected to avoid drift.)
2. **Inline reimplementation over delegation at consumer call.** The consumer (`_default_response_parser`) reimplements the loop inline rather than calling `extract_delimited` twice (once per tag). Delegation has a passthrough-vs-None ambiguity that silently kills the `<ImprovedProposal>` fallback path (Draft B v1's bug; resolved in Draft B v2 and here).
3. **Hard-fail loudly when the parser returns `None`.** Never let `None` propagate into write/match sites — convert to a structured `InferencerExecutionError` at the boundary so retry plumbing can react.
4. **Preserve existing semantics for non-echo inputs.** A raw output with no tags must still return the raw string (passthrough); a raw output with a clean `<ImprovedProposal>` block must still return its content. We are tightening the failure mode, not changing the success mode.
5. **Prompt templates: rename the *example block only*.** Keep every prose mention of "wrap your response in `<Response>` tags" intact — those are the *load-bearing* echo markers the detector keys on. Removing them would silently weaken the defense.
6. **Defense in depth, not duplication.** Multiple guards (parser, callsite, downstream sink) each handle `None` at their own level so a missed import or a non-default parser still fails safely.
7. **Symmetry across the inferencer family.** Apply the same defense pattern to `MultiFlowDualInferencer` / `parse_finalplan_tag` (sibling vulnerability) in the same PR family — leaving a sibling unfixed is technical debt that will bite later.
8. **Forensics breadcrumbs.** Preserve the `forensics_round_template_echo_bug` reference comment when porting RankEvolve-derived code. Doc-only, but invaluable for future debugging.

---

## 4. Fix Plan

### 4.1 Phase 1 — CRITICAL (closes the live vulnerability)

#### F0 — Pre-flight enumeration (do this first; no code change)

```bash
# 4.1.0.a — Confirm the 12-file / 14-block template list:
grep -rln '<Response>' src/agent_foundation/resources/prompt_templates/ \
  | grep -v '/_archive/' | grep '\.jinja2$' | sort

# 4.1.0.b — For each file, print every <Response> / </Response> /
# ResponseSchema line so reviewers can distinguish example blocks from prose:
for f in $(grep -rln '<Response>' src/agent_foundation/resources/prompt_templates/ \
             | grep -v '/_archive/' | grep '\.jinja2$' | sort); do
  echo "=== $f ==="
  grep -n '<Response>\|</Response>\|ResponseSchema' "$f"
done
```

**Acceptance:** Captured output matches the §2.4 table exactly OR documents
deltas (with reasoning) at the top of the F3 PR description.

---

#### F1 — Harden `DualInferencer._default_response_parser`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`
**Action:** Replace lines 1795-1802. Return type changes from `str` to `Optional[str]`.

```python
@staticmethod
def _default_response_parser(raw: str) -> Optional[str]:
    """Extract response content from delimiter tags, with prompt-echo defense.

    Inline reimplementation (not delegation to ``extract_delimited``) so the
    two-tag fallthrough behavior is unambiguous — see Design Principle #2
    in _docs/_plan/prompt_echo_defense_port_plan.md for why delegation is
    unsafe here.

    Three return modes:
      1. No matches of either tag       → returns raw unchanged (passthrough).
      2. ≥1 clean match (any tag)       → returns the most-recent clean match.
      3. Matches exist but ALL are echo → returns None (hard-failure signal).

    Originating incident: forensics_round_template_echo_bug (RankEvolve, 607-byte
    round-file corruption). Keep this reference for future debugging.
    """
    from agent_foundation.common.response_parsers.delimiter_parser import (
        _PROMPT_ECHO_MARKERS,
    )

    any_matches = False
    for tag in ("Response", "ImprovedProposal"):
        matches = list(re.finditer(rf"<{tag}>([\s\S]*?)</{tag}>", raw))
        if matches:
            any_matches = True
        for match in reversed(matches):
            content = match.group(1).strip()
            if all(marker in content for marker in _PROMPT_ECHO_MARKERS):
                continue
            return content
    if not any_matches:
        return raw
    return None
```

**Imports to verify/add** at top of file:
- `Optional` in the `typing` import (verify present; add if not).
- `re` already present at module level (verified).
- `_PROMPT_ECHO_MARKERS` import is intentionally **function-local** to avoid a circular import at module load (delimiter_parser is a leaf module).

**Why inline, not delegation:** delegating to `extract_delimited(raw, tag)`
twice has a passthrough-vs-None ambiguity — on a raw with only
`<ImprovedProposal>`, the first call (`<Response>`) returns `raw`
(passthrough sentinel), which `result is not None` treats as a successful
hit, silently killing the second-tag fallthrough. The inline form tracks
`any_matches` across BOTH tags before deciding passthrough vs. None.

**Acceptance:**
- `parser("<Response>**IMPORTANT...wrap your response...</Response>")` → `None`.
- `parser("<Response>echo</Response><Response>clean</Response>")` (where "echo" contains both markers, "clean" doesn't) → `"clean"`.
- `parser("no tags at all")` → `"no tags at all"` (passthrough).
- `parser("<ImprovedProposal>x</ImprovedProposal>")` → `"x"`.
- `parser("<Response>**IMPORTANT...wrap your response...</Response><ImprovedProposal>y</ImprovedProposal>")` → `"y"` (Response is echo-only, fall through to IP).
- A string containing only "important" or only "wrap your response" (one marker, not both) is NOT rejected (negative-space test).

---

#### F2 — Hard-fail on `None` at all three step callsites

**File:** same `dual_inferencer.py`

**F2.a — Add import** (extend the existing import block at lines 22-29):
```python
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    # ... existing imports ...,
    InferencerExecutionError,   # ADD
)
```

**F2.b — Insert after line 1070 (propose phase):**
```python
base_output_str = self.response_parser(_raw_base)
if base_output_str is None:                                                       # NEW
    raise InferencerExecutionError(                                                # NEW
        tool=f"<{self.phase or 'DualInferencer'}.propose>",                        # NEW
        error=(                                                                    # NEW
            "response_parser returned None — model emitted no clean "              # NEW
            "<Response> block (all matches were prompt-template echoes). "         # NEW
            "Common causes: max_tokens truncation, post-tool-call drop, or "       # NEW
            "CLI leaf echoing the rendered prompt. "                               # NEW
            f"raw_size_bytes={len(_raw_base)}"                                     # NEW
        ),                                                                         # NEW
    )                                                                              # NEW
```

**F2.c — Insert after line 1248 (review phase):** same pattern, with
`tool=f"<{self.phase or 'DualInferencer'}.review>"` and
`raw_size_bytes=len(_raw_review)`.

**F2.d — Insert after line 1420 (fix phase):** same pattern, with
`tool=f"<{self.phase or 'DualInferencer'}.fix>"` and
`raw_size_bytes=len(_raw_fix)`.

**Acceptance:**
- Mock `base_inferencer` returns echo-only string → propose phase raises `InferencerExecutionError`; `e.tool == "<DualInferencer.propose>"` (or configured phase label); `"raw_size_bytes=" in str(e)`.
- Same for review and fix.
- A run that previously crashed `TypeError` deep in `_maybe_replace_with_file_reference` now raises the structured error at the parse boundary.
- Existing retry plumbing (Devmate `common.py:90`, Resumable callbacks) catches the error and triggers retry per existing semantics — **verify with an integration test using a mock Devmate leaf**.

---

#### F2-Sibling — Harden `parse_finalplan_tag` (MultiFlowDualInferencer)

**File:** `src/agent_foundation/common/inferencers/flow_parsers.py`
**Action:** Replace the body of `parse_finalplan_tag` (line 111-117) with a
version that has the same echo defense.

```python
# Replace the module-level _FINALPLAN_RE (line 35) and the function body:

def parse_finalplan_tag(s: Any) -> Optional[str]:
    """Extract content from ``<FinalPlan>...</FinalPlan>`` with prompt-echo defense.

    Mirrors the semantics of ``extract_delimited`` and
    ``DualInferencer._default_response_parser``:

      1. No matches            → returns ``str(s)`` (passthrough — preserves
         backward-compat with the legacy "return raw on no-match" contract).
      2. ≥1 clean match        → returns the most-recent clean match's content.
      3. Matches exist but ALL → returns ``None`` (hard-failure signal).
         are prompt-echoes

    Originating incident: forensics_round_template_echo_bug (RankEvolve).
    """
    from agent_foundation.common.response_parsers.delimiter_parser import (
        _PROMPT_ECHO_MARKERS,
    )

    text = str(s) if not isinstance(s, str) else s
    matches = list(_FINALPLAN_RE.finditer(text))
    if not matches:
        return text
    for match in reversed(matches):
        content = match.group(1).strip()
        if all(marker in content for marker in _PROMPT_ECHO_MARKERS):
            continue
        return content
    return None
```

**F2-Sibling.b — Guard the MultiFlow callsite.** `MultiFlowDualInferencer`
passes `parse_finalplan_tag` into a child `DualInferencer` at line 340
(`response_parser=self.multi_flow_response_parser`). Once F1 + F2 land, the
child's propose/review/fix phases will raise `InferencerExecutionError` on
`None` automatically. **No code change at the MultiFlow callsite** — the
defense composes through the existing `response_parser` injection.

**F2-Sibling.c — Audit `<FinalPlan>` template usage.** Currently there are
no `.jinja2` files containing literal `<FinalPlan>...</FinalPlan>` example
blocks (verified by `grep -rln '<FinalPlan>' src/agent_foundation/resources/`).
**Action:** Add a CI/lint check that fails if a future template introduces a
`<FinalPlan>` example block. (Optional; if not added, add a comment in
`flow_parsers.py:35` noting "if `<FinalPlan>` ever appears in a template,
update the F3 enumeration and template rename plan accordingly".)

**Acceptance:**
- New unit tests in `test/.../test_flow_parsers.py`: echo-only → `None`; mixed → last clean; no tags → passthrough.
- End-to-end: a `MultiFlowDualInferencer` with a mock base inferencer that
  returns an echo-only string raises `InferencerExecutionError` at the
  propose phase (verifies F1 + F2 + F2-Sibling.a compose correctly).

---

#### F3 — Migrate prompt-template *example blocks* to `<ResponseSchema>`

For each of the 12 files in §2.4, change the example block (and ONLY the
example block) per this pattern:

**Before:**
```jinja2
<Response>
[Your concise natural language summary ...]
</Response>
```

**After:**
```jinja2
<ResponseSchema>
[Your concise natural language summary ...]
</ResponseSchema>

(Use literal `<Response>` and `</Response>` in your actual reply — `<ResponseSchema>` is just the example container.)
```

**Strict rules:**
- Change **example blocks only**.
- Do **NOT** touch inline prose: `**IMPORTANT: You MUST wrap your response in `<Response>` tags**` stays exactly as-is. (These are load-bearing echo markers.)
- Do **NOT** touch `_archive/**` files.
- **Special handling for `conversation/main/initial.jinja2`:** rename **only the lines 93-101 block**. Verify all other `<Response>` references in this file (lines 71, 72, 77, 78, 80, 105, 106) are unchanged — they are runtime contract about the conversation agent's output delimiter.
- **Special handling for `deep_research/main/initial.jinja2`:** rename **both** example blocks (lines 38-40 AND 48-50).
- **Special handling for `task_breakdown/main/initial.jinja2`:** the example block at lines 81-113 is large (32 lines of JSON). Rename the wrapper only; keep the JSON body intact. The prose reference at line 124 stays as-is.

**Acceptance:**
```bash
# After F3, this should return ZERO matches in non-archive templates:
grep -rn '^<Response>$' src/agent_foundation/resources/prompt_templates/ \
    | grep -v '/_archive/' | grep '\.jinja2$'
# (Matches a literal <Response> on its own line — i.e. an example-block opener.)

# And this should return exactly 14 matches across 12 files:
grep -rn '^<ResponseSchema>$' src/agent_foundation/resources/prompt_templates/ \
    | grep -v '/_archive/' | grep '\.jinja2$' | wc -l
# → 14

grep -rl '^<ResponseSchema>$' src/agent_foundation/resources/prompt_templates/ \
    | grep -v '/_archive/' | grep '\.jinja2$' | sort -u | wc -l
# → 12 (unique files)
```
- Snapshot tests (if any) regenerated and reviewed.

---

### 4.2 Phase 2 — IMPORTANT (downstream `None`-safety; parallelizable)

#### F4 — Guard `_finalize_output` (`inferencer_base.py:954-966`)

**Insert after the existing `cleaned = extract_delimited(str(response))` line (962):**
```python
if cleaned is None:
    logger.warning(
        "extract_delimited returned None for output_path=%s; writing raw "
        "response as fallback. All <Response> matches were prompt-template "
        "echoes — see _docs/_plan/prompt_echo_defense_port_plan.md for context.",
        resolved,
    )
    cleaned = str(response)
```

**Why a fallback (not a raise) here?** `_finalize_output` runs at the **end**
of a successful inference; raising here corrupts the successful return path
and confuses retry logic. We log loudly, write raw text (so the artifact is
investigable in forensics), and let the upstream parsers (which DID raise via
F2) own the hard-failure signal.

**Acceptance:** Input whose `extract_delimited(...)` returns `None` writes
the raw string and emits a `WARNING` log referencing this plan doc; no
`TypeError`.

---

#### F5 — Guard `_parse_analysis_response` (`plan_then_implement_inferencer.py:962-997`)

**After the existing `try / except Exception: cleaned = analysis_text` block:**
```python
try:
    cleaned = extract_delimited(analysis_text)
except Exception:
    cleaned = analysis_text
if cleaned is None:                              # NEW
    cleaned = analysis_text                      # NEW
```

**Acceptance:** Echo-only `analysis_text` returns `(False, "")` (the
existing "could not parse" fallback) without raising `TypeError` from a
downstream `re.search`.

---

#### F6 — Guard `_parse_json_subtasks` (`breakdown_then_aggregate_inferencer.py:735-755`)

**After `response_text = extract_delimited(str(raw_output))`:**
```python
if response_text is None:                        # NEW
    response_text = str(raw_output)              # NEW
```

**Acceptance:** Echo-only input no longer crashes the subtask parser;
function reverts to attempting JSON extraction from raw (which may fail
downstream with a meaningful parse error — acceptable).

---

### 4.3 Phase 3 — NICE-TO-HAVE (additional hardening; separate PRs)

#### F7 — Use `extract_response_text` for dict-returning leaves

**File:** `dual_inferencer.py`, callsites at lines ~1054, ~1226 (modern branch), ~1234 (legacy branch), ~1395 (modern), ~1403 (legacy).
**Change:** replace `str(await leaf.ainfer(...))` with `extract_response_text(await leaf.ainfer(...))` (importing `extract_response_text` from `agentic_inferencers.common`).
**Why:** Dict-returning leaves (DevmateCli, ClaudeCodeCli) currently stringify as `dict.__repr__`, escaping real newlines to literal `\n` and corrupting downstream artifact writes via `_maybe_replace_with_file_reference`.
**Why separate PR:** Touches the success path; needs an integration test with a mock dict-returning leaf to confirm artifact integrity end-to-end.

#### F8 — F0 no-op convergence detection in `_step_fix_impl`

**File:** `dual_inferencer.py:_step_fix_impl`.
**Behavior:** When `parsed_counter["items"] == []` AND `parsed_counter["summary"].startswith("no-op:")`, short-circuit the consensus loop (set `consensus_reached=True`, preserve prior `base_output_str`) instead of overwriting with the fixer's degenerate reply.
**Source:** RankEvolve `dual_inferencer.py:651-684`.
**Why separate PR:** Behavioral change to consensus invariants; needs domain review.

---

## 5. Migration Order & PR Shape

```
Phase 1 (CRITICAL — close the live vulnerability)
   F0 (enumeration) ────────────► F3 (template renames)
        │
        ▼
   F1 (harden parser) ──┐
                        ├──► F2 (raise on None at 3 callsites)
                        │
                        ▼
                  F2-Sibling (parse_finalplan_tag)

Phase 2 (IMPORTANT — downstream None-safety; parallelizable)
   F4  ║
   F5  ║─── independent; any order
   F6  ║

Phase 3 (NICE-TO-HAVE — separate PRs)
   F7  ──► dict-stringification fix
   F8  ──► F0 no-op convergence
```

**Recommended PR sequencing:**
- **PR-1 (Phase 1 core):** F1 + F2 + F2-Sibling + unit tests (§6.1) + regression fixture (§6.2). Smallest blast radius, biggest defense impact.
- **PR-2 (Phase 1 templates):** F0-enumerated F3 renames + any updated snapshots. Pure data change; trivial rollback.
- **PR-3 (Phase 2):** F4 + F5 + F6 with their unit tests.
- **PR-4 (Phase 3 — optional):** F7 alone.
- **PR-5 (Phase 3 — optional):** F8 alone.

---

## 6. Test Strategy

### 6.1 New unit tests (under `test/agent_foundation/common/inferencers/test_dual_inferencer/test_response_parser.py`)

| Test | Verifies |
|---|---|
| `test_default_parser_returns_clean_response` | Single clean `<Response>` block → returns content. |
| `test_default_parser_skips_echo_returns_none` | Only an echo `<Response>` (both markers) → `None`. |
| `test_default_parser_picks_last_clean_when_mixed` | Echo first + clean second → returns the clean one (reverse-iter). |
| `test_default_parser_passthrough_no_tags` | Raw with no tags → returns raw unchanged. |
| `test_default_parser_falls_through_to_improved_proposal` | Only `<ImprovedProposal>` → returns its content. |
| `test_default_parser_response_echo_then_clean_improved_proposal` | Echo-only `<Response>` + clean `<ImprovedProposal>` → returns IP content (regression for delegation-asymmetry bug). |
| `test_default_parser_no_false_positive_on_single_marker` | Content with only "important" or only "wrap your response" (one marker, not both) → NOT rejected. |
| `test_propose_raises_inferencer_execution_error_on_echo` | Mock leaf returns echo-only; assert `InferencerExecutionError`, `e.tool == "<DualInferencer.propose>"`, `"raw_size_bytes=" in str(e)`. |
| `test_review_raises_on_echo` | Same for review. |
| `test_fix_raises_on_echo` | Same for fix. |
| `test_finalize_output_handles_none` | `_finalize_output` with echo-only input writes raw fallback + WARNING. |
| `test_analysis_parser_handles_none` | `_parse_analysis_response` with echo-only input returns `(False, "")` without raising. |
| `test_breakdown_subtasks_handles_none` | `_parse_json_subtasks` with echo-only input handles gracefully. |
| `test_parse_finalplan_tag_skips_echo` | (`test_flow_parsers.py`) Only echo `<FinalPlan>` → `None`. |
| `test_parse_finalplan_tag_passthrough` | No tags → passthrough. |
| `test_parse_finalplan_tag_mixed` | Echo + clean → returns clean. |
| `test_multiflow_dual_propagates_echo_failure` | Mock base in `MultiFlowDualInferencer` returns echo-only; assert `InferencerExecutionError` is raised (composition test). |

### 6.2 Regression / end-to-end fixture

**Fixture:** capture a "Devmate echo" raw output — the literal rendered
`implementation/main/initial.jinja2`. Save under
`test/agent_foundation/common/inferencers/test_dual_inferencer/fixtures/devmate_echo_output.txt`.

**Test:** instantiate a `DualInferencer` using only `_default_response_parser`
(no `extract_delimited` override; mirrors real-world callers who haven't
wired the explicit override). Feed the fixture via a mock leaf. Assert
`InferencerExecutionError` is raised at the propose phase with diagnostic
context.

**Pre-fix behavior:** round file contains prompt text (silent corruption).
**Post-fix behavior:** structured error at the parse boundary.

### 6.3 Existing tests to update / audit

- `test_dual_inferencer/__main__.py:496` and `test_plan_then_implement.py:727,743` — these explicitly pass `response_parser=extract_delimited`. After F1, the default parser converges on the same semantics. **Audit; do NOT blindly remove the overrides** — they remain valid defense-in-depth.
- `test_multi_flow_dual_inferencer.py` — verify no test asserts that `parse_finalplan_tag` returns `str` (now `Optional[str]`).
- Any prompt-snapshot tests under `test/**/templates/` — regenerate after F3.

### 6.4 Negative-space tests (must not regress)

- Legit output containing only `"important"` or only `"wrap your response"` (one marker) is NOT treated as echo.
- Legit output with multiple valid `<Response>` blocks returns the **last** one (most-recent-wins).
- A real model output that quotes the prompt's "**IMPORTANT** ..." marker only inline (e.g. in a code comment) but does NOT contain "wrap your response" anywhere → NOT rejected.

---

## 7. Risk & Rollback

### 7.1 Risk

| Risk | Likelihood | Mitigation |
|---|---|---|
| F1 false-positive: legitimate output rejected as echo | Low — requires BOTH markers verbatim (`**IMPORTANT` AND `wrap your response`) | Negative-space tests (§6.4). Markers chosen because they are unique to instruction prose. |
| F1 delegation-asymmetry bug (Draft B v1 had this) | Eliminated by design | Inline reimplementation tracks `any_matches` across both tags; explicit regression test (`test_default_parser_response_echo_then_clean_improved_proposal`). |
| F2 raises break a caller that swallows `InferencerExecutionError` upstream | Low — `RuntimeError` subclass already in active use by Devmate retry plumbing (`devmate/common.py:90`) | Existing catchers continue to work; F2 just adds another raise site. |
| F2-Sibling `parse_finalplan_tag` now returns `Optional[str]` instead of `str` — type contract change | Low | Audit `test_multi_flow_dual_inferencer.py` and any other call sites; verify no `.upper()` / `.strip()` on the result without a `None` check. |
| F3 template renames break prompt-fingerprint caches | Medium for downstream consumers | Bump prompt version OR regenerate snapshots; list affected teams in F3 PR description. |
| F3 misses a special-case template (conversation, deep_research, task_breakdown) | Medium without diligence | Special handling spelled out per file in §4.1 F3; reviewer must diff the **non-example-block** lines to confirm they are unchanged. |
| F4 silent fallback hides a real bug | Low — WARNING log emitted; raw artifact preserved | Log message references this plan doc for forensics trail. |
| F7 changes dict stringification semantics | Medium for downstream consumers of `_raw_base`/`_raw_review`/`_raw_fix` strings | Scope strictly to DualInferencer; add integration test with mock dict leaf; deliver in its own PR. |

### 7.2 Rollback

- **F1 + F2 + F2-Sibling must be reverted together** (F2 depends on F1's `None` return semantics; F2-Sibling depends on F2's hard-fail composition).
- **F3 is pure data** — `git revert` of the prompt-templates directory restores prior behavior. No code coupling.
- **F4, F5, F6** are independent single-line guards; revert each individually.
- **F7, F8** are in their own PRs by design.

---

## 8. Definition of Done

### Phase 1 (CRITICAL)
- [ ] **F0**: Enumeration script run; file list documented in PR-2 description; matches §2.4 exactly OR deltas justified.
- [ ] **F1**: `DualInferencer._default_response_parser` is inline-reimplemented, has `Optional[str]` return type, preserves tag-fallthrough semantics, and passes all §6.1 parser unit tests.
- [ ] **F2**: All three step callsites (propose/review/fix) raise `InferencerExecutionError` with `e.tool` and `raw_size_bytes=...` diagnostic when parser returns `None`; verified by §6.1 callsite tests.
- [ ] **F2-Sibling**: `parse_finalplan_tag` has echo defense and `Optional[str]` return; MultiFlow composition verified by `test_multiflow_dual_propagates_echo_failure`.
- [ ] **F3**: All 12 (F0-confirmed) live templates have 14 example blocks renamed to `<ResponseSchema>` + literal-tag note; the bash checks in §4.1 F3 acceptance section all pass; snapshot tests regenerated.
- [ ] Regression fixture in §6.2 passes post-fix and fails pre-fix.

### Phase 2 (IMPORTANT)
- [ ] **F4**: `inferencer_base._finalize_output` `None`-safe; emits WARNING.
- [ ] **F5**: `_parse_analysis_response` `None`-safe.
- [ ] **F6**: `_parse_json_subtasks` `None`-safe.

### Phase 3 (NICE-TO-HAVE)
- [ ] **F7**: `extract_response_text` adopted in DualInferencer leaf calls; round files written by dict-returning leaves contain real newlines (verified by integration test).
- [ ] **F8**: F0 no-op convergence short-circuits the fix step.

---

## 9. Cross-references

### RankEvolve source-of-truth (`forensics_round_template_echo_bug`):
- `atlassian-packages/rankevolve/src/agentic_foundation/common/response_parsers/delimiter_parser.py` (markers + reverse-iter + `None` semantics).
- `atlassian-packages/rankevolve/src/agentic_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`
  - lines 433-456 / 513-539 / 613-643: `None`-handling at propose/review/fix
  - lines 651-684: F0 no-op detection (port basis for F8)
  - lines 1090-1132: hardened `_default_response_parser`
- `atlassian-packages/rankevolve/src/resources/prompt_templates/implementation/main/initial.jinja2:37-47` — `<ResponseSchema>` example pattern.

### AgentFoundation gap sites (live as of 2026-05-19 21:13):
- `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py:1070, 1248, 1420, 1795-1802`
- `src/agent_foundation/common/inferencers/flow_parsers.py:35, 111-117`
- `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py:123, 161, 340`
- `src/agent_foundation/common/inferencers/inferencer_base.py:954-966`
- `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/plan_then_implement_inferencer.py:962-997`
- `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py:735-755`

### Forensics note (preserve when porting)
RankEvolve cites `forensics_round_template_echo_bug` (original 607-byte corruption) in `delimiter_parser.py:49-50`, `dual_inferencer.py:659, 1114-1115`. When porting RankEvolve-derived code, **keep this reference comment** — it is doc-only (no runtime cost) and provides a debugging breadcrumb to the original incident.

---

## Appendix A — Reconciliation of the prior drafts

### A.1 Side-by-side comparison

| Dimension | Draft A (v2, this file pre-edit) | Draft B (updated `~/.claude/plans/...`) | v3 integration decision |
|---|---|---|---|
| Defense placement | Inline reimpl with `_PROMPT_ECHO_MARKERS` re-declared on `DualInferencer` | Inline reimpl with `_PROMPT_ECHO_MARKERS` re-declared on `DualInferencer` (after Draft B v2 update) | **v3: import from `delimiter_parser` (single source of truth).** Both drafts re-declared the markers — that's a drift hazard. v3 uses a function-local import to avoid circular imports while preserving SSoT. |
| Two-tag fallthrough | Inline `for tag in ("Response", "ImprovedProposal")` with `any_matches` tracking | Same (after Draft B v2 update) | Both correct; v3 keeps the form. |
| Template count | Said "12 files" (approximate lines) | Said "13 files / 14 blocks" — counted deep_research twice | **v3: VERIFIED 12 files / 14 blocks** via live grep. Draft B's table is otherwise correct but its count is wrong; v3 fixes the count and adds per-special-case guidance. |
| Special-case templates | Not called out | Not called out | **v3 NEW: explicit §2.5 + per-file F3 guidance** for `conversation/main/initial.jinja2` (runtime-contract use of `<Response>`), `deep_research/main/initial.jinja2` (two blocks), `task_breakdown/main/initial.jinja2` (large JSON example). |
| `MultiFlowDualInferencer` / `parse_finalplan_tag` sibling vulnerability | **Missed entirely** | **Identified** as out-of-scope follow-up | **v3 PROMOTES TO IN-SCOPE as F2-Sibling.** Leaving it unfixed violates Design Principle #7 (symmetry); fixing it is mechanically identical to F1/F2 and composes automatically via existing `response_parser` injection. |
| Phasing / PR shape / DoD / Risk & Rollback | Comprehensive | Lighter | **v3 keeps Draft A's structural rigor**, extends DoD with F2-Sibling entry and the bash-check acceptance criteria. |
| Forensics breadcrumbs | Mentioned | Mentioned | Kept in v3; promoted to Design Principle #8. |
| Out-of-scope statement | Implicit (Phase 3) | Explicit "Known follow-up" section | **v3 keeps explicit out-of-scope statements per fix** ("Why separate PR" sub-bullets in F7, F8); makes scope cuts auditable. |
| Test list | 9 + regression | 11 | **v3: 17 tests** (added 4 for `parse_finalplan_tag` and MultiFlow composition). |

### A.2 Real defects found in each draft (and how v3 resolves them)

**Draft A defects:**
1. Re-declared `_PROMPT_ECHO_MARKERS` on `DualInferencer` → drift hazard. **v3 fix:** function-local import from `delimiter_parser`.
2. Missed the `MultiFlowDualInferencer` / `parse_finalplan_tag` sibling vulnerability. **v3 fix:** added F2-Sibling.
3. Template line numbers were approximate. **v3 fix:** verified 12 files / 14 blocks by live grep; added per-file special-case handling.
4. Did not call out `conversation/main/initial.jinja2`'s runtime-contract use of `<Response>` → blindly renaming it would break the conversation agent. **v3 fix:** §2.5 explicit special-case; F3 strict-handling note.

**Draft B defects:**
1. **Original delegation bug** (Draft B v1): silently killed the `<ImprovedProposal>` fallback. **Acknowledged and fixed by Draft B v2; preserved in v3.**
2. Template count off-by-one ("13 files" vs. actual 12). **v3 fix:** verified count of 12 files / 14 blocks.
3. No phasing / PR shape / DoD checklist → harder to coordinate review. **v3 fix:** imported from Draft A.
4. F2-Sibling identified but explicitly punted to "Known follow-up". **v3 fix:** promoted to in-scope (Design Principle #7).
5. No special-case guidance for `conversation/main/initial.jinja2`, `deep_research/main/initial.jinja2`, `task_breakdown/main/initial.jinja2`. **v3 fix:** §2.5 + per-file F3 notes.

### A.3 Honest answer: if forced to pick ONE plan, which?

**Draft B (updated)** — with three non-negotiable caveats:

1. Draft B is more **investigative** (it discovered the MultiFlow sibling that Draft A missed) and more **concrete** (line-numbered template table). Those are real engineering virtues.
2. **BUT** Draft B's template count is wrong (13 vs. actual 12) and it punts the sibling fix to "follow-up" instead of recognizing it as in-scope. Anyone implementing Draft B verbatim would file PRs for 13 templates (one of which doesn't exist) and leave the MultiFlow vulnerability open.
3. **AND** Draft B lacks structured phasing/PR shape/DoD/rollback that makes a multi-file change safe to land in a team setting. It would land as "one giant PR" with no separation of concerns.

So: I would take **Draft B** for **investigative depth and concrete enumeration**, then patch the count, promote F2-Sibling to in-scope, and graft on Draft A's structural rigor. That is exactly what v3 in this file does.

**Picking Draft A alone** would ship working code but with: duplicated markers (drift hazard), missing MultiFlow fix, and approximate template line numbers. Risk of silent corruption via the unfixed `parse_finalplan_tag`.

**Picking Draft B alone** would ship: correct DualInferencer fix and well-enumerated templates, but with a wrong file count, an unfixed MultiFlow sibling, and no rollback discipline.

Neither alone is sufficient. v3 is not a compromise — it is a strict superset of the correct parts of both, with the bugs of each removed.
