# Conversational Workflows + Enhanced SOP Framework — Integrated v2 Plan

**Author:** Tony Chen (integrating Rovo Dev v1 + Claude plan + v3 runtime unification + v3.1 conditional-goto correction + v3.2 WorkGraph-rejected-alternative + runaway protection)
**Date drafted:** 2026-05-24 07:46 (v2) · 2026-05-24 11:30 (v3) · 2026-05-24 16:25 (v3.1) · 2026-05-24 17:12 (v3.2)
**Status:** Draft v3.2 — adds explicit rejected-alternative analysis for WorkGraph-as-substrate, runaway-protection knobs borrowed from WorkGraph's expansion-cap pattern, and explicit "no new loop grammar" conclusion from WorkGraph loop-subsystem investigation; pending review
**Scope:** Multi-domain plan spanning SOP grammar v2 enhancement, first-class workflow runtime in `ConversationalInferencer`, `task` tool relocation, new `sop` tool, and CLI UI module under `agent_foundation/ui/cli/`.

---

## §0. Audit history (round-by-round)

| Round | Verdict on Rovo Dev v1 | Action |
|---|---|---|
| **R0** (07:46) — comparison vs Claude plan | **Rovo Dev v1 was empirically wrong on two foundational claims:** (a) "No structured SOP parser exists" — FALSE, `SOPManager` exists at `RichPythonUtils/src/rich_python_utils/string_utils/formatting/template_manager/sop_manager.py:207` with `SOPPhase` (line 132), `SOPSubsection` (line 124), `StateNode` (line 38), `StateGraphTracker` (line 113); (b) "`agent_foundation/ui/` is empty seed" — FALSE, `RichInteractiveBase` exists at `agent_foundation/ui/rich_interactive_base.py:22`, `InputMode` at `ui/input_modes.py:11`, `ConversationToolType` at `conversational/conversation_tools.py:17`. **Claude's plan correctly extends these.** v2 adopts Claude's architectural pointers + retains v1's grammar formalization, risks, and ACs. |
| **R1** (11:30) — runtime unification with workgraph/StateGraph | **v2 underspecified the dynamic side.** Phase progression was "the LLM updates the prompt; somehow `completed_phases` grows" — no code-level contract. v2 also declared `tracker` as a separate field on `WorkflowInstance` even though `SOP IS a StateGraph` (verified: `sop_manager.py:144 — class SOP(StateGraph)`). v3 introduces: (1) `PhaseAwaiter` as a composition wrapper around `StateGraphTracker.complete()` (zero modifications to RichPythonUtils); (2) `WorkflowRuntime` as the async driver that owns the tracker and lifecycle loop; (3) explicit `complete_phase(**artifacts)` conversation tool — the client-server bridge between inferencer (client) and runtime (server); (4) `__branch__` execution via `asyncio.gather`+`asyncio.Semaphore` (same pattern WorkGraph uses at `workgraph.py:1541, 2676`); (5) `__goto__ __wait__` via durable `ScheduledThreadRegistry` for restarts. **The SOP runtime now reuses StateGraphTracker as its bookkeeper, WorkGraph's parallel-async pattern for branches, and asyncio.Event for one-shot awaiters — no parallel reinvention.** See §5.9, §5.10, §5.11 for details. |
| **R2** (16:25) — v3.1 conditional-goto / gate correction | **v3 silently treated `__goto__` as unconditional and proposed adding `goto_target` as a "new field" — both wrong.** StateGraph already implements full conditional goto/gate semantics: `StateNode.goto_target` (`stategraph.py:55`), `goto_condition_var` (`:56`), `goto_condition_value` (`:57`), `gate_var` (`:52`), `gate_value` (`:53`), `StateGraphTracker.goto_counts` (`:125`) bounded by `max_goto_iterations` (`:127`), `_check_condition()` (`:262`) supports both truthy and string-equality forms. The runtime gating at `:163-172` (goto) and `:183-185` (gate) is already in production. v3.1 corrects: (a) §4.2 adds `__if__` to EBNF (both `__goto__ ... __if__` and top-of-phase forms); (b) §4.5 removes 3 fields that already exist; (c) §4.6 adds `_GOTO_IF_RE` regex; (d) §4.7 acknowledges conditional firing; (e) §5.11.1 acknowledges `max_goto_iterations` is the runtime guard (not just lint); (f) §11.1 adds 3 RED tests for conditional goto/gate; (g) §10 adds the "complex condition expression" risk. Net: ~30 lines removed (overclaim), ~50 lines added (omitted surface); plan becomes more honest, no parallel reinvention. |
| **R3** (17:12) — v3.2 WorkGraph rejected-alternative + loop-grammar conclusion + runaway protection | **Two questions pre-empted, one bug fix.** (a) "Why not use WorkGraph as the SOP runtime substrate, given it has parallel async + dynamic subgraph expansion?" New §5.13 documents the rejected alternative with 4-row analysis: WorkGraph nodes are *callables* (SOP phases are *prompt-context waiting for inferencer signal* — wrong shape); WorkGraph owns the event loop (would invert ConversationalInferencer control flow); subgraph hierarchy ≠ SOP sibling branches; v3's `asyncio.gather+Semaphore` already steals WorkGraph's parallel primitive without inheriting structural mismatch. (b) "Should SOP grammar expose WorkGraph's loop subsystem (4 test files: forced_loop_mode, loop_resume, name_based_loop_counts, self_loop_expansion)?" Investigated empirically; WorkGraph's "loop" is **`NextNodesSelector(include_self=True)`-driven** (return-value pattern, not declarative attribute). Conclusion: **NO new loop markdown surface needed.** `__goto__ Phase X __if__ Y` + `max_goto_iterations` already covers the only loop pattern SOP authors plausibly need (conditional retry / iterate-while). The remaining WorkGraph loop primitives (forced expansion naming, name-based loop count migration, GraphExpansionResult-based self-expansion) are **code-only mechanics for code-authored DAGs** — exposing them in markdown would create a second DSL with no SOP-author benefit. §4 EBNF stays as is. (c) Three runaway-protection knobs borrowed from WorkGraph's `_max_expansion_depth` cap pattern (workgraph.py:204): `max_branch_fan_out`, `max_total_active_phases`, `max_concurrent_threads` added to `WorkflowRuntime` in §5.9.2. (d) New open question Q6 in §12 about v4 `__include__ another_workflow` (where WorkGraph's `subgraph_registry` becomes the right pattern to copy). |

**Net of integration:** v2 = Claude's empirical accuracy + v1's grammar formalization (EBNF + scope rules + `__goto__`/`__branch__` semantics) + v1's risk register + v1's acceptance criteria + Claude's yolo-mode mechanics + Claude's CLI component mapping.

---

## §1. Why this plan exists

Four threads couple in practice and must be designed together:

1. **SOP format v2** — markdown-friendly tags on separate lines (`__goto__`, `__afterwards__`, `__wait__`, `__branch__`, `__requires confirmation__`).
2. **First-class workflows in `ConversationalInferencer`** — workflows become peer to tools/skills: enter / exit / re-enter, stateful, identified by `workflow_id`, multiple concurrent instances.
3. **Move `task` tool** from `OpenStartup/openteam/server/resources/tools/task/` to `AgentFoundation/.../resources/tools/task/`.
4. **New `sop` tool** + new CLI UI module that enables non-yolo interactive SOP execution.

Single architectural pivot: **the SOP becomes a parsed, structured, stateful object** that the runtime understands, and `Workflow` becomes a runtime concept peer to tools/skills.

---

## §2. Empirical baseline (corrected from v1)

| Area | Reality (verified) | Source |
|---|---|---|
| **SOP parser** | EXISTS. `SOPManager.load(path) → SOP` produces a parsed AST. Handles inline directives (`## Phase 1 [__depends on__ Phase 0]`). | `RichPythonUtils/.../sop_manager.py:207` |
| **SOP AST types** | EXIST. `SOPPhase` extends `StateNode`; `SOPSubsection` for `**Tools**[__must__]:` blocks. | `sop_manager.py:124`, `:132` |
| **State graph + tracker** | EXIST. `StateGraphTracker` tracks phase progression with `state_outputs`, `current_phase`. | `RichPythonUtils/.../workflow/stategraph.py:38`, `:113` |
| **Workflow context in AF** | EXISTS. `WorkflowContext` at `agent_foundation/server/workflow_context.py` — per-session, single-instance, no enter/exit semantics. | (verified) |
| **`RichInteractiveBase`** | EXISTS. Already supports structured input modes (`InputMode.FREE_TEXT`, `SINGLE_CHOICE`, `MULTIPLE_CHOICE`). | `agent_foundation/ui/rich_interactive_base.py:22` |
| **`InputMode`** | EXISTS. | `agent_foundation/ui/input_modes.py:11` |
| **`ConversationToolType`** | EXISTS — enum with `CLARIFICATION`, `SINGLE_CHOICE`, `MULTIPLE_CHOICE`, `CONFIRMATION`, `TOOL_ARGUMENT_FORM`. | `conversational/conversation_tools.py:17` |
| **Yolo mode** | Single boolean on `RovoDevCliInferencer.yolo` — subprocess CLI flag only. NO inferencer-level filtering today. | `rovodev_cli_inferencer.py:75+` |
| **`task` tool** | At `OpenStartup/src/openteam/server/resources/tools/task/`. Two OpenStartup deps: `bootstrap.ensure_siblings_on_path` + `tool_cli.run_cli`. Otherwise framework-agnostic. | (verified) |
| **CLI UI references** | `rankevolve/src/cli` uses `rich` + `prompt_toolkit` + `click` cleanly. `acra-python` minimal. | (verified) |

**The core architectural insight v1 missed and Claude got right: we are *extending* an existing parser + workflow base, not building a new one.**

---

## §3. Goals and non-goals

### Goals
- Extend the existing `SOPManager` parser to support v2 grammar (separate-line tags, `__goto__`/`__afterwards__`/`__wait__`/`__branch__`).
- Add a new `WorkflowManager` in `agent_foundation/common/workflow/` that wraps the existing `StateGraphTracker` and adds enter/exit/re-entry/multi-instance semantics.
- Keep `WorkflowContext` as a per-instance state container (now owned by a `WorkflowInstance` rather than the session).
- Make `Workflow` a first-class peer to tools/skills in the `ConversationalInferencer` prompt.
- Move `task` tool to AgentFoundation; leave a thin shim in OpenStartup.
- Add new `sop` tool + new CLI UI module (`agent_foundation/ui/cli/`) that **extends `RichInteractiveBase`** to render Rich Panels + prompt_toolkit input dialogs.
- Add `yolo_mode` to `ConversationalInferencer` with explicit `_yolo_auto_resolve` mechanics.

### Non-goals (explicit)
- No rewriting `SOPManager` from scratch — extend in place.
- No rewriting `WorkflowContext` — preserve and re-purpose as per-instance state.
- No full TUI framework — small focused CLI components only.
- No database for workflow state — JSON persistence is sufficient for v1.
- No parallel/distributed workflow execution — single-process scope. Branch threads queued and executed sequentially in v1.
- No migration of other OpenStartup tools — only `task`.
- No first-class "skill" overhaul — skills exist partially; out of scope.

---

## §4. SOP grammar v2 — extending the existing `SOPManager`

### §4.1 Design principles
1. **Markdown-friendly first.** Tags on separate lines render cleanly in GitHub/Confluence previews.
2. **Tag scope is unambiguous.** Every tag has exactly one owner (phase / subsection / instruction line).
3. **Orchestration is declarative.** `__goto__` / `__branch__` / `__depends on__` describe state-machine intent; runtime owns execution.
4. **Two tag categories.** **Phase / subsection tags** (drive runtime behavior) vs. **instruction tags** (advisory; `[__requires confirmation__]` has filter semantics under yolo mode).
5. **Forward compatible.** Unknown tags MUST parse and survive round-trip; runtime warns once but does not error.
6. **Fully backward compatible.** OLD inline format (`## Phase 1 [__depends on__ Phase 0]`) continues to work; v2 is additive.

### §4.2 Production grammar (EBNF)

```ebnf
sop                = workflow_metadata? phase+

workflow_metadata  = "# Workflow:" workflow_name NEWLINE description_block?
                   ; Optional H1 line declaring display name.

phase              = phase_header (tag_line | inline_tags)? body subsection*
phase_header       = "##" "Phase" phase_id "--" title NEWLINE
phase_id           = integer suffix?
suffix             = letter+                     ; e.g. "b", "c"

inline_tags        = "[" tag (separator tag)* "]"     ; OLD: same line as header
tag_line           = blank_line "[" tag (separator tag)* "]" trailing_text? NEWLINE
                   ; NEW: separate line below header. Trailing text after the
                   ; bracket close (e.g., "[__requires confirmation__] IMPORTANT...")
                   ; is preserved as body content (matching Claude plan §1.3 #4).
separator          = "]" whitespace "["

subsection         = subsection_header (tag_line | inline_tags)? body tools_block?
subsection_header  = "###" title NEWLINE

tag                = orchestration_tag | semantic_tag | unknown_tag

orchestration_tag  = "__initial__"
                   | "__depends on__" phase_ref ("," phase_ref)*
                   | goto_directive                          ; see below — v3.1 expanded
                   | "__branch__" branch_arg?
                   | "__for each__" identifier "__in__" identifier
                   | if_directive                            ; v3.1 — top-of-phase gate

semantic_tag       = "__requires confirmation__"
                   | "__must__"
                   | "__optional__"
                   | "__prioritize__"
                   | "__requires_confirmation_first__"   ; legacy subsection variant

; ─── v3.1: __goto__ with optional condition AND optional afterwards/wait ───
goto_directive     = "__goto__" phase_ref if_clause? afterwards?
if_clause          = "__if__" condition                       ; v3.1 — goto guard
afterwards         = "__afterwards__" wait_arg?
wait_arg           = "__wait__" duration

; ─── v3.1: top-of-phase __if__ (compiles to StateNode.gate_var/gate_value) ───
if_directive       = "__if__" condition

; ─── v3.1: condition syntax — DELIBERATELY MINIMAL ───
; StateGraphTracker._check_condition (stategraph.py:262) supports only:
;   (1) truthy check:    bool(state_outputs[var])
;   (2) equality check:  str(state_outputs[var]) == value
; The grammar refuses to parse anything more complex (e.g., "X > 5", "X and Y",
; "not X") and fails fast with a clear error pointing to af-sop inspect.
condition          = identifier                              ; truthy check
                   | identifier "==" string_literal          ; equality check
string_literal     = "\"" any_char* "\"" | "'" any_char* "'"

duration           = integer time_unit
time_unit          = "s" | "m" | "h" | "d"

branch_arg         = "`" identifier "`" | identifier
                   ; e.g. "[__branch__ proposals]" — explicit branch source variable
                   ; matching Claude plan _BRANCH_RE pattern

phase_ref          = "Phase" phase_id
unknown_tag        = "__" identifier "__" (whitespace tag_arg)*
                   ; Preserved verbatim for forward compatibility.

tools_block        = "**Tools**" (tag_line | inline_tags) ":" NEWLINE tool_list
tool_list          = ("- " tool_name NEWLINE)+

body               = (text_line | blank_line)*
```

### §4.3 Scope rules (priority order)
1. A `tag_line` directly under a `phase_header` (allowing one blank line between) belongs to that phase.
2. A `tag_line` directly under a `subsection_header` belongs to that subsection.
3. A `tag_line` directly under a `**Tools**` block belongs to the tools subsection.
4. Inline `[__requires confirmation__]` mid-paragraph is an **instruction tag** — modifies only that sentence/bullet, not the phase.
5. Duplicate tags in the same scope are deduplicated; parser warns once.
6. Unknown tags preserved in `StateNode.unknown_tags: list[str]`; runtime ignores them.

### §4.4 Two-pass parsing (Claude §1.3, adopted)

In `SOPManager.parse_markdown()`, after extracting the body between headings:
1. Scan first lines of body for `_TAG_LINE_RE` matches (allow blank lines between heading and tag line).
2. Parse each tag line into known directives (`depends_on`, `goto`, `goto_afterwards`, `branch`, `requires_confirmation`, `for_each`, `initial`).
3. Merge with any inline directives from the heading (dedupe with warning).
4. Tag lines with trailing text preserve that text as body content.
5. Remaining body passes to `_parse_subsections()` as before.

This is fully backward-compatible — OLD inline format continues unchanged.

### §4.5 New fields on existing AST types (v3.1 — corrected)

**Critical correction (v3.1):** Earlier drafts proposed adding `goto_target`, `gate_var`, `gate_value`, `goto_condition_var`, `goto_condition_value` as new fields. **All five already exist on `StateNode`** (verified: `stategraph.py:52-57`). The parser must populate them; no field additions are required for conditional `__goto__` / `__if__`.

**Fields already in `StateNode` (parser populates these; do NOT redeclare):**
```python
# Already at stategraph.py:52-57 — RE-USE, do not duplicate
gate_var: str = attrib(default=None)                  # populated by top-of-phase __if__
gate_value: str = attrib(default=None)                # populated by __if__ X == "val"
goto_target: str = attrib(default=None)               # populated by __goto__ Phase X
goto_condition_var: str = attrib(default=None)        # populated by __goto__ ... __if__ X
goto_condition_value: str = attrib(default=None)      # populated by __goto__ ... __if__ X == "val"
# Already at stategraph.py:42 — RE-USE
foreach_collection_var: str = attrib(default=None)    # populated by __for each__
# Already at stategraph.py:125-127 — runtime guard, configurable via SOP frontmatter
# goto_counts: dict[str, int] / max_goto_iterations: int = 10
```

**Genuinely new fields v3.1 must add to `StateNode` (4 fields, all absent today):**
```python
# stategraph.py — small extension
@attrs
class StateNode:
    # ... existing fields above ...
    goto_afterwards: bool = attrib(default=False)            # __goto__ ... __afterwards__
    goto_wait_duration: str | None = attrib(default=None)    # "1h" / "30m" / "60s" / "2d"
    branch: bool = attrib(default=False)                     # __branch__ present
    branch_source_var: str | None = attrib(default=None)     # explicit identifier
    unknown_tags: list[str] = attrib(factory=list)           # forward-compat
```

**SOPPhase extensions (genuinely new):**
```python
# sop_manager.py — extend SOPPhase
@attrs
class SOPPhase(StateNode):
    # ... existing fields ...
    requires_confirmation: bool = attrib(default=False)      # programmatic access
    instruction_confirmations: list[str] = attrib(factory=list)
                                                              # inline-tagged sentences
                                                              # (for non-yolo prompts)
```

**Why this matters:** if v3.1 had naively re-declared `goto_target` / `gate_var` on a SOPPhase subclass, `attrs` field inheritance would have either silently shadowed the parent (breaking StateGraphTracker's runtime checks at `stategraph.py:163-185`) or required `kw_only=True` plumbing to disambiguate. Removing the redundant declarations eliminates a class of subtle bugs.

### §4.6 New regex patterns (v3.1 — unified `__goto__` matcher + `__if__`)

```python
_TAG_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)?$", re.MULTILINE)

# v3.1: single unified __goto__ matcher
# Matches: __goto__ Phase X [__if__ var [== "val"]] [__afterwards__ [__wait__ Nh]]
# All trailing clauses are optional and order-fixed (__if__ before __afterwards__).
_GOTO_RE = re.compile(
    r"__go\s*to__\s+Phase\s+(?P<target>\w+)"
    r"(?:\s+__if__\s+(?P<cond_var>\w+)"
    r"(?:\s*==\s*[\"'](?P<cond_val>[^\"']+)[\"'])?)?"
    r"(?:\s+__afterwards__(?:\s+__wait__\s+(?P<wait>\d+[smhd]))?)?",
    re.IGNORECASE,
)

# v3.1: top-of-phase __if__ (compiles to StateNode.gate_var / gate_value)
# Matches: __if__ var [== "val"]
# Distinguishable from __goto__ ... __if__ because this stands alone in the tag line.
_IF_GATE_RE = re.compile(
    r"^__if__\s+(?P<gate_var>\w+)"
    r"(?:\s*==\s*[\"'](?P<gate_val>[^\"']+)[\"'])?\s*$",
    re.IGNORECASE,
)

_BRANCH_RE = re.compile(r"__branch__(?:\s+`?(\w+)`?)?", re.IGNORECASE)

# v3.1: condition complexity guard — anything more than `var [ == "val" ]` is rejected.
# Catches accidental Python-expression authoring like "X > 5", "X and Y", "not X".
_CONDITION_VALID_RE = re.compile(
    r"^\w+(?:\s*==\s*[\"'][^\"']*[\"'])?$"
)

def validate_condition(condition_text: str, sop_file: str, line_no: int) -> None:
    if not _CONDITION_VALID_RE.match(condition_text.strip()):
        raise SOPGrammarError(
            f"{sop_file}:{line_no}: condition {condition_text!r} is too complex. "
            f"Supported: 'var' (truthy) or 'var == \"value\"' (equality). "
            f"Run `af-sop inspect` for detailed diagnostics."
        )
```

**Why merge `_GOTO_AFTERWARDS_RE` into a single `_GOTO_RE`:** the three optional clauses (`__if__`, `__afterwards__`, `__wait__`) are orthogonal. Three separate regexes would require three sweeps over the same tag content, with order-of-application sensitivity. One regex with named groups makes the data flow explicit and parsing trivially correct.

### §4.7 `__goto__` / `__if__` / `__afterwards__` semantics (v3.1 — conditional firing)

> User quote: *"`__goto__ Phase 3 __afterwards__ __wait__ 1h` is an orchestration tag … there is a thread of the SOP going back to phase 3 after the current phase 3b … and also that thread starts with wait time 1hour."*

**Critical correction (v3.1):** `__goto__` is **NOT** unconditionally fired. `StateGraphTracker._check_condition()` (`stategraph.py:262`) already evaluates the condition; only if it holds does the runtime re-enable the goto target. The parser maps `__goto__ ... __if__ X` to `StateNode.goto_condition_var`; runtime gating is automatic.

| Directive | Fires when | StateNode fields populated |
|---|---|---|
| `__goto__ Phase X` | **Always** after parent completes | `goto_target="X"` |
| `__goto__ Phase X __if__ Y` | After parent completes AND `bool(state_outputs["Y"])` is True | `goto_target="X"`, `goto_condition_var="Y"` |
| `__goto__ Phase X __if__ Y == "z"` | After parent completes AND `str(state_outputs["Y"]) == "z"` | `goto_target="X"`, `goto_condition_var="Y"`, `goto_condition_value="z"` |
| `__goto__ Phase X __afterwards__` | Same as `__goto__ Phase X` (temporal qualifier) | `goto_target="X"`, `goto_afterwards=True` |
| `__goto__ Phase X __afterwards__ __wait__ 1h` | Same; spawned thread sleeps `wait` duration before its first step | `goto_target="X"`, `goto_afterwards=True`, `goto_wait_duration="1h"` |
| `__goto__ Phase X __if__ Y == "z" __afterwards__ __wait__ 1h` | All conditions combined | All 5 fields populated |

**Top-of-phase `__if__` (gate, not goto):**

| Directive | Fires when | StateNode fields populated |
|---|---|---|
| `## Phase 4\n[__if__ Y]` | Phase 4 only becomes available when `bool(state_outputs["Y"])` is True | `gate_var="Y"` |
| `## Phase 4\n[__if__ Y == "z"]` | Same, with equality check | `gate_var="Y"`, `gate_value="z"` |

The runtime check is at `stategraph.py:183-185` — already in production.

**Multiple `__goto__` directives → multiple threads.** Tracked via `WorkflowThread(parent_instance_id, thread_id, target_phase, wake_time, branch_item)`.

**Runtime contract (v3.1 — cleaner than v3):** the runtime does NOT need a separate `get_pending_thread_spawns()` API. `StateGraphTracker.get_available_next()` (`stategraph.py:155`) already re-resolves available phases after a `complete()` call — and that resolution already honors `goto_target` + condition (`:163-172`) and `gate_var` (`:183-185`). `WorkflowRuntime.drive()` (§5.9.2) simply loops on `get_available_next()` and activates each result. **All goto/if/gate semantics flow through the existing tracker without a new API surface.**

### §4.8 `__branch__` semantics (v1 formalization, preserved + Claude execution model)

> User quote: *"'branch' means the previous phase will output a list of items, and each item triggers a separate next phase."*

```
## Phase 3b -- Proposals
[__must__ output: proposals: list[Proposal]]

## Phase 4 -- Implement Proposal
[__depends on__ Phase 3b] [__branch__ proposals]
```

**Runtime contract (Claude §1.4, adopted):** `StateGraphTracker.get_branch_items(node_id)` looks up the branch source variable in `state_outputs`. Returns `list[Any]` if branching applies, `None` otherwise.

**Execution model (Claude §1.4):** Parallel via `task_queue` with configurable `max_parallel_tasks`. When a branch phase becomes available with N items, runtime creates N task-queue entries. Each carries full workflow context plus its specific `branch_item`. **v1: max_parallel_tasks defaults to 1 (sequential).** v2 can lift.

**Failure mode:** If `branch_source_var` is missing from `state_outputs` or not a list, `WorkflowManager` raises `BranchSourceMissing` with the parent phase's actual outputs dumped in the error. `af-sop inspect` catches this statically.

### §4.9 Yolo-mode rendering (Claude §1.5 + v1 edge-case discipline)

Add to `SOPManager`:
```python
def render_for_mode(sop: SOP, mode: Literal["default", "yolo"] = "default") -> str:
    """
    mode='yolo': Re-renders SOP markdown with all [__requires confirmation__]
                 tag lines AND their trailing instruction text stripped.
    mode='default': Returns SOP text as-is.
    """
```

**Renderer is separate from parser** — AST is invariant; mode only affects text output.

**Edge cases the filter MUST handle correctly** (v1 risk #6, preserved):
1. Marker inside a code block — DO NOT strip.
2. Marker inside a markdown link `[__requires confirmation__](url)` — DO NOT strip.
3. Marker mid-sentence — strip from the line containing it, not adjacent lines.
4. Marker followed by trailing instruction text on the same line — strip both.
5. Marker on its own dedicated tag line below a phase header — strip the entire tag-line entry from the phase's tag list AND the line itself.
6. Multiple markers on one line — single removal.

Six unit tests required, one per edge case.

### §4.10 Subsection tag parsing (Claude §1.6, adopted)
Existing `_SUBSECTION_RE` captures `**Tools**[__must__]:`. Verify it handles `requires_confirmation_first`. Regex `\w[\w\s]*` covers all current cases. Subsection directives parse into `SOPSubsection.directive`.

### §4.11 OLD-format losses to avoid
The current `code_optimization.md` (the v2 example) is **partial** — it lacks `__for each__` and `__if__` from OLD format. The full grammar in §4.2 preserves both. When migrating `model_optimization.md` (Phase 8), these MUST round-trip.

---

## §5. Workflow runtime — first-class concept

### §5.1 New module structure (v3 adds runtime.py + awaiter.py)
```
agent_foundation/common/workflow/
├── __init__.py
├── definition.py       # WorkflowDefinition (static, parsed from .md)
├── instance.py         # WorkflowInstance (stateful, holds SOP + outputs snapshot)
├── manager.py          # WorkflowManager (lifecycle: enter/exit/resume/complete)
├── registry.py         # WorkflowRegistry (discovery)
├── awaiter.py          # NEW v3 — PhaseAwaiter (async client-server bridge)
├── runtime.py          # NEW v3 — WorkflowRuntime (async driver loop, owns tracker)
└── scheduled.py        # NEW v3 — ScheduledThreadRegistry (durable __wait__)
```

**Layer responsibilities (clean separation):**
- `definition.py` — static parsing (no runtime state).
- `instance.py` — stateful per-instance container (SOP, outputs snapshot, threads list).
- `manager.py` — orchestrates instances per session (enter/exit/resume/complete).
- `runtime.py` — owns the **`StateGraphTracker`** per active instance + the async drive loop.
- `awaiter.py` — one-shot `asyncio.Event`-based phase awaiters.
- `scheduled.py` — JSON-persisted scheduler for `__wait__` directives that must survive restarts.
- `registry.py` — file-system discovery of workflow definitions.

**Critical architectural reframe (v3):** `WorkflowInstance` does NOT own the tracker. The tracker is **runtime state**, not instance state — it lives in `WorkflowRuntime`. The instance owns only the durable snapshot (`completed_phases`, `state_outputs`, `pending_threads`). When an instance is resumed, the runtime reconstructs a fresh `StateGraphTracker` from the snapshot. This mirrors how `Resumable` (`workflow/common/resumable.py:38-184`) persists results without persisting in-flight coroutine state.

### §5.2 `WorkflowDefinition` (Claude §2.2, adopted + v1 typing)
```python
@dataclass
class WorkflowDefinition:
    workflow_id: str           # derived from filename stem: "code_optimization"
    name: str                  # display name (H1 or frontmatter)
    description: str           # text before first ## Phase heading
    sop: SOP                   # parsed SOP from SOPManager.load()
    sop_path: str              # original file path
    available_tools: list[str] # extracted from SOP tool subsections
    metadata: dict[str, Any]   # optional YAML frontmatter
```

### §5.3 `WorkflowInstance` + `WorkflowThread` (v3 — tracker moved to runtime)
```python
@dataclass
class WorkflowInstance:
    """Durable per-instance state. Reconstructible from snapshot."""
    instance_id: str                  # uuid + workflow_id prefix
    definition_id: str
    workflow_context: WorkflowContext # REPURPOSED — now per-instance state container
    status: Literal["active", "suspended", "completed", "failed"]
    created_at: float
    last_active_at: float

    # Durable snapshot (NOT the live tracker — see §5.9 §runtime architecture)
    tracker_snapshot: dict[str, Any]  # StateGraphTracker.to_dict() output
    context_snapshot: dict[str, Any]  # ConversationalInferencer prior_context at suspend
    pending_threads: list[WorkflowThread]  # __goto__/__branch__ spawned threads

@dataclass
class WorkflowThread:
    thread_id: str
    parent_instance_id: str
    target_phase: str
    wake_time: float | None           # epoch; set by __wait__ — see §5.11
    branch_item: Any | None           # set by __branch__
    branch_source_phase: str | None   # phase whose output produced this branch_item
    status: Literal["pending", "active", "completed", "failed"]

# IMPORTANT v3: the runtime tracker is NOT a field of WorkflowInstance.
# See §5.9. The instance holds tracker_snapshot (serializable dict);
# WorkflowRuntime rebuilds a live StateGraphTracker from it on enter/resume.
```

**Why the change:** `StateGraphTracker` (`stategraph.py:113`) is a sync dataclass with no event mechanism. To make it "client-server-shaped" we wrap it from outside (composition), not modify it. The wrapper (`WorkflowRuntime`) is naturally async and per-execution; persisting it across restarts is wrong (it would carry transient asyncio state). Persisting only the *outputs* (`to_dict()`) is the well-known checkpoint pattern (cf. `resumable.py:77-117`).

### §5.4 `WorkflowRegistry` (Claude §2.4, adopted)
Discovery scans `resources/prompt_templates/conversation/main/_variables/workflow_sop/` for `.md` files.

**Supports two layouts (Claude flexibility, kept):**
- Flat: `workflow_sop/<workflow_id>.md`
- Directory: `workflow_sop/<workflow_id>/WORKFLOW.md` + `sop.md` (allows long workflows to colocate supplementary docs)

Additional scan dirs configurable.

### §5.5 `WorkflowManager` (Claude §2.5, adopted)
```python
class WorkflowManager:
    def __init__(self, registry: WorkflowRegistry): ...

    # Lifecycle
    def enter_workflow(self, workflow_id: str, **params) -> WorkflowInstance: ...
    def exit_workflow(self, instance_id: str | None = None) -> None: ...   # suspend
    def resume_workflow(self, instance_id: str) -> WorkflowInstance: ...
    def complete_workflow(self, instance_id: str) -> None: ...

    # State
    @property
    def focused_instance(self) -> WorkflowInstance | None: ...
    @property
    def active_instances(self) -> dict[str, WorkflowInstance]: ...

    # Prompt integration
    def render_prompt_sections(self) -> dict[str, str]: ...
    # Returns: available_workflows, ongoing_workflows,
    #          workflow_description, workflow_status, workflow_nextstep_guidance

    # Serialization
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict, registry: WorkflowRegistry) -> WorkflowManager: ...
```

### §5.6 Prompt integration (Claude §2.6, adopted)

**`ConversationalInferencer` changes:**
- Add `workflow_manager: WorkflowManager | None = attrib(default=None, kw_only=True)`
- In `_render_prompt()`: when `workflow_manager` is set, delegate to `workflow_manager.render_prompt_sections()` for all workflow-related template variables.
- When `workflow_manager` is None: fall back to current behavior (find_sop_file + direct tracker) — **strict backward compat**.

**Template additions to `initial.jinja2`:**
```jinja2
{# Always show available workflows (like tools) #}
{% if available_workflows is defined and available_workflows %}
## Available Workflows
{{ available_workflows }}
{% endif %}

{# Ongoing/suspended workflows #}
{% if ongoing_workflows is defined and ongoing_workflows %}
## Ongoing Workflows
{{ ongoing_workflows }}
{% endif %}

{# Full SOP context only when a workflow is focused #}
{% if workflow_description is defined and workflow_description %}
<WorkflowDescription>{{ workflow_description }}</WorkflowDescription>
<WorkflowStatus>{{ workflow_status }}</WorkflowStatus>
<WorkflowNextStepGuidance>{{ workflow_nextstep_guidance }}</WorkflowNextStepGuidance>
{% endif %}
```

### §5.7 Four new lifecycle tools (Claude tool surface, v1 expansion)

| Tool | Location | Effect |
|---|---|---|
| `enter_workflow` | `resources/tools/enter_workflow/` | Instantiate `WorkflowInstance`; set focus; persist |
| `exit_workflow` | `resources/tools/exit_workflow/` | Suspend focused instance; instance stays in `active_instances` |
| `resume_workflow` | `resources/tools/resume_workflow/` | Re-focus a suspended instance |
| `complete_workflow` | `resources/tools/complete_workflow/` | Mark `completed`; remove from active |

Surfaced as ordinary tools — agent decides when to invoke.

### §5.8 Backward compatibility

The existing `WorkflowContext` is **kept as a class** but its role changes from "per-session singleton" to "per-instance state container owned by a `WorkflowInstance`." Migration:
- Old session JSON with top-level `workflow_context` field loads with `DeprecationWarning`; `WorkflowManager.from_dict()` wraps it as a single `WorkflowInstance` with `instance_id` derived from `session_id + workflow_id`.
- New session JSON has `workflow_manager: {instances: {...}, focused_instance_id: ...}`.
- Both shapes coexist for 1 release.

### §5.9 Runtime architecture — `PhaseAwaiter` + `WorkflowRuntime` (NEW v3)

The single biggest gap in v2 was that **phase advancement was implicit** — the LLM "knew" via the prompt, and `completed_phases` got updated "somehow." v3 makes the advancement explicit, async, and code-grounded by composing three existing primitives:

| Existing primitive | Role in v3 runtime |
|---|---|
| `StateGraphTracker.complete(state_id, **outputs)` (`stategraph.py:137`) | The "phase done, here are outputs" call. v3 keeps this sync. |
| `StateGraphTracker.get_available_next() → list[StateNode]` (`stategraph.py:155`) | The dependency resolver. Naturally returns multiple — v3 runs them in parallel. |
| `asyncio.Event` + `asyncio.Queue` (Python stdlib, already used in `web_interactive.py` and `workgraph.py:1268`) | The async signal primitive. |
| `WorkGraph._arun()`'s `asyncio.gather` + per-group `asyncio.Semaphore` (`workgraph.py:2676, 2670`) | The parallel-execution pattern v3 uses for `__branch__`. |

#### §5.9.1 `PhaseAwaiter` — the async client-server bridge

```python
# awaiter.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PhaseAwaiter:
    """One-shot async waiter for a single phase activation.

    Created by WorkflowRuntime when a phase is activated. The ConversationalInferencer
    (the client) eventually calls `signal_complete(**artifacts)` via the new
    `complete_phase` conversation tool. The runtime (the server) awaits via `wait()`.
    """
    phase_id: str
    branch_item: Any | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event)
    _outputs: dict[str, Any] = field(default_factory=dict)
    _error: BaseException | None = None

    async def wait(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the inferencer signals completion."""
        if timeout is not None:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        else:
            await self._done.wait()
        if self._error is not None:
            raise self._error
        return dict(self._outputs)

    def signal_complete(self, **outputs: Any) -> None:
        """Called by the inferencer (via complete_phase tool) to wake the awaiter."""
        if self._done.is_set():
            raise RuntimeError(f"PhaseAwaiter[{self.phase_id}] already signaled")
        self._outputs.update(outputs)
        self._done.set()

    def signal_failure(self, error: BaseException) -> None:
        self._error = error
        self._done.set()
```

**Properties:**
- ~25 LoC. No subclassing of RichPythonUtils types.
- One-shot — re-activating the same phase creates a new awaiter (matches `__branch__` semantics where each iteration is a new logical phase activation).
- Thread-safe enough for asyncio (single event-loop scope).

#### §5.9.2 `WorkflowRuntime` — the async driver

```python
# runtime.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from rich_python_utils.common_objects.workflow.stategraph import StateGraphTracker
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOP, SOPPhase
from .awaiter import PhaseAwaiter
from .instance import WorkflowInstance, WorkflowThread
from .scheduled import ScheduledThreadRegistry

@dataclass
class WorkflowRuntime:
    """Owns the live StateGraphTracker for one focused WorkflowInstance and
    drives phase progression by awaiting PhaseAwaiters signaled by the inferencer."""
    instance: WorkflowInstance
    sop: SOP                                              # SOP IS a StateGraph
    scheduler: ScheduledThreadRegistry
    max_parallel_branches: int = 1                        # v3 default; configurable

    # ─── v3.2: runaway-protection knobs (borrowed from WorkGraph cap pattern) ───
    # Pattern from worknode_base.py:204 (_max_expansion_depth) — applied to OUR
    # failure modes. ALL three default to safe-ish values; override per-SOP via frontmatter.
    max_branch_fan_out: int = 50                          # max N items a single __branch__ may fan into
    max_total_active_phases: int = 100                    # total active phases (including branches+threads)
    max_concurrent_threads: int = 8                       # cap on simultaneously-running __goto__ threads
    # When any cap is hit, runtime raises RunawayWorkflowError with the SOP file, phase id,
    # and current count — never silently degrades.

    inferencer_focus_callback: Callable[[SOPPhase, Any | None], Awaitable[None]] | None = None
    inferencer_render_callback: Callable[[], Awaitable[None]] | None = None

    # Internal — populated at start time:
    _tracker: StateGraphTracker = field(init=False)
    _live_awaiters: dict[str, PhaseAwaiter] = field(default_factory=dict, init=False)
    _branch_semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        # Rehydrate tracker from instance snapshot (idempotent on first activate).
        self._tracker = StateGraphTracker.from_dict(
            {**self.instance.tracker_snapshot, "graph": self.sop}
        ) if self.instance.tracker_snapshot else StateGraphTracker(graph=self.sop)
        self._branch_semaphore = asyncio.Semaphore(self.max_parallel_branches)

    # ----- Server-side API (the runtime drives) -----
    async def drive(self) -> None:
        """Main driver loop. Runs until all phases are complete OR instance is suspended.

        Each iteration:
          1. Ask the tracker for available next phases (zero, one, or many).
          2. For each available phase (or branch fan-out), activate it and await.
          3. When a phase signals complete, record its outputs into the tracker.
          4. If the phase declared __goto__ or __branch__, enqueue threads / spawn branches.
        """
        while self.instance.status == "active" and not self._is_complete():
            next_nodes = self._tracker.get_available_next()
            if not next_nodes:
                # No available work — either we're done, or we're waiting on
                # scheduled threads. Yield briefly.
                if self._is_complete():
                    self.instance.status = "completed"
                    break
                await asyncio.sleep(0.1)
                continue

            # Parallel activation when multiple are available (e.g., from __branch__)
            await asyncio.gather(
                *[self._activate_and_wait(node) for node in next_nodes],
                return_exceptions=False,
            )

    async def _activate_and_wait(self, node: SOPPhase, branch_item: Any | None = None) -> None:
        """Activate one phase, await its completion signal, record outputs."""
        async with self._branch_semaphore:                # honors max_parallel_branches
            awaiter = PhaseAwaiter(phase_id=node.id, branch_item=branch_item)
            self._live_awaiters[node.id] = awaiter
            self._tracker.start(node.id)
            try:
                if self.inferencer_focus_callback:
                    await self.inferencer_focus_callback(node, branch_item)
                if self.inferencer_render_callback:
                    await self.inferencer_render_callback()
                outputs = await awaiter.wait()
                self._tracker.complete(node.id, **outputs)
                await self._handle_post_completion_directives(node, outputs)
            finally:
                self._live_awaiters.pop(node.id, None)
                self._snapshot_into_instance()

    # ----- Client-side API (the inferencer calls these via complete_phase tool) -----
    def signal_phase_complete(self, phase_id: str, **artifacts: Any) -> None:
        """Called by ConversationalInferencer via complete_phase conversation tool."""
        awaiter = self._live_awaiters.get(phase_id)
        if awaiter is None:
            raise RuntimeError(
                f"No live awaiter for phase {phase_id!r}. "
                f"Live: {list(self._live_awaiters)}. "
                f"Maybe the agent tried to complete a non-active phase?"
            )
        awaiter.signal_complete(**artifacts)

    # ----- __goto__ / __branch__ handling -----
    async def _handle_post_completion_directives(self, node: SOPPhase, outputs: dict) -> None:
        """After a phase completes, spawn threads for __goto__ and __branch__."""
        # __goto__ Phase X __afterwards__ __wait__ Yh
        if node.goto_target:
            wake = self._compute_wake_time(node.goto_wait_duration)
            thread = WorkflowThread(
                thread_id=f"goto-{node.id}-{node.goto_target}",
                parent_instance_id=self.instance.instance_id,
                target_phase=node.goto_target,
                wake_time=wake,
                branch_item=None,
                branch_source_phase=node.id,
                status="pending",
            )
            self.instance.pending_threads.append(thread)
            if wake is not None and wake > asyncio.get_event_loop().time():
                self.scheduler.schedule(thread)            # see §5.11
            else:
                # Wake immediately — the next drive() iteration will pick it up
                # via a new phase activation (tracker re-resolves dependencies).
                pass
        # __branch__: handled by tracker — see §5.11.2
```

**Note the design's two clean halves:**
- **Server-side (`drive`, `_activate_and_wait`, `_handle_post_completion_directives`)** — what the runtime does autonomously.
- **Client-side (`signal_phase_complete`)** — the only entry point exposed to the inferencer.

This is precisely the client-server node pattern. The inferencer **chitchats**, **calls conversation tools** (clarification, choice, confirmation), **decides** the phase is done, **calls `complete_phase(**artifacts)`**, which routes through the inferencer's tool handler into `runtime.signal_phase_complete(...)`. The awaiter wakes. The driver records the outputs. The tracker advances. The next phase activates. Loop.

### §5.10 The `complete_phase` conversation tool — inferencer's only new surface (NEW v3)

The inferencer needs **one new conversation tool** to bridge to the runtime: `complete_phase`. Everything else stays the same.

```python
# conversational_inferencer.py additions (post-v3 patch)

# 1. New enum member
class ConversationToolType(Enum):
    # ... existing types ...
    COMPLETE_PHASE = "complete_phase"

# 2. New optional attrib
class ConversationalInferencer(...):
    workflow_runtime: "WorkflowRuntime | None" = attrib(default=None, kw_only=True)
    # When set, the agent gets `complete_phase` exposed as a tool; calling it
    # routes to runtime.signal_phase_complete(active_phase_id, **artifacts).

# 3. New tool-handler branch in _handle_conversation_tool
def _handle_conversation_tool(self, tool: ConversationTool) -> dict:
    # ... existing branches ...
    if tool.type == ConversationToolType.COMPLETE_PHASE:
        if self.workflow_runtime is None:
            raise RuntimeError("complete_phase called but no workflow_runtime attached")
        active = self._infer_active_phase_id()         # from prior_context.focused_phase
        artifacts = tool.metadata.get("artifacts", {})
        self.workflow_runtime.signal_phase_complete(active, **artifacts)
        return {"phase_completed": active, "ok": True}
```

**That is the entire client-server bridge.** ~30 LoC of additions to `conversational_inferencer.py` plus ~25 LoC for the new tool descriptor. The existing agentic loop (`run_conversation` lines 907–1082) is **unchanged**.

**Failure-mode considerations:**
- If the LLM calls `complete_phase` with an inactive phase id, the runtime raises a clear error captured as a tool error in the next LLM turn — model self-corrects.
- If the LLM never calls `complete_phase`, the awaiter times out per `WorkflowRuntime.drive()`'s `timeout` parameter (default unbounded; configurable per-phase via SOP frontmatter).
- If multiple LLM turns repeatedly fail, runtime captures the error and surfaces it as a phase failure — `signal_failure(error)` flips the awaiter to error state.

### §5.11 `__goto__` / `__branch__` runtime execution (NEW v3)

#### §5.11.1 `__goto__ Phase X [__afterwards__] [__wait__ Yh]`

| Sub-case | Runtime behavior |
|---|---|
| `__goto__ Phase X` (no wait) | After current phase completes, `_handle_post_completion_directives` appends a `WorkflowThread` to `instance.pending_threads`. The tracker's dependency graph for Phase X already accepts re-entry; the next `drive()` iteration's `get_available_next()` picks it up. |
| `__goto__ Phase X __afterwards__ __wait__ 1h` | Same, but the `WorkflowThread.wake_time = now + 1h`. The `ScheduledThreadRegistry` (§5.11.3) persists this to JSON. A background coroutine in `drive()` polls scheduled threads; when one wakes, it marks the thread `pending → active` and the next iteration picks it up. |
| Cycle (`__goto__` re-enters a node already executed many times) | **Runtime guard is in production:** `StateGraphTracker.goto_counts` (`stategraph.py:125`) tracks `f"{node.id}->{node.goto_target}"` re-entries; capped by `max_goto_iterations` (`:127`, default 10). When the cap is hit, the goto silently no-ops (the target is not re-enabled at `:170`). v3.1 surfaces this via SOP frontmatter: `max_goto_iterations: 20` overrides the default per-workflow. Static lint (`af-sop inspect`) is additive — warns on detected cycles without `max_goto_iterations` override, but does not replace the runtime guard. |
| Conditional goto cycle (e.g., infinite retry loop where condition never flips) | Same `max_goto_iterations` cap protects — but author should also lint via `af-sop inspect` which traces likely state-output values to detect "this condition can never become False" patterns. |

**Restart safety:** If the process restarts mid-wait, the registry's JSON survives. On rehydration, `WorkflowManager.from_dict()` re-creates the thread; `WorkflowRuntime.__post_init__` re-arms the scheduler.

#### §5.11.2 `__branch__` — N parallel children, one convergent downstream

**Pattern:** Phase `3b` outputs `proposals: list[Proposal]`. Phase `4` is declared `[__depends on__ Phase 3b] [__branch__ proposals]`. We want N parallel instances of Phase 4 (one per proposal), all converging into Phase 5.

**The clean composition (no new abstractions):**

```python
# Inside _handle_post_completion_directives, when node has branch consumers:
async def _spawn_branches_for(self, completing_node: SOPPhase) -> None:
    """When phase 3b completes with proposals, this fans out phase 4 instances."""
    sop_outputs = self._tracker.state_outputs
    for downstream in self.sop.get_nodes_branching_from(completing_node.id):
        # downstream is e.g. Phase 4 with branch=True, branch_source_var="proposals"
        items = sop_outputs.get(downstream.branch_source_var)
        if items is None or not isinstance(items, list):
            raise BranchSourceMissing(
                f"Phase {downstream.id} branches on {downstream.branch_source_var!r} "
                f"but parent phase {completing_node.id} did not emit a list. "
                f"Got: {type(items).__name__}={items!r}"
            )
        # Spawn N parallel activations — same _activate_and_wait, different branch_item.
        # asyncio.gather + semaphore = the WorkGraph._arun parallel pattern (workgraph.py:2676)
        await asyncio.gather(
            *[self._activate_and_wait(downstream, branch_item=item) for item in items],
            return_exceptions=False,                       # one fail = whole branch fails
        )
        # When all N complete, their outputs are aggregated into sop_outputs[node.id]
        # as a list — downstream Phase 5 sees aggregated artifacts.
```

**Critical leverage point**: this is **exactly the same `asyncio.gather + Semaphore` pattern** used by `WorkGraph._arun()` at `workgraph.py:1541` and `workgraph.py:2676`. We don't introduce a new orchestration model — we apply the existing one.

**`max_parallel_branches` semantics:**
- Default `1` (sequential — minimal cost, simplest debugging).
- Set higher to allow concurrent branch execution.
- The single `asyncio.Semaphore` on the runtime gates *all* activations (branches AND `__goto__` threads), preventing runaway concurrency.

**Failure isolation:**
- v3 default: `return_exceptions=False` — one branch fails, whole branch group fails fast.
- Optional `branch_isolation=True` flag on the SOP phase (v3.1) flips to `return_exceptions=True`, collecting per-item failures into the aggregated outputs.

**Result aggregation:**
- After all N awaiters complete, `_activate_and_wait` writes individual `branch_item` results into `state_outputs[downstream.id]` as a list `[{branch_item: ..., outputs: ...}, ...]`.
- Phase 5's prompt sees this aggregation through the standard SOP renderer (`SOPManager.render_guidance` — existing call site at `conversational_inferencer.py:687`).

**Cost / context implications (honest tradeoffs):**

| Design choice | Cost | Context | Failure | Aggregation |
|---|---|---|---|---|
| **One inferencer, N parallel asyncio.Tasks (v3 chosen)** | N parallel LLM calls; serial only if `max_parallel=1` | Each task carries `branch_item`; shares parent SOP context via prompt rendering | Fast-fail default; per-item isolation optional | Built-in aggregation list |
| One inferencer, N sequential tasks | Cheaper if rate-limited; serial latency | Identical context model | Per-item recovery natural | Identical aggregation |
| Subprocess per branch | Process isolation; OS overhead; serialization cost | Hard to share parent inferencer state | True isolation | Requires IPC |

**v3 picks #1** because (a) it reuses the existing `asyncio.gather` pattern from WorkGraph, (b) it shares the parent inferencer's KV cache when supported, (c) `max_parallel_branches` gives knob control for cost.

#### §5.11.3 `ScheduledThreadRegistry` (durable `__wait__`)

```python
# scheduled.py
class ScheduledThreadRegistry:
    """JSON-persisted schedule for WorkflowThreads with wake_time.

    On start, reads <workspace>/_runtime/scheduled_threads.json.
    drive() polls every 1s; threads whose wake_time has passed flip to 'pending'.
    """
    def schedule(self, thread: WorkflowThread) -> None: ...
    def due_threads(self, now: float) -> list[WorkflowThread]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledThreadRegistry": ...
```

**Why a separate scheduler vs. just a per-instance asyncio task:**
- Per-instance `asyncio.sleep(3600)` dies on process restart.
- A JSON-persisted registry survives restart.
- When the server restarts, `WorkflowManager.from_dict()` reads the registry, sets up post-wake events, and resumes wherever wait time has elapsed.

### §5.12 Existing-primitive composition map (the unification table)

This table is the v3 north star — what each new concept reuses vs. introduces:

| v3 concept | Reuses | Introduces (new code) | Why |
|---|---|---|---|
| `WorkflowInstance` | `WorkflowContext`, `StateGraphTracker.to_dict()` | Dataclass shape | Snapshot only; tracker rebuilt at activation |
| `WorkflowRuntime` | `StateGraphTracker.start/complete/get_available_next`, `asyncio.Event`, `asyncio.Semaphore`, `asyncio.gather` (WorkGraph pattern) | ~150 LoC driver | Replaces hand-wave "LLM updates `completed_phases`" with explicit code path |
| `PhaseAwaiter` | `asyncio.Event` | ~25 LoC | One-shot signal; same pattern as `WebUIInteractive._response_queue` |
| `__branch__` execution | `asyncio.gather + Semaphore` (workgraph.py:2676 pattern), `StateGraphTracker.state_outputs` | Branch-spawning helper (~30 LoC) | Identical concurrency model to existing WorkGraph |
| `__goto__ __wait__` | `asyncio.sleep` for short, `ScheduledThreadRegistry` JSON for long | ~60 LoC registry | JSON persistence is the only restart-safe option |
| `complete_phase` tool | Existing `ConversationToolType` enum + `_handle_conversation_tool` dispatch | One enum entry + one branch (~30 LoC) | Smallest possible client-server bridge |
| `WorkflowManager` lifecycle | `WorkflowRuntime` (new), `WorkflowRegistry` (new) | ~120 LoC | Same conceptual role as Claude §2.5 |

**Net new code: ~415 LoC across 3 new files. Zero modifications to RichPythonUtils.**

### §5.13 Rejected alternative — `WorkGraph` as the runtime substrate (NEW v3.2)

A natural question on first inspection: *"Why don't we just make SOP execution use `WorkGraph` directly? It already has async parallel execution, dynamic subgraph expansion (`subgraph_registry`), result-save-and-resume, and a loop subsystem. Why build `WorkflowRuntime` at all?"*

The answer is principled and worth recording so future readers don't re-raise the question. Four reasons, in priority order:

| # | Why `WorkGraph` is the wrong substrate for SOP | Evidence |
|---|---|---|
| 1 | **WorkGraph nodes are *callables*; SOP phases are *prompt-context-waiting-for-external-signal*.** A `WorkGraphNode` wraps `value: Callable` and the framework invokes it with merged upstream inputs when the node becomes reachable. An SOP phase is not a function to call — it is a prompt context whose "execution" is the conversational inferencer's chitchat over many turns, terminated by an LLM-issued `complete_phase(**artifacts)` call. Fitting "wait for inferencer signal" into a callable means writing `node.value = lambda *a, **kw: asyncio.run(phase_awaiter.wait())` — the same `PhaseAwaiter` pattern (§5.9.1) wrapped in a callable shim with zero added value. | `worknode_base.py:183` (`WorkNodeBase`), `workgraph.py:34` (`WorkGraphNode(Node, WorkNodeBase)`) — both require `value: Callable`. |
| 2 | **WorkGraph owns its own event loop; `ConversationalInferencer` already owns one.** `WorkGraph._arun()` blocks on `asyncio.gather` over child nodes. The conversational inferencer's `run_conversation` (lines 907+ of `conversational_inferencer.py`) is *also* the top-level coroutine for a chat session. Wrapping the inferencer inside `WorkGraph._arun()` would invert control flow — now WorkGraph is top-level and the inferencer becomes a sub-callable. That is a major refactor of the inferencer's lifecycle for unclear gain. | `workgraph.py:2670–2680` (`asyncio.gather`+`asyncio.Semaphore` in `_arun`); `conversational_inferencer.py:907+` (top-level `run_conversation`). |
| 3 | **WorkGraph's expansion model is "parent spawns subgraph and waits for it to complete" (hierarchical nesting); SOP's branch model is "parent emits list → N peer phases fan out → all converge to one downstream phase" (sibling fan-out).** Forcing one into the other requires either (a) a synthetic "branch parent" node that doesn't exist in the SOP, or (b) treating each branch as a one-node subgraph whose only purpose is to host one phase. Both are accidental complexity. | `workgraph.py:1829` (`subgraph_registry`); `workgraph.py:2045-2048` (factory-based expansion). |
| 4 | **The single thing we want from WorkGraph (parallel async execution for `__branch__`) is `asyncio.gather + Semaphore` — a Python stdlib pattern, not a WorkGraph invention.** v3 already uses this pattern directly in `WorkflowRuntime._spawn_branches_for` (§5.11.2) without inheriting WorkGraph's structural constraints. We get the parallelism without the inversion of control. | Compare `workgraph.py:2670–2680` to §5.11.2's branch code — same primitives, simpler topology. |

**Patterns we DO steal from WorkGraph (composition, not inheritance):**

| Pattern | Where in v3 | Why borrowed | Why not inherited |
|---|---|---|---|
| `asyncio.gather + Semaphore` for fan-out | §5.11.2 `_spawn_branches_for` | Exact same primitive WorkGraph uses for parallel DAG execution | Python stdlib; no need to inherit from `WorkGraph` to use it |
| `_expansion_depth` / `_max_expansion_depth` cap pattern (`worknode_base.py:203–204`) | §5.9.2 new knobs: `max_branch_fan_out`, `max_total_active_phases`, `max_concurrent_threads` | Prevents runaway recursion when LLM emits 10,000 branch items or chains 1,000 `__goto__` threads | Same cap-pattern, applied to *our* failure modes (branch fan-out, total active phases, concurrent threads) — not to graph-expansion depth which doesn't exist in SOP |
| `subgraph_registry` factory-based dynamic expansion | **Reserved for v4**, NOT v3.2. If we ever introduce `__include__ another_workflow` semantics, this is the right pattern to copy. | Same problem shape ("dynamically materialize a subgraph at runtime") | Premature for v3 — SOP doesn't have nested-workflow semantics yet; would solve a problem we don't have |

### §5.14 Why no new loop grammar (the WorkGraph loop investigation conclusion, NEW v3.2)

A second natural question: *"WorkGraph has 4 dedicated loop test files (`test_forced_loop_mode.py`, `test_loop_resume.py`, `test_name_based_loop_counts.py`, `test_self_loop_expansion.py`). Should SOP grammar expose any of these loop primitives?"*

Empirical investigation of the WorkGraph loop subsystem reveals it is **`NextNodesSelector(include_self=True)`-driven** (a return-value pattern from the executing callable, not declarative attributes on the node). The 4 test files exercise:

| Test file | What it exercises | Use case | SOP-author relevance |
|---|---|---|---|
| `test_forced_loop_mode.py` | Sequential `___seqN` result naming when `max_expansion_events > 0` | Large-scale distributed expansion with deterministic naming | ❌ Internal bookkeeping — SOP authors never see result naming |
| `test_loop_resume.py` | Checkpoint-save/resume across loop iterations | Process N items in loop; crash at 600; resume at 601 | 🟡 The *concept* matters; the *mechanism* (pickle checkpoints, loop_counts dict) is too low-level for markdown |
| `test_name_based_loop_counts.py` | Name-keyed iteration counts (vs index-keyed) for migration after graph expansion | Step identity across graph mutations | ❌ Pure data-structure detail; SOP authors don't care |
| `test_self_loop_expansion.py` | Node returns `GraphExpansionResult` on iter 1 (expands graph), then loops over expanded nodes | Dynamic workflow generation: "generate sub-tasks from input, then loop over them" | ❌ Requires code-defined `GraphExpansionResult` return value — no markdown equivalent |

**Conclusion:** The only loop pattern SOP authors plausibly want to express is "repeat phase X while condition Y" — which `__goto__ Phase X __if__ Y` + `max_goto_iterations` (§4.7, §5.11.1) **already covers**. The remaining WorkGraph loop primitives are code-only mechanics for code-authored DAGs; exposing them in markdown would create a second DSL (markdown-as-programming-language) with no SOP-author benefit.

**Decision:** **v3.2 adds NO new loop grammar.** The §4.2 EBNF stays as is. If you find yourself wanting to write `__loop_until__` or `__loop_max_iters__`, that's a signal to express the same intent with `__goto__ ... __if__` instead — the runtime semantics are equivalent and the AST is simpler.

---

## §6. Move `task` tool to AgentFoundation (Claude §3, adopted)

### §6.1 Verdict
Movable with **thin bridge shim** in OpenStartup. Two OpenStartup deps stay: `bootstrap.ensure_siblings_on_path` + `tool_cli.run_cli`.

### §6.2 New layout
```
AgentFoundation/src/agent_foundation/
├── common/workspace/allocator.py       # NEW (Claude §3.1) — generalized
└── resources/tools/task/
    ├── __init__.py
    ├── tool.json                        # moved + extended (Claude §3.4)
    ├── executor.py                      # moved
    ├── topologies/                      # moved
    │   └── *.yaml
    └── cli.py                           # NEW AF-side CLI (uses ui from §8)

OpenStartup/src/openteam/server/resources/tools/task/
├── __init__.py
├── cli.py                               # KEPT — thin shim
└── _shared/workspace_allocator.py       # KEPT as re-export shim only
```

### §6.3 OpenStartup bridge shim
```python
# openteam/server/resources/tools/task/cli.py (post-move)
from openteam.bootstrap import ensure_siblings_on_path
ensure_siblings_on_path()
from agent_foundation.resources.tools.task.executor import execute as _execute
from openteam.server.services.tool_cli import run_cli

def main():
    run_cli(execute_fn=_execute, tool_name="task")
```

### §6.4 Workspace allocator location (improved per Claude)
**v1 placed allocator under `agent_foundation/resources/tools/_shared/`**; Claude correctly notes that's still tool-locked. v2 places it at **`agent_foundation/common/workspace/allocator.py`** — it is a framework primitive, not a tool subordinate.

OpenStartup's `_shared/workspace_allocator.py` becomes a 2-line re-export shim:
```python
from agent_foundation.common.workspace.allocator import *  # back-compat
```

### §6.5 Dependent tool migrations
After the allocator moves, three OpenStartup tools also import it: `create_role/executor.py`, `role_setup/executor.py`, `project_onboarding/executor.py`. Their imports update to `from agent_foundation.common.workspace.allocator import ...` (Claude §3 phase 7). The re-export shim makes this update *non-blocking* — it can land later.

---

## §7. New `sop` tool (Claude §4, adopted)

### §7.1 What it does
Peer to `task`: given a workflow id or SOP file, spin up a `ConversationalInferencer`, enter the workflow, drive it to completion. Yolo or non-yolo mode.

### §7.2 CLI surface
```bash
af-sop run --workflow-id code_optimization --target-path /repo
af-sop run --sop-file ./my.md --target-path /repo
af-sop run --workflow-id code_optimization --target-path /repo --yolo
af-sop resume --instance-id <id>
af-sop list
af-sop inspect ./my.md     # parse + show AST + lint
```

### §7.3 Implementation surface (Claude §4)
```
agent_foundation/
├── resources/tools/sop/
│   ├── __init__.py
│   ├── tool.json
│   ├── executor.py
│   ├── cli.py
│   └── lint.py
└── common/sop/registry.py    # SOPRegistry — thin wrapper over WorkflowRegistry
```

### §7.4 9-stage executor pipeline (Claude §4.2, adopted)
1. Parse arguments (workflow name, yolo flag, model override, params JSON).
2. Resolve workflow via `WorkflowRegistry.get_definition(workflow_name)`.
3. Parse SOP via `SOPManager.load(sop_path)`; extract `tool_to_phase_map`.
4. Yolo setup: if yolo, set `yolo_mode=True` on inferencer.
5. Allocate workspace via `allocate_tool_workspace("sop", base_dir=...)`.
6. Build `ConversationalInferencer` (tool registry filtered to SOP-referenced tools; prompt renderer; prior_context with workflow params; interactive: `RichTerminalInteractive` or provided).
7. Create `WorkflowInstance` via `WorkflowManager.enter_workflow()`.
8. Run agentic loop — seed with initial message, loop until all phases complete.
9. Return `ToolExecutionResult` with workspace artifacts and phase summary.

### §7.5 Yolo-mode mechanics in `ConversationalInferencer` (Claude §4.3, adopted)
```python
yolo_mode: bool = attrib(default=False, kw_only=True)

# New methods
def _gate_requires_user(self, tool: ConversationTool) -> bool:
    """True if the tool's phase has [__must__] marking — user must intervene."""

def _yolo_auto_resolve(self, tool: ConversationTool) -> dict:
    """Default response: CONFIRMATION→yes, CLARIFICATION→yolo_vars lookup,
       SINGLE_CHOICE→first option, MULTIPLE_CHOICE→empty."""

# In _handle_conversation_tool():
if self.yolo_mode and not self._gate_requires_user(tool):
    return self._yolo_auto_resolve(tool)
```

**Non-yolo:** uses `RichTerminalInteractive` (Workstream 8) or whatever `InteractiveBase` the caller provides.

### §7.6 `SOPRegistry` (thin wrapper for tool use)
```python
class SOPRegistry:
    def __init__(self, extra_dirs: list[Path] | None = None): ...
    def get(self, name: str) -> SOPDefinition | None: ...
    def list_available(self) -> list[str]: ...

@dataclass
class SOPDefinition:
    name: str
    sop_path: str
    description: str
    required_tools: list[str]
```

---

## §8. CLI UI module — extending `RichInteractiveBase` (Claude §5, adopted)

### §8.1 Module structure
```
agent_foundation/ui/cli/
├── __init__.py                    # lazy imports (no eager Rich import)
├── console.py                     # shared Console singleton + get_console()
├── theme.py                       # ThemeManager (colors, symbols, Rich Theme)
├── rich_terminal_interactive.py   # RichTerminalInteractive(RichInteractiveBase)
├── components/
│   ├── __init__.py
│   ├── phase_progress.py          # PhaseProgressDisplay
│   ├── confirmation_gate.py       # ConfirmationGate
│   ├── choice_selector.py         # Single/MultipleChoiceSelector
│   ├── streaming_panel.py         # StreamingPanel (Rich.Live wrapper)
│   ├── tool_feedback.py           # ToolExecutionDisplay
│   ├── workflow_overview.py       # WorkflowOverviewPanel
│   └── input_prompt.py            # RichInputPrompt (prompt_toolkit)
└── sop_runner_ui.py               # SOPRunnerUI facade
```

### §8.2 Why `RichTerminalInteractive` extends `RichInteractiveBase` (not `TerminalInteractive`)
`RichInteractiveBase` already supports structured `InputMode` dispatch. Extending it gains `SINGLE_CHOICE` / `MULTIPLE_CHOICE` / `FREE_TEXT` routing for free. Extending `TerminalInteractive` would require re-implementing that dispatch.

### §8.3 ConversationToolType → CLI component mapping (Claude §5.5, adopted verbatim)

| `ConversationToolType` | `InputMode` | CLI Component |
|---|---|---|
| `CLARIFICATION` | `FREE_TEXT` | `RichInputPrompt` |
| `SINGLE_CHOICE` | `SINGLE_CHOICE` | `SingleChoiceSelector` |
| `MULTIPLE_CHOICE` | `MULTIPLE_CHOICE` | `MultipleChoiceSelector` |
| `CONFIRMATION` | `metadata.widget_type="confirmation"` | `ConfirmationGate` |
| `TOOL_ARGUMENT_FORM` | compound metadata | sequential field prompts |

### §8.4 `RichTerminalInteractive` key methods (Claude §5.2)
```python
class RichTerminalInteractive(RichInteractiveBase):
    def _send_response(self, ...):
        # Dispatches on self._current_input_mode:
        # widget_type="confirmation" -> ConfirmationGate
        # SINGLE_CHOICE -> SingleChoiceSelector
        # MULTIPLE_CHOICE -> MultipleChoiceSelector
        # default -> Rich Markdown Panel

    def _get_input(self, ...):
        # Dispatches on self._pending_input_mode

    def stream_token_batches(self, ...):
        # Wraps StreamingPanel for token-by-token Rich.Live rendering
```

### §8.5 Component sketches (Claude §5.3, summarized)
- **PhaseProgressDisplay** — phase status with icons (done/running/pending/failed); live updates from `prior_context["completed_phases"]` + SOP AST.
- **ConfirmationGate** — Rich Panel + `[Y/N/A/V]` keybindings; `V` opens detail view.
- **StreamingPanel** — `Rich.Live` wrapper; spinner during thinking; token accumulation; Markdown render. Pattern from rankevolve's `ChatDisplay.stream_assistant_response()`.
- **SingleChoice / MultipleChoiceSelector** — numbered list with keyboard input; returns `{"choice_index": int}` / `{"selections": [...]}` matching `RichInteractiveBase` expected shapes.
- **SOPRunnerUI** — top-level facade composing all components; creates `RichTerminalInteractive`, initializes phase progress, provides phase-transition callbacks.

### §8.6 Dependencies (Claude §5.4)
- **Required:** `rich >= 13.0`, `prompt_toolkit >= 3.0`
- **NOT required:** `textual` — too heavy; owns the event loop.

Add as an optional extra: `agent_foundation[ui]`. Core agent imports do not pull `ui/cli/`.

---

## §9. Phased rollout

Each phase is independently mergeable; later phases depend on earlier ones for testability but not for compilation.

| Phase | Scope | Risk | New tests | Person-days |
|---|---|---|---|---|
| **0** | RED-first parser tests for v2 grammar (10 tests) | very low | +10 | 1 |
| **1** | SOP parser v2 extension in RichPythonUtils (`SOPManager` + `StateNode` fields + regex + two-pass) | low | +15 | 3 |
| **2** | `render_for_mode("yolo")` + 6 edge-case tests | low | +6 | 1 |
| **3** | `WorkflowDefinition` / `Instance` / `Thread` / `Registry` / `Manager` skeleton + serialization | medium | +14 | 4 |
| **4** | 4 lifecycle tools (enter/exit/resume/complete) + `initial.jinja2` template additions + `ConversationalInferencer.workflow_manager` integration | medium | +10 | 4 |
| **5** | `WorkflowContext` migration shim + back-compat tests | medium | +6 | 2 |
| **6** | Allocator move to `common/workspace/` + re-export shims + 3 dependent-tool import updates | medium | +4 | 2 |
| **7** | `task` tool move + OpenStartup bridge shim | medium | (regression suite) | 3 |
| **8** | `sop` tool + executor + `SOPRegistry` + `af-sop` CLI + yolo-mode mechanics | medium | +12 | 4 |
| **9** | `ui/cli/` module + `RichTerminalInteractive` + 7 components + `SOPRunnerUI` facade | medium | +18 | 5 |
| **10** | End-to-end SOP execution in both yolo and non-yolo modes; migrate `model_optimization.md` to v2 format (round-trip test) | low | +6 | 2 |

**Total: ~31 person-days, ~101 new tests, ~14 files touched in AF, ~3 in OpenStartup, ~2 in RichPythonUtils.**

### §9.1 Phase ordering rationale
- Phases 0–2 are pure RichPythonUtils — can ship as a standalone library release.
- Phase 3 is the structural core; Phase 4 wires it into the inferencer; Phase 5 protects existing sessions.
- Phases 6–7 are the tool relocation — independent of workflow framework.
- Phases 8–9 are the new tooling that depends on everything before.
- Phase 10 is the integration acceptance gate.

### §9.2 Mergeability — what can land independently
- Phases 0–2 form one PR (RichPythonUtils v2 parser).
- Phases 3–5 form a second PR (workflow framework).
- Phases 6–7 form a third PR (task move).
- Phases 8–9 form a fourth PR (sop tool + UI).
- Phase 10 is the acceptance gate, not a code PR.

---

## §10. Risks register

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | Two-pass parser changes break OLD inline format | 🔴 high | low | RED tests for OLD format pin parity; back-compat is acceptance-gate criterion |
| 2 | `WorkflowContext` migration loses session state | 🔴 high | medium | `DeprecationWarning` + side-by-side load test; 1-release coexistence window |
| 3 | `RichInteractiveBase` semantics shift unexpectedly when extended | 🟡 medium | low | Inherit + override only `_send_response` / `_get_input`; do NOT modify base |
| 4 | `__branch__` execution diverges from user mental model (parallel vs sequential) | 🟡 medium | medium | v1 defaults `max_parallel_tasks=1`; user opts in; runtime semantics documented in `WorkflowThread` docstring |
| 5 | `RichInputPrompt` (prompt_toolkit) collides with `Rich.Live` event loop | 🟡 medium | medium | Wrap input in `console.capture()` block; pause Live before prompt; reference rankevolve pattern |
| 6 | Yolo-mode filter strips marker inside code block / link / paragraph | 🟡 medium | high | 6 explicit edge-case tests; renderer separate from parser |
| 7 | `task` tool move breaks 47+ existing OpenStartup tests | 🟡 medium | medium | Phase 7 bridge shim is the back-compat surface; regression suite must pass before merge |
| 8 | `WorkflowDefinition` discovery picks up archived/template SOPs | 🟢 low | medium | Convention: prefix archive dirs with `_`; registry skips |
| 9 | `__goto__` thread creation introduces unbounded thread growth on bad SOP | 🟡 medium | low | `af-sop inspect` lint warns on cycles; runtime caps active threads per instance (configurable, default 8) |
| 10 | `rich` / `prompt_toolkit` become required everywhere via transitive import | 🟡 medium | medium | `ui/cli/` lazy-imports; `__init__` exposes nothing eagerly; optional extra `agent_foundation[ui]` |
| 11 | `WorkflowManager` non-thread-safe under WebSocket concurrent updates | 🟡 medium | medium | Single-instance per session; mutations gated by `asyncio.Lock` per manager (added in §5.5) |
| 12 | Two SOP storage layouts (flat vs directory) confuse contributors | 🟢 low | low | README in `workflow_sop/` documents both; flat preferred; directory only when supplementary files needed |
| 13 (v3) | `PhaseAwaiter` orphaned if inferencer crashes between `start(phase)` and `complete_phase` call | 🟡 medium | medium | `_activate_and_wait` `try/finally` removes awaiter on exception; `WorkflowRuntime.drive()` catches awaiter timeout → marks phase failed → instance can be resumed |
| 14 (v3) | `asyncio.Semaphore` from one event loop reused across loops (e.g., suspend/resume) breaks | 🟡 medium | low | Semaphore created in `__post_init__` of fresh runtime per drive session; never persisted across loops |
| 15 (v3) | Branch fan-out floods rate limits (N=20 concurrent LLM calls) | 🟡 medium | medium | `max_parallel_branches` defaults to 1; documented as per-SOP knob; doc warns about provider rate caps |
| 16 (v3) | LLM-emitted `complete_phase` artifacts don't match the schema the next phase expects | 🟡 medium | high | Per-phase output schema (optional) parsed from SOP front-matter; `runtime.signal_phase_complete` validates against schema; returns helpful tool-error if invalid |
| 17 (v3.1) | SOP author writes `__goto__ Phase X __if__ count > 5` or `__if__ X and Y` expecting Python-expression evaluation | 🟡 medium | high | `_CONDITION_VALID_RE` rejects anything beyond `var` (truthy) or `var == "val"` (equality) at parse time; `validate_condition()` raises `SOPGrammarError` with line number + remediation hint. `af-sop inspect` runs this validation eagerly. Documentation explicitly lists supported forms. Migration path: if Python-expression support is genuinely needed (v4), introduce a separate `__where__` tag with sandboxed `simpleeval` — do NOT extend `__if__` semantics silently. |
| 18 (v3.1) | SOP author confuses top-of-phase `[__if__ X]` (gate) with `[__goto__ Phase X __if__ Y]` (conditional re-entry) | 🟢 low | medium | Parser disambiguates by position (top-of-phase tag-line uses `_IF_GATE_RE`; clause inside `__goto__` uses the embedded `__if__` group of `_GOTO_RE`). `af-sop inspect` renders the parsed AST with explicit "gate" / "goto-condition" labels so author can verify intent. |

---

## §11. Acceptance criteria

### §11.1 Parser (SOP grammar v2 + v3.1 conditional surfaces)
- ✅ All 6 v2 grammar elements (`__goto__`, `__afterwards__`, `__wait__`, `__branch__`, `__requires confirmation__` separate-line, unknown-tag survival) round-trip parse correctly.
- ✅ OLD `model_optimization.md` parses identically before/after Phase 1.
- ✅ NEW `code_optimization.md` parses to expected AST shape.
- ✅ Migrating `model_optimization.md` to v2 format produces equivalent AST.
- ✅ `render_for_mode("yolo")` strips correctly on all 6 edge cases (test fixture per case).
- ✅ `af-sop inspect` exits 0 on both old and new SOPs; reports tags + tree.

**v3.1 RED tests (conditional goto / gate — pin behavior before parser changes land):**
- ✅ **`test_goto_with_truthy_condition`:** `[__goto__ Phase 3 __if__ needs_iteration]` parses to `goto_target="Phase 3"`, `goto_condition_var="needs_iteration"`, `goto_condition_value=None`. Runtime test: with `state_outputs={"needs_iteration": True}`, `get_available_next()` re-enables Phase 3 after parent completes; with `{"needs_iteration": False}` or absent, Phase 3 stays completed.
- ✅ **`test_goto_with_equality_condition`:** `[__goto__ Phase 3 __if__ status == "retry"]` parses to `goto_target="Phase 3"`, `goto_condition_var="status"`, `goto_condition_value="retry"`. Runtime: only re-enables Phase 3 when `state_outputs["status"] == "retry"`.
- ✅ **`test_top_of_phase_if_gate`:** `## Phase 4\n[__if__ fix_applied]` parses to `gate_var="fix_applied"`, `gate_value=None`. Runtime: Phase 4 is excluded from `get_available_next()` until `bool(state_outputs["fix_applied"])` is True.
- ✅ **`test_combined_goto_if_afterwards_wait`:** `[__goto__ Phase 3 __if__ Y == "z" __afterwards__ __wait__ 1h]` parses ALL 5 fields correctly in one pass (no order sensitivity within the bracket).
- ✅ **`test_goto_max_iterations_runtime_guard`:** SOP with `__goto__ Phase X __if__ retry` and `state_outputs["retry"]=True` permanently — after `max_goto_iterations` (default 10) re-entries, the goto silently no-ops and `get_available_next()` returns `[]`. SOP frontmatter override `max_goto_iterations: 20` extends to 20.
- ✅ **`test_condition_complexity_rejected`:** Parsing `[__goto__ Phase 3 __if__ count > 5]` or `[__if__ X and Y]` or `[__if__ not X]` raises `SOPGrammarError` with line number and remediation hint pointing to `af-sop inspect`. Confirms we don't silently mis-interpret unsupported expressions.

### §11.2 Workflow runtime
- ✅ `WorkflowManager.enter_workflow → exit_workflow → resume_workflow → complete_workflow` lifecycle traceable in test.
- ✅ Two concurrent `WorkflowInstance`s coexist; switching focus updates prompt sections accordingly.
- ✅ `WorkflowManager.to_dict()` / `from_dict()` round-trip preserves instance state, thread state, and tracker snapshot.
- ✅ Old session JSON with top-level `workflow_context` loads with `DeprecationWarning` and produces equivalent runtime behavior.

### §11.2.1 PhaseAwaiter / WorkflowRuntime contract (NEW v3)
- ✅ `WorkflowRuntime.drive()` blocks on `PhaseAwaiter.wait()` until the inferencer calls `complete_phase`; verified via test that mocks the inferencer's `complete_phase` after a 100ms delay.
- ✅ `runtime.signal_phase_complete(<inactive_id>)` raises clear `RuntimeError`; LLM gets it as tool error and self-corrects in the next turn.
- ✅ `__goto__` directive spawns `WorkflowThread`; tracker re-resolves dependencies; thread runs after parent phase completes (no wait).
- ✅ `__goto__ __wait__ 1h` persists thread to `ScheduledThreadRegistry`; if process restarts, thread resumes once wake time has elapsed.
- ✅ `__branch__` on N-item list spawns N parallel `_activate_and_wait` calls via `asyncio.gather`; with `max_parallel_branches=1` they run sequentially; with `=N` they run concurrently.
- ✅ `BranchSourceMissing` raised cleanly when branch source variable is absent or non-list in parent outputs.
- ✅ Branch result aggregation into `state_outputs[downstream.id]` as a list `[{branch_item, outputs}, ...]` — verified by round-trip test.
- ✅ Runtime suspends and resumes cleanly: `manager.exit_workflow(instance_id)` snapshots tracker via `tracker.to_dict()`, drops live awaiters, marks instance `suspended`; subsequent `resume_workflow` re-creates runtime and re-arms scheduler.
- ✅ No modifications to `RichPythonUtils/stategraph.py` or `RichPythonUtils/sop_manager.py` (verified by checksum diff in CI).

### §11.3 Tool relocation
- ✅ `task` tool runs identically from new AF location with no test regressions in OpenStartup.
- ✅ OpenStartup `task/cli.py` shim continues to work; `tool_cli.run_cli` finds it.
- ✅ All 3 dependent tools (`create_role`, `role_setup`, `project_onboarding`) work both with old `_shared` import path AND new `common.workspace.allocator` import path during the deprecation window.

### §11.4 SOP tool
- ✅ `af-sop run --workflow-id code_optimization --target-path /tmp/repo` in yolo mode completes without prompting and emits workspace artifacts.
- ✅ Same command in non-yolo mode prompts user via `RichTerminalInteractive` for each `[__must__]` gate.
- ✅ `af-sop resume --instance-id <id>` continues from suspended phase with state intact.
- ✅ `af-sop inspect <file>` lints unknown tags, missing `branch_source_var`, `__goto__` cycles.

### §11.5 CLI UI
- ✅ `RichTerminalInteractive` correctly dispatches all 5 `ConversationToolType` cases per §8.3 table.
- ✅ Streaming token rendering does not corrupt when interrupted by phase transition.
- ✅ `prompt_toolkit` input does not visually conflict with `Rich.Live` panels (tested with `pytest-asyncio` + tty fixture).

---

## §12. Open questions for the user

1. **Branch concurrency default.** v2 plan defaults `max_parallel_tasks=1` (sequential). Confirm or set higher default?
2. **`workflow_id` derivation.** v2 uses filename stem (`code_optimization.md` → `code_optimization`). Sufficient, or allow YAML frontmatter `workflow_id` override?
3. **Per-session multiple instance limit.** v2 has no cap — user can `enter_workflow` indefinitely. Add a soft cap (e.g., 5) with override flag?
4. **`__wait__` semantics under server restart.** If wake_time is in the past after restart, run immediately or skip? v2 plan: run immediately. Confirm?
5. **`agent_foundation[ui]` extras-vs-default.** v2 plan makes `ui/cli/` an optional extra. Confirm that's the right tradeoff vs. making it a default dep?
6. **(v3.2) Future `__include__ another_workflow` semantics.** Should a v4 SOP grammar allow one workflow to include another (e.g., `[__include__ shared_review_phases]` to inline a reusable phase block)? If yes, this is *exactly* the problem `WorkGraph.subgraph_registry` (`workgraph.py:1829`) was designed to solve — and the right pattern to copy. If no, we keep SOPs flat. **v3.2 explicitly defers this** but pre-registers the design pointer so a future implementer knows where to look. Confirm defer-to-v4, or escalate to v3?

---

## §13. Honest comparison with input plans

| Aspect | Rovo Dev v1 | Claude plan | INTEGRATED v2 |
|---|---|---|---|
| Acknowledges existing `SOPManager` | ❌ no | ✅ yes | ✅ yes (corrected) |
| Acknowledges existing `RichInteractiveBase` | ❌ no | ✅ yes | ✅ yes (corrected) |
| Acknowledges existing `WorkflowContext` | ⚠️ partial | ✅ yes | ✅ yes |
| EBNF grammar | ✅ yes | ❌ no | ✅ yes (kept from v1) |
| Scope rules | ✅ yes | ❌ no | ✅ yes (kept from v1) |
| `__goto__` / `__branch__` semantics formalized | ✅ yes | ⚠️ partial | ✅ yes (v1 prose + Claude runtime contract) |
| Yolo edge cases | ✅ 6 explicit | ⚠️ implicit | ✅ yes (6 from v1) |
| Runtime contract for thread spawning | ❌ no | ✅ yes | ✅ yes (Claude) |
| ConvToolType → CLI mapping table | ❌ no | ✅ yes | ✅ yes (Claude) |
| Workspace allocator location | ❌ tool-locked | ✅ framework-primitive | ✅ Claude location |
| Risk register | ✅ 10 | ⚠️ 2 (verification plan) | ✅ 12 (v1 + 2 new) |
| Acceptance criteria | ✅ ~25 | ⚠️ ~6 | ✅ ~30 |
| Open questions | ✅ 5 | ❌ 0 | ✅ 5 |
| Empirical grounding | ❌ 2 wrong claims | ✅ 7 verified pointers | ✅ all verified |

**If forced to pick exactly one of the two source plans: Claude's plan.** It is empirically grounded; mine started from a wrong baseline in two of four workstreams. But v2 is strictly better than either: Claude's architecture + Claude's verified pointers + my grammar formalization + my operational discipline (risks, ACs, open questions, non-goals).

---

*End of plan v3. Reviewers: please challenge §4.2 (EBNF correctness), §5.8 (back-compat migration), §5.9 (PhaseAwaiter + WorkflowRuntime architecture), §5.11.2 (branch parallelism design), §6.4 (allocator location), §8.6 (extras-vs-default), §11.2.1 (runtime ACs), and §12 (open questions) most carefully.*

### §14. v3 architectural summary (the elevator pitch)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Existing primitives reused (zero modifications)                        │
│                                                                          │
│    SOPManager.load() ──► SOP (IS-A StateGraph)                          │
│                              │                                          │
│                              ▼                                          │
│                       StateGraphTracker (sync ledger)                   │
│                              │                                          │
│                              ▼                                          │
│                       state_outputs / completed_states                  │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  v3 additions (~415 LoC; 3 new files in agent_foundation/common/workflow)│
│                                                                          │
│   WorkflowRuntime (async driver, owns tracker, asyncio.gather pattern)  │
│         │                                                                │
│         ├──► PhaseAwaiter (one-shot asyncio.Event)  ◄──┐                 │
│         │                                                │                │
│         └──► ScheduledThreadRegistry (JSON persistence) │                 │
│                                                          │                │
│   ConversationalInferencer.workflow_runtime ──► signal_phase_complete() │
│                                  via                                     │
│                       complete_phase conversation tool                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**The architectural unification, in one sentence:** *`SOP` already IS a `StateGraph`; `StateGraphTracker.complete()` already IS the "phase done with outputs" call; we add a thin async client-server adapter (`PhaseAwaiter` + `WorkflowRuntime`) that wraps these in `asyncio.Event` + `asyncio.gather` (the same pattern `WorkGraph` already uses for parallelism), and expose one new conversation tool (`complete_phase`) so the inferencer can signal completion. No fork. No parallel reinvention. ~415 LoC of new code; zero modifications to RichPythonUtils.*

