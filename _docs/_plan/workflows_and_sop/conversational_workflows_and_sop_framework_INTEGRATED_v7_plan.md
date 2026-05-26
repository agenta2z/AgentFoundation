# Conversational Workflows + Enhanced SOP Framework — Integrated Plan v7

**Author:** Tony Chen
**Date drafted:** 2026-05-24
**Status:** Draft v7 — execution-ready; historical audit trail consolidated into §0; load-bearing safety rules kept at point-of-use with back-references; pending review

---

## §0. Change history & lessons learned

> *Why this section exists:* The plan evolved through 7 iterations as Claude's plan and mine cross-pollinated. Earlier iterations had real architectural bugs that took multiple rounds to catch. To keep the rest of the document focused on *current execution intent* (not "v3 said X, v4 fixed Y, v5 changed Z"), all version-comparison narrative lives here. Each load-bearing rule that survived into the current spec has a one-line **DO NOT** marker at point-of-use that back-references this section.

### §0.1 Version-by-version evolution

| Version | Substrate | LoC | Verdict | What changed | What stayed |
|---|---|---|---|---|---|
| v1 (Rovo Dev) | Bespoke `SOPRunner` | unspec | ❌ Wrong substrate; missed SOP parser exists | First-pass plan; misclassified `SOPManager` as non-existent | — |
| Claude v1 | `SOPWorkGraphNode(WorkGraphNode)` | ~200 | ✅ Substrate right | Discovered existing SOP parser; cited BTA's `breakdown_node = WorkGraphNode(wrapping ConversationalFlowNodeAdapter)` precedent (`breakdown_then_aggregate_inferencer.py:1226, 1307, 1667, 1862`) | Substrate insight → v4 |
| v2 (Rovo Dev) | New `WorkflowRuntime` | ~415 | ❌ Wrong substrate (rejected WorkGraph) | EBNF grammar formalized | Grammar → v4 |
| v3 (Rovo Dev) | New `WorkflowRuntime` + `PhaseAwaiter` | ~415 | ❌ Elegant but wrong substrate; reinvented WorkGraph machinery | Added bridge tools, async driver | Most discarded in v4 |
| v3.1 (Rovo Dev) | Same as v3 | ~415 | ✅ Grammar-side correct | `__if__` maps to existing `StateNode.goto_condition_var`/`gate_var` — no new fields needed (`stategraph.py:33-60, 262`) | Grammar mapping → v4 §4.4 |
| v3.2 (Rovo Dev) | Same as v3 + rejected-alt analysis | ~415 | ❌ Rejection arguments were 3-of-4 strawmen | Added "Why not WorkGraph" decision table | Deleted in v4 |
| v4 (Rovo Dev) | `SOPWorkGraphNode(WorkGraphNode)` | ~200 | ✅ Substrate right; ❌ branch convergence bespoke | Substrate reversal: WorkGraph IS the right runtime substrate. Per-sibling counter for branch convergence | Substrate → v5 |
| v5 (Rovo Dev) | `SOPWorkGraphNode` + `BranchBarrierNode` | ~250 | ❌ Critical integration bug: `_run` override was dead code | Adopted Claude's `BranchBarrierNode` pattern (cleaner than per-sibling counter); added `is_branch_leaf` flag, explicit §5.9 long-running-async treatment; fixed Claude's silent `phases[0]` fallback to explicit `InvalidSOPError`; added `branch: bool` field | Architecture → v6 |
| v6 (Rovo Dev) | Same as v5 with bug fix | ~250 | ✅ First runnable version | Fixed `_run` override → `value=self._execute_phase` (ActionNode pattern at `action_node.py:229`); same fix for `BranchBarrierNode` | All → v7 |
| **v7 (this plan)** | Same as v6 | ~250 | **✅ Execution-ready; history consolidated** | **No spec change.** Reorganization: all version-comparison narrative moved into this §0; load-bearing rules kept at point-of-use with concise `DO NOT` markers and §0 back-references; fixed stale per-sibling-counter code remnants in §5.5; deleted §5.8 (v3.2 deletion meta-narrative) | — |
| **v7.0.1** (audit, 2026-05-24 21:12) | Same | ~250 | **✅ Convergence verified — no new integration warranted** | **No content change.** Re-read Claude's plan (still v5 status; updated 20:24 with 9-line cosmetic delta). Verified v7 already contains 100% of Claude's substantive content (architecture, BranchBarrierNode, `value=callable` pattern, `complete_phase`/`abort_phase` tools, 15-row risk register, `__if__` mapping). v7 retains 4 things Claude lacks: consolidated audit history, in-place DO NOT markers, strict multi-`[__initial__]` enforcement (no silent `phases[0]` fallback), `_build_branch_siblings` stale-helper fix. | — |
| **v7.1** (audit, 2026-05-24 22:06) | Same + `goto_condition_negate`, `gate_negate` fields | ~252 | **✅ Spec gaps closed** | Applied 6 of 7 audit findings from external reviewer (1 rejected as over-engineering): **F1** added `!=` runtime path with 2 new StateNode fields + 2-line polarity inversion in `_check_condition`/`_check_gate`; **F2** §6.5 tool table expanded from 4 → 6 (added `confirm_action`, `abort_phase` with explicit scoping column); **F3** §5.1 ASCII diagram corrected (`SOPWorkGraphNode._run()` → `SOPWorkGraphNode.value(*upstream) = _execute_phase`); **F4** count fixes in §4.7/§12/§14 (3 → 6 new StateNode fields); **F6** §9 CLI UI expanded with §9.6 tool→component mapping + §9.7 streaming surface + §9.8 async wrapper pattern; **F7** §4.6 clarified combined `__goto__ __afterwards__ __if__` condition timing. **Rejected: F5 `complete_workflow` tool** (over-engineering — natural WorkGraph drain + manual `exit_workflow` is sufficient; matches user-stated "workflow is stateful, can re-enter" semantics with no terminal "completed and unrecoverable" state). | — |

### §0.2 Lessons learned (load-bearing for future plans)

These three lessons emerged from the v3→v6 evolution and are repeated here so future implementers / plan-authors don't re-introduce the bugs:

**Lesson 1 — Verify dispatch sites by reading the actual implementation.** Two of v6's three predecessor errors (v3.2's "WorkGraph wrong substrate" rejection and v5's `_run`-override dead-code) were made by reasoning about WorkGraph from class signatures / docstrings instead of reading `workgraph.py:1215-1437` (sync `_run`) and `workgraph.py:1394-1395` (async `_arun`'s `await async_execute_with_retry(func=self.value, ...)`). **Rule:** Architectural claims about how a piece of code dispatches execution must cite line numbers from the actual dispatch site.

**Lesson 2 — `WorkGraphNode` extension uses `value=callable`, not method override.** WorkGraph's `_arun()` always dispatches through `self.value`. Subclasses set `self.value = self._my_execute_method` in `__attrs_post_init__` (canonical example: `ActionNode` at `action_node.py:229`). An `async def _run` override on a `WorkGraphNode` subclass is dead code. **The §5.5 and §5.3 code blocks enforce this; do NOT regress.**

**Lesson 3 — Backward `__goto__` is forward expansion, not graph cycle.** When `Phase 3b __goto__ Phase 3` fires, `StateGraphTracker.complete("3b")` re-enables Phase 3 via `truly_completed.discard("3")`. The Phase 3b node creates a NEW `Phase3_iter_1` node via `GraphExpansionResult`. The WorkGraph remains a DAG; the loop semantics live in the tracker. **Bounded by `max_goto_iterations` (default 10) at `stategraph.py:127, 163-172`.**

### §0.3 Bugs caught and fixed (per-version detail)

**v3.2 → v4 — Substrate rejection was empirically wrong.** v3.2 claimed "WorkGraph wrong substrate" on 4 grounds. Three were empirically wrong:
- "WorkGraph nodes are callables; SOP phases are prompt-context-waiting" → BTA already wraps `ConversationalFlowNodeAdapter` (a multi-turn agentic inferencer) in `WorkGraphNode` at 4 production sites (BTA file lines 1226, 1307, 1667, 1862).
- "WorkGraph owns the event loop; ConversationalInferencer also does" → Not an inversion; the inferencer runs *inside* the node via `await inferencer.ainfer()`. SOP entry point IS the WorkGraph runtime.
- "We only need `asyncio.gather + Semaphore`" → WorkGraph also provides multi-parent Queue merge, checkpoint/resume, expansion-record persistence, graph reporter, `max_expansion_depth`/`max_total_nodes` safety, `Terminate`/`AbstainResult` propagation (~415 LoC if reinvented).

The fourth row (hierarchical-vs-sibling expansion) was a real concern, solved in v5 via `BranchBarrierNode` (~50 LoC). v4 reversed the verdict and adopted WorkGraph as substrate.

**v4 → v5 — Branch convergence was bespoke.** v4 tracked branch convergence via per-sibling counter + "last sibling creates convergence node" semantics under `tracker_lock`. v5 replaced this with `BranchBarrierNode` + `is_branch_leaf=True` flag: branch leaves never call `tracker.complete()`; the barrier does, once. Uses WorkGraph's existing `_merge_upstream_inputs` Queue (zero race window). Strictly cleaner; mirrors BTA's workers→aggregator pattern. v5 also added explicit long-running-async treatment (§5.9) and fixed Claude's silent `phases[0]` fallback to an explicit `InvalidSOPError`.

**v5 → v6 — `_run` override was dead code.** v5's `SOPWorkGraphNode` defined `async def _run(self, *upstream_results)` as a method override. WorkGraph dispatches via `self.value`, never invokes a user `_run` override. Had v5 shipped: `AttributeError: 'NoneType' object has no attribute '__call__'` on first phase execution. Fix: `value=self._execute_phase` in `__attrs_post_init__` (ActionNode pattern). Same fix applied to `BranchBarrierNode` (`value=self._barrier_aggregate`).

**v6 → v7 — Stale per-sibling-counter remnants in §5.5.** While the §5.3 branch design was updated to barrier pattern in v5, the §5.5 `SOPWorkGraphNode._build_successor_expansion` body still referenced `self.branch_convergence_phase` and `_pending_branch_siblings()` (v4 per-sibling fields that no longer exist on the class). v7 deletes those stale lines.

### §0.4 Where load-bearing rules from this section live in the current spec

| Lesson | Enforced at | Marker |
|---|---|---|
| Lesson 1 (verify dispatch sites) | Implementation guideline (whole plan) | This §0 |
| Lesson 2 (`value=callable`, not `_run` override) | §5.5 `SOPWorkGraphNode.__attrs_post_init__`, §5.3 `BranchBarrierNode.__attrs_post_init__` | `# DO NOT override _run/_arun — see §0.2 Lesson 2` |
| Lesson 3 (backward `__goto__` is forward expansion) | §4.6 goto semantics, §5.7 mapping table | `# DO NOT add graph cycles — see §0.2 Lesson 3` |
| `[__initial__]` required (no `phases[0]` fallback) | §5.2 + AC §11.1 | `raise InvalidSOPError(...)` |
| Branch leaves NEVER call `tracker.complete()` | §5.3 + §5.5 `is_branch_leaf` short-circuit | `# DO NOT complete tracker in branch leaves — barrier owns convergence` |

---

> **Note:** v6's §0 contained a separate "Round (timestamp) | What changed | Honest verdict" table with one row per version. That table has been folded into the §0.1 evolution table above (same content, denser layout). Per-round narrative now lives in §0.3.

---

## §1. Why this plan exists

Four related threads must be designed together:

1. **SOP format v2** — markdown-friendly tags on separate lines; new orchestration directives (`__goto__`, `__afterwards__`, `__wait__`, `__branch__`, `__if__`); formal `[__requires confirmation__]` instruction tag with yolo-mode filtering.
2. **First-class workflows** — workflows become a peer to tools and skills: enter/exit/re-enter, stateful, identified by `workflow_id`, multiple concurrent instances per session.
3. **Move `task` tool** — from `OpenStartup` to `AgentFoundation` (it is framework-level, not server-specific).
4. **New `sop` tool** — peer to `task`, runs an SOP end-to-end in yolo/non-yolo mode. Non-yolo mode requires CLI UI components → `agent_foundation/ui/cli/`.

The architectural pivot: **the SOP stops being just text in a prompt variable and becomes a parsed, structured, stateful object that the runtime actively manages.** Everything else follows.

The substrate question — *what runs the SOP at runtime?* — was the central open issue across plan iterations. v4 resolves it: **WorkGraph (RichPythonUtils) is the runtime substrate. StateGraph is the static blueprint. SOPWorkGraphNode bridges them.** This is the production-proven pattern (BTA).

---

## §2. Verified empirical baseline (re-verified 2026-05-24)

| Area | Actual state | Source |
|---|---|---|
| **SOP parsing** | `SOPManager` in RichPythonUtils parses markdown via regex into `SOPPhase`/`SOP` AST (485 lines). Supports `depends_on`, `goto`, `for_each`, `gate_var`, phase outputs, subsections. `StateGraphTracker` (269 lines) evaluates the state machine — non-executing. | `RichPythonUtils/.../sop_manager.py`, `stategraph.py` |
| **StateGraph as static blueprint** | `class SOP(StateGraph)` at `sop_manager.py:144`. StateNode fields: `id`, `depends_on`, `outputs`, `gate_var`/`gate_value`, `goto_target`/`goto_condition_var`/`goto_condition_value`, `foreach_*`. `StateGraphTracker._check_condition()` (`stategraph.py:262`) already supports truthy + string-equality conditions. `goto_counts` bounded by `max_goto_iterations` (`:127`). All in production. | `stategraph.py:33-260` |
| **WorkGraph as runtime** | `WorkGraph(DAG, WorkNodeBase)` — full DAG executor with concurrent fan-out (`asyncio.gather`), multi-parent merging (`_merge_upstream_inputs:297`), `GraphExpansionResult`/`SubgraphSpec` for dynamic subgraph attachment, per-node checkpoint/resume, `max_concurrency` semaphore, `max_expansion_depth`/`max_total_nodes` safety, graph reporter integration. ~2600 lines. Used by BTA, PTI, ActionGraph. | `RichPythonUtils/.../workgraph.py`, `worknode_base.py` |
| **The production precedent** | BTA at `breakdown_then_aggregate_inferencer.py:33` imports `WorkGraphNode`. Lines 1226, 1307, 1667, 1862 — **4 instantiation sites** of `breakdown_node = WorkGraphNode(...)` where the wrapped callable runs `await breakdown_inferencer.ainfer(...)` and the inferencer can be a `ConversationalFlowNodeAdapter` (test_conversational_in_bta_breakdown.py:1, 70, 94). **This is the exact pattern SOP needs.** | BTA production code + real-integration tests |
| **ConversationalFlowNodeAdapter** | Wraps a `ConversationalInferencer` to act as a long-running inferencer in BTA/PTI/LWI worker slots. Inherits `TemplatedInferencerBase`. Provides: checkpoint/resume (`flow_node_adapter.py:221, 229, 247, 330`), timeout via `asyncio.wait_for` (line ~391), fallback inferencer, per-turn checkpoint save, dynamic_context persistence. **NOT a WorkGraphNode itself** — it is the callable hosted by a WorkGraphNode. | `flow_node_adapter.py:72-450` |
| **SOP in prompt** | `ConversationalInferencer._render_prompt()` (lines 619-689) loads SOP, builds `StateGraphTracker`, calls `SOPManager.render_guidance()` for next-step guidance text. Tools NOT filtered by available phases; guidance is advisory. | `conversational_inferencer.py:619-689` |
| **WorkflowContext (existing)** | Tracks `current_phase`, `phase_status`, `completed_phases`, `phase_outputs`, `task_queue`. Lifecycle methods: `start_phase()`, `complete_phase()`, `fail_phase()`. Single workflow per session — no enter/exit/suspend/resume. | `workflow_context.py:25-501` |
| **Tools registry** | Full code-level `tools/registry.py`: `load_all_tools()`, `load_tools_by_type()`, `get_bridge_tools()`. `ToolDefinition` dataclass with JSON serialization. | `tools/registry.py` |
| **Skills registry** | Full code-level `skills/registry.py`: `load_all_skills()`, `format_all_skills()`. | `skills/registry.py` |
| **UI framework** | `InteractiveBase` protocol, `RichInteractiveBase`, `TerminalInteractive`, `QueueInteractive`. `InputModeConfig` with `InputMode` enum (FREE_TEXT, SINGLE_CHOICE, MULTIPLE_CHOICE, PRESS_TO_CONTINUE, EXACT_STRING). Web/Dash/terminal transports. **No Rich-CLI components yet.** | `ui/interactive_base.py`, `ui/rich_interactive_base.py`, `ui/input_modes.py` |
| **`task` tool** | Executor (611 lines) in OpenStartup. `tool.json` metadata-only in AgentFoundation. Only OpenStartup-specific import: `workspace_allocator`. Topologies (10 YAMLs) in OpenStartup. | `openteam/.../task/executor.py` |
| **Yolo mode** | Boolean on `RovoDevCliInferencer` — affects subprocess invocation, not prompt-side filtering. No existing mechanism in `ConversationalInferencer`. | `rovodev_cli_inferencer.py` |
| **Graph reporter** | `WebSocketGraphReporter` bridges multi-node graph execution to WebSocket for per-node streaming, status, topology events. `NodeStreamInteractive` tags tokens with node_id. Used by BTA for per-worker streaming. | `graph_interactive_adapter.py` |

### §2.1 The substrate decision (the heart of v4)

| Aspect | StateGraph (SOP) | WorkGraph | Decision |
|---|---|---|---|
| **What it is** | Static state-machine blueprint with declarative gates/gotos/branches/foreach | Dynamic DAG executor with concurrent fan-out, expansion, checkpointing | **Both are needed — as two layers** |
| **Executes nodes?** | ❌ No (tracker is passive) | ✅ Yes (callables with await semantics) | WorkGraph runs; StateGraph evaluates |
| **Concurrent fan-out?** | N/A | ✅ `asyncio.gather` + `max_concurrency` semaphore | WorkGraph gives `__branch__` for free |
| **Multi-parent merge?** | N/A | ✅ Queue-based via `_merge_upstream_inputs` | WorkGraph gives `__depends on__ A, B` for free |
| **Dynamic node creation?** | N/A | ✅ `GraphExpansionResult` + `SubgraphSpec` | WorkGraph lets SOPWorkGraphNode emit successors on completion |
| **Backward goto?** | ✅ `goto_target` + tracker re-enable | ✅ Forward expansion to a NEW phase-iteration node | Tracker re-enables; SOPWorkGraphNode creates `Phase3_iter1` node. **Graph stays DAG, semantics stays backward-goto.** |
| **Conditional firing?** | ✅ `goto_condition_var` / `gate_var` / `_check_condition()` | N/A | StateGraphTracker evaluates; phase only appears in `get_available_next()` if condition holds |
| **Checkpoint/resume?** | N/A | ✅ Per-node files + `_reconstruct_graph_expansions()` | WorkGraph reconstructs running graph on resume; each node's internal state (per `ConversationalFlowNodeAdapter` checkpoint pattern) is per-node |
| **Long-blocking awaits?** | N/A | ✅ Proven in production via BTA's `breakdown_node = WorkGraphNode(wrapping ConversationalFlowNodeAdapter)` | The exact SOP execution shape is already running in BTA |
| **Graph reporter / streaming?** | N/A | ✅ Per-node status/streaming events | WorkGraph gives observability for free |
| **Bounded recursion?** | ✅ `max_goto_iterations` | ✅ `max_expansion_depth` + `max_total_nodes` | Both layers compose for runaway protection |

**Substrate verdict:** **WorkGraph is the right runtime substrate.** v3.2's `WorkflowRuntime` would have reinvented WorkGraph's concurrency, checkpoint, expansion-record persistence, graph reporter integration, and safety limits — ~415 LoC of bespoke code. The v4 architecture is **~200 LoC of `SOPWorkGraphNode` + factory** that compose existing primitives. This is principled (not a hack): the substrate has been carrying multi-turn agentic inferencers in production via BTA for months.

---

## §3. Goals and non-goals

### Goals
- Define a parseable, extensible **SOP grammar v2** with formalized tag scoping; backward-compatible with v1.
- Make **`Workflow` a first-class runtime concept**: enter/exit/re-enter, `workflow_id`, multi-instance per session.
- **Decouple the inferencer prompt from any single workflow.** Prompt always shows: available workflows + ongoing instances + focused-instance rich context.
- **Adopt WorkGraph as the SOP runtime substrate** via `SOPWorkGraphNode(WorkGraphNode)` — reuse, don't reinvent.
- Move `task` tool to AgentFoundation; leave thin shim in OpenStartup.
- Add `sop` tool with yolo/non-yolo modes.
- Build `agent_foundation/ui/cli/` as a Rich-based CLI UI toolkit extending existing `InteractiveBase`.
- Migrate `model_optimization.md` to v2 format.

### Non-goals
- Not rewriting `WorkflowContext` from scratch — extend it; preserve serialization compatibility (one-release `DeprecationWarning` bridge for old shape).
- Not building a custom event loop or async driver — WorkGraph is the substrate.
- Not modifying RichPythonUtils — all SOP/Workflow code is composition in AgentFoundation.
- Not building a full TUI framework (no Textual dependency).
- Not migrating other OpenStartup tools (only `task`).
- Not adding database persistence — in-memory + session JSON suffice for v1.
- Not designing distributed/multi-process workflow execution.
- Not touching skills system (already has registry; workflows are a new peer concept).
- Not exposing WorkGraph's loop-subsystem primitives in SOP grammar — `__goto__ Phase X __if__ Y` + `max_goto_iterations` already covers the only loop pattern SOP authors need.

---

## §4. SOP grammar v2

### §4.1 Design principles

1. **Markdown-friendly first.** Tags on separate lines render cleanly in GitHub/Confluence; inline form preserved for backward compatibility.
2. **Tag scope is unambiguous.** Every tag has exactly one owner (phase header / subsection header / instruction line).
3. **Orchestration is declarative.** `__goto__` / `__branch__` / `__depends on__` / `__if__` describe state-machine intent; the runtime owns execution.
4. **Two tag categories.**
   - **Phase / subsection tags** (drive runtime behavior): `__depends on__`, `__goto__`, `__branch__`, `__for each__`, `__if__`, `__initial__`, `__afterwards__`, `__wait__`.
   - **Instruction tags** (advisory): `__requires confirmation__` — stripped at render-time in yolo mode; AST always retains the marker.
5. **Forward-compatible.** Unknown tags MUST parse and survive round-trip; the runtime emits a warning but does not error.
6. **Backward-compatible.** Old inline format (`## Phase 1 [__depends on__ Phase 0]:`) continues to parse correctly via existing `_PHASE_HEADING_RE`.

### §4.2 EBNF (specification — implementation extends existing regex parser)

```ebnf
sop                = phase+ ;
phase              = phase_header  blank_line*  tag_line*  body ;

phase_header       = "##" ws phase_label inline_tags? ":" eol ;
phase_label        = "Phase" ws phase_id (ws "—" ws phase_title)? ;
phase_id           = identifier ;                  (* e.g., "0", "3b", "review" *)

inline_tags        = "[" tag_content "]" ( ws "[" tag_content "]" )* ;
tag_line           = "[" tag_content "]" ws? trailing_text? eol ;
trailing_text      = any_until_eol ;               (* preserved as body when tag is instruction *)

tag_content        = orchestration_tag | instruction_tag | unknown_tag ;

orchestration_tag  = depends_on_tag | goto_directive | branch_tag
                   | foreach_tag | if_tag | initial_tag ;

depends_on_tag     = "__depends on__"   ws  phase_ref_list ;
phase_ref_list     = "Phase" ws phase_id ( ws? "," ws? "Phase" ws phase_id )* ;

goto_directive     = "__go" ws? "to__"  ws  "Phase" ws phase_id
                     ( ws  "__afterwards__" )?
                     ( ws  "__wait__" ws duration )?
                     ( ws  "__if__" ws condition_expr )? ;

branch_tag         = "__branch__" ( ws "`" identifier "`" )? ;
foreach_tag        = "__for each__" ws identifier  ws  "in"  ws  identifier
                     ( ws "(sequential)" )? ;
if_tag             = "__if__" ws condition_expr ;
initial_tag        = "__initial__" ;

instruction_tag    = "__requires confirmation__" ;

unknown_tag        = "__" identifier "__" ( ws .* )? ;     (* survives round-trip *)

duration           = digit+ ( "s" | "m" | "h" | "d" ) ;
condition_expr     = identifier                              (* truthy check *)
                   | identifier ws "==" ws literal           (* string equality *)
                   | identifier ws "!=" ws literal ;
literal            = identifier | '"' .* '"' | digit+ ;

body               = ( subsection | paragraph | code_block | tag_line | blank_line )* ;

subsection         = "###" ws subsection_label inline_tags? ":" eol  body ;
subsection_label   = identifier ( ws word )* ;     (* "Tools", "Steps", etc. *)
```

**Notes:**
- `phase_id` allows alphanumerics and hyphens; runtime treats `3b` and `Phase 3b` as the same identity.
- `condition_expr` is *intentionally restrictive* — see §10 risk register row "complex conditions."
- Unknown tag form is preserved literally in `SOPPhase.unknown_tags: list[str]` for round-trip safety.

### §4.3 Tag scope — the unambiguous rule

| Tag location | Owner | Applies to |
|---|---|---|
| **Inline in `##` heading** (old format) | The phase | The whole phase |
| **`[` on its own line, immediately after `##` heading** (≤ 3 blank lines between) | The phase | The whole phase |
| **Inline in `###` heading** | The subsection | The subsection only |
| **`[` on its own line, immediately after `###` heading** | The subsection | The subsection only |
| **`[` on its own line within a phase body** | The next non-tag line | Instruction tag only (`__requires confirmation__`); applies to that line/paragraph |
| **`[` inline within prose** | Itself | Instruction tag only |

**Lint rules:**
- Orchestration tag in prose body (not header) → ERROR.
- Orchestration tag in instruction position → ERROR.
- Instruction tag at phase/subsection header → WARNING (suggest moving to specific line).
- Duplicate orchestration tag in same scope → WARNING (last wins, but suspicious).

### §4.4 `__if__` mapping rule (load-bearing)

**Rule:** Parser MUST map `__if__` constructs onto existing `StateNode` fields — `goto_condition_var`/`goto_condition_value` for `__goto__ ... __if__`, and `gate_var`/`gate_value` for `[__if__ ...]` at phase top. **DO NOT add new StateNode fields for `__if__`** — they already exist at `stategraph.py:33-60` and `StateGraphTracker._check_condition()` at `stategraph.py:262` already evaluates truthy + string-equality conditions. (Bug history: §0.3, "v3.1 correction" — earlier drafts proposed redundant new fields.)

### §4.5 Parser changes (regex additions in `sop_manager.py`, two-pass scanner)

```python
# NEW regexes (add to existing _PHASE_HEADING_RE, _DEPENDS_ON_RE, _GOTO_RE, _FOR_EACH_RE, _IF_RE):

_TAG_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)?$", re.MULTILINE)

_GOTO_AFTERWARDS_RE = re.compile(
    r"__go\s*to__\s+Phase\s+(\w+)"
    r"(?:\s+__afterwards__)?"
    r"(?:\s+__wait__\s+(\d+[smhd]))?"
    r"(?:\s+__if__\s+(.+?))?$",
    re.IGNORECASE,
)

_BRANCH_RE = re.compile(r"__branch__(?:\s+`(\w+)`)?", re.IGNORECASE)
_INITIAL_RE = re.compile(r"__initial__", re.IGNORECASE)
_REQUIRES_CONFIRMATION_RE = re.compile(r"\[__requires confirmation__\]", re.IGNORECASE)
```

**Two-pass body scanning** (extends existing logic at `sop_manager.py:287-289`):

1. Extract body between `##` headings.
2. Scan first N lines (where blank lines are allowed but no prose) for `_TAG_LINE_RE` matches.
3. Parse each match for known directive via the regexes above. Unknown directives → `SOPPhase.unknown_tags`.
4. Trailing text after `]` on the same line is preserved as body (used for instruction tags like `[__requires confirmation__] IMPORTANT: ...`).
5. Merge with any inline directives from the heading bracket (deduplicate with warning).
6. Remaining body passes to `_parse_subsections()` unchanged.

Fully backward-compatible: old inline format works untouched; both formats can coexist in the same SOP.

### §4.6 `__goto__` semantics (v3.1 retained — conditional firing acknowledged)

| Form | Maps to (StateNode fields) | Behavior |
|---|---|---|
| `__goto__ Phase X` | `goto_target="X"` | Unconditional re-enable of Phase X on completion of current phase |
| `__goto__ Phase X __if__ var` | `goto_target="X"`, `goto_condition_var="var"` | Re-enable if `state_outputs["var"]` is truthy |
| `__goto__ Phase X __if__ var == "value"` | `goto_target="X"`, `goto_condition_var="var"`, `goto_condition_value="value"` | Re-enable if equality holds |
| `__goto__ Phase X __if__ var != "value"` | `goto_target="X"`, `goto_condition_var="var"`, `goto_condition_value="value"`, `goto_condition_negate=True` (NEW field — see §4.7) | Re-enable if inequality holds |
| `__goto__ Phase X __afterwards__` | `goto_target="X"`, `goto_afterwards=True` (NEW field, see §4.7) | Spawn a parallel thread to Phase X after current phase; current flow continues independently |
| `__goto__ Phase X __afterwards__ __wait__ 1h` | + `goto_wait_duration="1h"` (NEW field) | Same, with sleep before child node executes |
| `__goto__ Phase X __afterwards__ __if__ var` | + `goto_condition_var="var"` | Spawn-time-checked: thread is **only spawned if condition holds at completion of source phase**. Once spawned, thread runs unconditionally (after optional `__wait__`). |
| `[__if__ var == "value"]` at top of phase | `gate_var="var"`, `gate_value="value"` | Gate: phase only runs if condition holds |
| `[__if__ var != "value"]` at top of phase | `gate_var="var"`, `gate_value="value"`, `gate_negate=True` (NEW field — see §4.7) | Gate with inequality: phase only runs if values differ |

**Condition timing for combined modifiers:** When `__goto__` carries both `__afterwards__` and `__if__`, the condition is evaluated **at thread-spawn time** (i.e., at source-phase completion), NOT at thread-wake time. This means `__wait__` does not re-check the condition. Rationale: the simpler timing matches how `goto_counts` already increment at goto-firing time, not at execution time.

**Runtime cycle guard:** `StateGraphTracker.goto_counts[phase_id]` is incremented each time `goto_target` is followed; bounded by `max_goto_iterations` (default 10, configurable per-SOP via frontmatter). Exceeding the bound raises `GotoBoundExceeded`. **Already in production at `stategraph.py:127, 163-172`.**

**Inequality (`!=`) support requires a small runtime extension.** The existing `StateGraphTracker._check_condition()` (`stategraph.py:261-268`) only evaluates truthy + equality. v7.1 adds two new boolean fields — `goto_condition_negate: bool = False` on `StateNode` and `gate_negate: bool = False` on `StateNode` — and one line in `_check_condition()`: `if node.goto_condition_negate: result = not result`. See §4.7 and §11.1 RED test list.

### §4.7 Genuinely-new StateNode fields (v7.1 — six new, all minimal)

Six new fields total (v4 introduced 2; v5 added explicit `branch: bool` + `branch_source_var`; v7.1 adds inequality-support `goto_condition_negate` + `gate_negate`):

```python
@attrs(slots=False, kw_only=True)
class StateNode:
    # ... existing fields ...
    goto_afterwards: bool = attrib(default=False)            # NEW v4
    goto_wait_duration: Optional[str] = attrib(default=None) # NEW v4 (e.g., "1h", "30m")
    branch: bool = attrib(default=False)                     # NEW v5 — explicit branch declaration
    branch_source_var: Optional[str] = attrib(default=None)  # NEW v5 — the `var` in `__branch__ \`var\``
    goto_condition_negate: bool = attrib(default=False)      # NEW v7.1 — `__goto__ X __if__ var != "v"`
    gate_negate: bool = attrib(default=False)                # NEW v7.1 — `[__if__ var != "v"]`
```

One-line addition to `StateGraphTracker._check_condition()` (`stategraph.py:261-268`):

```python
result = self._evaluate_existing_condition(...)             # existing logic
if node.goto_condition_negate:                              # NEW v7.1
    result = not result
return result
```

Symmetric one-liner for `_check_gate()`. **No new runtime semantics — just polarity inversion of an existing boolean result.**

```python
@attrs(slots=False, kw_only=True)
class SOPPhase(StateNode):
    # ... existing fields ...
    requires_confirmation: bool = attrib(default=False)      # NEW v5 — from [__requires confirmation__]
    unknown_tags: list[str] = attrib(factory=list)           # NEW v5 — forward-compat round-trip preservation
```

**Why `branch: bool` AND `branch_source_var`?** The boolean allows declaring a branch *without* a source variable — useful when the LLM is expected to emit items inline (e.g., generated proposals) rather than reading from a named output variable. When both are set, the source var wins. When only `branch=True`, the tracker calls a hook to ask the inferencer for items. This is forward-compatible with v3 SOPs (where `branch_source_var=None` ⇒ `branch=False`).

All other v5 grammar maps onto existing fields. The minimal additions confine the RichPythonUtils change to ~10 lines; everything else is composition in AgentFoundation.

### §4.8 Yolo-mode rendering

Add `SOPManager.render_for_mode(sop: SOP, mode: str = "default") -> str`:
- `mode="yolo"`: Re-render the markdown stripping any line matching `_REQUIRES_CONFIRMATION_RE` and any trailing text on the same line. Apply only to lines containing the literal `[__requires confirmation__]` marker — code blocks, markdown links, and prose mentions of the string are unaffected (the regex requires bracket boundaries).
- `mode="default"`: Return unchanged.

**Filter at render time, not parse time.** The AST always retains the marker for inspection (`SOPPhase.instruction_tags`).

### §4.9 Migration: `model_optimization.md` → v2

Out-of-scope content change. Phase 8 of the implementation plan covers the mechanical migration:
- Convert inline `[__depends on__ Phase 0]` to a separate-line tag.
- Add `[__initial__]` to Phase 0 if not present.
- Insert `[__requires confirmation__]` markers on confirmation-gate sections (if any).
- Run `af-sop inspect` to verify round-trip equivalence.

---

## §5. Two-layer execution architecture — StateGraph blueprint + WorkGraph runtime

This is the heart of v4. The architecture composes two existing primitives instead of inventing a third.

### §5.1 Architectural overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Static blueprint (RichPythonUtils, unchanged except §4.7) │
│                                                                       │
│  SOP markdown ──parse──▶ SOP(StateGraph)                             │
│                          + StateGraphTracker (per-instance)           │
│                                                                       │
│  Evaluates declaratively:                                             │
│   • tracker.get_available_next()  — which phases may run now          │
│   • tracker.get_pending_thread_spawns()  — NEW: deferred __afterwards__│
│   • tracker.get_branch_items(phase_id)  — NEW: list for __branch__   │
│   • tracker._check_condition()  — gates and conditional gotos         │
│                                                                       │
│  Never mutated by the executor; pure data + pure functions.          │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ "what's next?"
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2 — Runtime executor (RichPythonUtils WorkGraph, unmodified)   │
│                                                                       │
│  Initial state: WorkGraph with one or more start_nodes (one per      │
│  [__initial__] phase, see §5.2 correction 1)                          │
│                                                                       │
│  Each node: SOPWorkGraphNode(WorkGraphNode) wrapping a              │
│             ConversationalFlowNodeAdapter (BTA precedent — see §5.3) │
│                                                                       │
│  Execution loop (existing WorkGraph._arun, unchanged):                │
│   1. WorkGraph dispatches via SOPWorkGraphNode.value(*upstream)       │
│      (= _execute_phase callable, ActionNode pattern — see §0.2 L2)    │
│        → inferencer runs multi-turn agentic loop                      │
│        → checkpoints per turn (FlowNodeAdapter pattern)               │
│        → completes when LLM emits `complete_phase(**artifacts)` tool  │
│   2. node.complete() updates StateGraphTracker                        │
│   3. node returns GraphExpansionResult with successor nodes           │
│        → branch: SubgraphSpec with N entry nodes (sibling fan-out)    │
│        → goto-afterwards: deferred child with min_repeat_wait         │
│        → backward-goto: NEW Phase{id}_iter{N} node (DAG-safe)         │
│   4. WorkGraph runs successors with max_concurrency semaphore         │
│   5. Graph reporter streams per-node events to UI                     │
│                                                                       │
│  Free from WorkGraph: concurrency, checkpoint/resume, expansion       │
│  reconstruction, graph reporter, max_expansion_depth/max_total_nodes  │
│  safety, Terminate/AbstainResult stop propagation.                    │
└─────────────────────────────────────────────────────────────────────┘
```

### §5.2 Multi-`[__initial__]` enumeration (load-bearing)

**Rule:** `build_sop_workgraph` MUST enumerate ALL `[__initial__]`-marked phases as `start_nodes`. **DO NOT silently fall back to `phases[0]`** if zero `[__initial__]` markers exist — raise `InvalidSOPError` instead. (Bug history: §0.3 "v4 → v5" — earlier drafts collapsed multi-initial SOPs to single start; would have silently mis-executed SOPs with parallel entry points.)

```python
def build_sop_workgraph(...) -> WorkGraph:
    initial_phases = [p for p in definition.sop.phases if "initial" in p.directives]
    if not initial_phases:
        raise InvalidSOPError(
            f"SOP {definition.id!r} has no [__initial__] phase. "
            f"Mark at least one phase with [__initial__] to designate the entry point."
        )
    start_nodes = [
        SOPWorkGraphNode(phase=p, tracker=tracker, sop=definition.sop, ...)
        for p in initial_phases
    ]
    return WorkGraph(start_nodes=start_nodes, ...)
```

**Lint rule:** `af-sop inspect` fails with clear error if zero `[__initial__]` phases are declared. Multiple are permitted (parallel start). No silent `phases[0]` fallback.

### §5.3 Branch convergence — the `BranchBarrierNode` pattern (load-bearing)

**Problem:** When `Phase 3b __branch__` emits `[p1, p2, p3]` and creates sibling nodes `Phase4_A`, `Phase4_B`, `Phase4_C`, the downstream `Phase 5` (`depends_on: Phase 4`) must wait for **ALL** siblings — and `tracker.complete("4")` must be called exactly once (not three times).

**Rule:** Branch expansion produces N branch-leaf `SOPWorkGraphNode`s (each with `is_branch_leaf=True`) + 1 `BranchBarrierNode` whose `previous` list is all leaves. WorkGraph's existing multi-parent `_merge_upstream_inputs` Queue waits for all leaves before invoking the barrier. **DO NOT call `tracker.complete()` from branch leaves — the barrier owns convergence.** Mirrors BTA's workers→aggregator production pattern. (Bug history: §0.3 "v4 → v5" — earlier drafts used a per-sibling counter; bespoke logic with race-prone "last sibling completes" semantics. The barrier pattern eliminates the race window entirely.)

**Pattern:**

```python
# Inside SOPWorkGraphNode._execute_phase(), branch handling:
if next_phase.branch:
    items = self.tracker.get_branch_items(next_phase.id)

    # 1. Create branch leaves — each runs the phase for ONE item.
    #    Critical: is_branch_leaf=True means the leaf does NOT call
    #    tracker.complete(); only the barrier does.
    branch_leaves = [
        SOPWorkGraphNode(
            phase=next_phase,
            tracker=self.tracker,
            sop=self.sop,
            inferencer_factory=self.inferencer_factory,
            workspace=self.phase_workspace.parent / f"phase_{next_phase.id}_branch_{i}",
            branch_item=item,
            tracker_lock=self.tracker_lock,
            iteration=i,
            is_branch_leaf=True,         # ← NEW v5 — disables self-completion
        )
        for i, item in enumerate(items)
    ]

    # 2. Create the barrier — its `previous` list is ALL leaves.
    #    WorkGraph's existing multi-parent Queue merge waits for ALL parents
    #    before invoking barrier._aggregate_and_complete. No new merge logic.
    barrier = BranchBarrierNode(
        phase_id=next_phase.id,
        tracker=self.tracker,
        tracker_lock=self.tracker_lock,
        sop=self.sop,
        inferencer_factory=self.inferencer_factory,
        workspace=self.phase_workspace.parent / f"phase_{next_phase.id}_barrier",
        previous=branch_leaves,           # WorkGraph wires multi-parent Queue
    )

    return GraphExpansionResult(
        result=result,
        subgraph=SubgraphSpec(
            nodes=branch_leaves + [barrier],
            entry_nodes=branch_leaves,
        ),
        expansion_id=f"sop_branch_{next_phase.id}",
    )
```

**`BranchBarrierNode` (~50 LoC, new file `branch_barrier_node.py`) — v6 ActionNode pattern:**

```python
@attrs(slots=False, kw_only=True)
class BranchBarrierNode(WorkGraphNode):
    """Convergence node for __branch__ — gathers all sibling outputs once.

    Mirrors BTA's aggregator-after-workers pattern.
    Uses WorkGraph's existing multi-parent Queue merge (`_merge_upstream_inputs`)
    to wait for ALL branch leaves. No per-sibling counter, no race conditions.

    v6 CRITICAL: Uses `value=self._barrier_aggregate` (ActionNode pattern at
    action_node.py:229). Does NOT override `_run`/`_arun` — those overrides
    would be dead code because WorkGraph dispatches through self.value.
    """
    phase_id: str
    tracker: StateGraphTracker
    tracker_lock: asyncio.Lock
    sop: SOP
    inferencer_factory: Callable
    workspace: Path
    expected_count: int                          # number of leaves we expect to receive

    def __attrs_post_init__(self) -> None:
        # v6 CRITICAL: wire execution via `value`, matching ActionNode pattern
        self.value = self._barrier_aggregate
        self.name = f"sop_branch_barrier_{self.phase_id}"
        # Barrier output IS the merged phase output — keep it resumable for downstream phases
        self.enable_result_save = StepResultSaveOptions.SaveAndResume

    async def _barrier_aggregate(self, *leaf_results, **_) -> Any:
        """Aggregate all branch leaf outputs into one phase completion.

        WorkGraph's _arun() collects all parent (leaf) outputs via
        _merge_upstream_inputs Queue and passes them as positional args here.
        """
        # `leaf_results` is the tuple of branch-leaf outputs, ordered by Queue arrival.
        merged_outputs = self._merge_leaf_outputs(leaf_results)

        async with self.tracker_lock:
            # tracker.complete called EXACTLY ONCE here, regardless of N leaves.
            self.tracker.complete(self.phase_id, **merged_outputs)
            available = self.tracker.get_available_next()
            pending_threads = self.tracker.get_pending_thread_spawns()

        # Build successor expansion (shared helper with SOPWorkGraphNode).
        next_nodes = self._build_successors(available, pending_threads)
        if not next_nodes:
            return merged_outputs
        return GraphExpansionResult(
            result=merged_outputs,
            subgraph=SubgraphSpec(nodes=next_nodes, entry_nodes=next_nodes),
            expansion_id=f"sop_branch_complete_{self.phase_id}",
        )

    def _merge_leaf_outputs(self, leaf_results: tuple) -> dict:
        """Default merge: collect all leaf outputs into a list keyed by branch_item.

        SOPs that want different merge semantics can subclass; the default
        produces {"branch_results": [leaf_1_out, leaf_2_out, leaf_3_out]}.
        """
        return {"branch_results": list(leaf_results)}
```

**Why this is strictly better than v4's per-sibling-counter approach:**

| Dimension | v4 (per-sibling counter) | v5 (`BranchBarrierNode`) |
|---|---|---|
| New machinery | `branch_siblings: list[str]`, `branch_convergence_phase`, "last sibling completes" logic | `BranchBarrierNode` + `is_branch_leaf` flag |
| Race conditions | Counter increment under `tracker_lock` (correct but bespoke) | None — WorkGraph's Queue merge has zero race window |
| Lines of code | ~50 LoC of bespoke completion logic | ~50 LoC of standard WorkGraphNode subclass |
| Mirrors existing pattern? | No — invents per-leaf coordination | Yes — identical to BTA's workers → aggregator |
| `tracker.complete()` call count for N=3 branch | 3 calls (with race-safe "last one wins") | 1 call (in barrier only) |
| Phase 5 sees how many parents? | 1 (last leaf to complete) | 1 (the barrier) |

**Acceptance criterion S11.2 (revised):** 3-sibling branch with stagger (200ms / 500ms / 900ms delays). Assert:
- Barrier node executes exactly once, after the 900ms leaf completes.
- `tracker.complete("4")` called exactly once.
- Phase 5 sees the barrier (not the leaves) as its single parent.
- Barrier output contains all 3 leaf results.

### §5.4 Confirmation-gate flow (load-bearing)

Confirmation gates require three explicit mechanisms — not "handled inside ConversationalInferencer" hand-wave. (Bug history: §0.3 "v3.2 → v4" — earlier drafts left this unspecified; v4 made it explicit so user-refusal semantics, yolo-mode propagation, and abort handling are unambiguous.)

1. **yolo_mode propagation:** `WorkflowInstance.yolo_mode: bool` (default False). Propagates to inferencer via `template_extra_feed["yolo_mode"]` at SOPWorkGraphNode construction. Inferencer's prompt template reads it for `render_for_mode` selection (yolo strips `[__requires confirmation__]`).

2. **Confirmation tool:** New conversation tool `confirm_action(prompt: str, default: bool = False) -> bool` registered when `yolo_mode=False`. In yolo mode, the tool is registered as a no-op stub returning `True` immediately (so prompts in non-yolo SOPs still parse but never block).

3. **Refusal semantics:** If user refuses (`confirm_action` returns False), the inferencer:
   - Has the option to re-prompt (LLM may rephrase and re-ask once).
   - On second refusal, calls `abort_phase(reason: str)` tool — which raises `PhaseAbortedByUser` from SOPWorkGraphNode.
   - WorkGraph propagates the abort via `AbstainResult`, stopping downstream phases that `depends_on` this phase.
   - Parallel branches NOT downstream continue unaffected.

4. **Bridge tool: `complete_phase(**artifacts)`:** This is the SOP-specific tool registered ONLY when the inferencer is running inside a SOPWorkGraphNode. It validates the artifacts against `SOPPhase.outputs` declaration; on success, sets the inferencer's `pending_completion` flag which `_ainfer` checks each turn. When set, the inferencer exits cleanly and returns the artifacts as the node's result.

### §5.5 SOPWorkGraphNode skeleton (~150 LoC target — v6: ActionNode pattern)

```python
# AgentFoundation/src/agent_foundation/common/workflow/sop_workgraph_node.py

@attrs(slots=False, kw_only=True)
class SOPWorkGraphNode(WorkGraphNode):
    """A WorkGraphNode that runs one SOP phase via a ConversationalInferencer.

    Production precedent:
    * BTA's `breakdown_node = WorkGraphNode(...)` at
      breakdown_then_aggregate_inferencer.py:1226 wrapping ConversationalFlowNodeAdapter.
    * ActionNode's `self.value = self._execute_action` pattern at
      action_node.py:229 — the canonical extension pattern.

    v6 CRITICAL: We do NOT override `_run`/`_arun`. WorkGraph executes via
    `self.value`, dispatched through `async_execute_with_retry` at workgraph.py:1394-1395.
    Setting `value=self._execute_phase` in the constructor is the only correct integration.
    """
    phase: SOPPhase                                        # the phase definition
    sop: SOP                                                # the full SOP (for cross-phase lookups)
    tracker: StateGraphTracker                              # shared per WorkflowInstance
    tracker_lock: asyncio.Lock                              # serializes tracker mutations
    inferencer_factory: Callable[..., ConversationalFlowNodeAdapter]
    workspace: Path                                          # per-node workspace for checkpoints

    branch_item: Any = attrib(default=None)                 # if branch: the single item for this sibling
    is_branch_leaf: bool = attrib(default=False)            # v5: leaves do NOT call tracker.complete; barrier does
    iteration: int = attrib(default=0)                      # for goto-iteration naming
    min_repeat_wait: Optional[float] = attrib(default=None) # seconds; set when goto-afterwards-wait
    yolo_mode: bool = attrib(default=False)

    def __attrs_post_init__(self) -> None:
        # v6 CRITICAL: wire execution via `value`, matching ActionNode pattern (action_node.py:229)
        # WorkGraph's _arun() calls self.value, NOT a _run override.
        self.value = self._execute_phase

        # Generate stable node name (used for graph topology + checkpoints + UI)
        node_name = f"sop_phase_{self.phase.id}_iter{self.iteration}"
        if self.branch_item is not None:
            node_name += f"_branch{hash(self.branch_item) & 0xFFFF:04x}"
        self.name = node_name

        # Skip-resumable: each node's checkpoints are owned by its inner ConversationalFlowNodeAdapter
        # (per-turn JSON in self.workspace). WorkGraph's coarse-grained result save is redundant for SOP.
        self.enable_result_save = StepResultSaveOptions.SkipResumable

    async def _execute_phase(self, *upstream_results, **_) -> Any:
        """Execute one SOP phase end-to-end. INVOKED BY WorkGraph._arun VIA self.value."""
        if self.min_repeat_wait:
            await asyncio.sleep(self.min_repeat_wait)

        # 1. BUILD: Create the inferencer for this phase
        inferencer = self.inferencer_factory(
            phase=self.phase,
            sop=self.sop,
            tracker_snapshot=self.tracker.snapshot_for_phase(self.phase.id),
            branch_item=self.branch_item,
            yolo_mode=self.yolo_mode,
            workspace=self.workspace,
            upstream_results=upstream_results,
        )

        # 2. EXECUTE: Run the multi-turn agentic loop
        # ConversationalFlowNodeAdapter handles checkpoint/resume internally via on_turn_complete.
        try:
            result = await inferencer.run_agentic_loop(
                content=self._build_seed_message(),
                on_turn_complete=self._make_checkpoint_callback(),
            )
        except PhaseAbortedByUser as e:
            return AbstainResult(reason=str(e), phase_id=self.phase.id)

        # 3. RECORD: Branch leaves SKIP tracker.complete — only the barrier does that.
        #    See §5.3 v5 BranchBarrierNode pattern.
        if self.is_branch_leaf:
            return result  # Pass-through to BranchBarrierNode via WorkGraph's multi-parent Queue

        # Normal (non-leaf) path: complete the tracker and build successors
        artifacts = self._extract_declared_outputs(result)
        async with self.tracker_lock:
            self.tracker.complete(self.phase.id, **artifacts)

        return await self._build_successor_expansion(artifacts)

    async def _build_successor_expansion(self, artifacts) -> GraphExpansionResult:
        """Query tracker for next phases; build SOPWorkGraphNodes for them.

        v7 NOTE: No per-sibling convergence logic here — branch convergence is
        owned by BranchBarrierNode (§5.3). Branch leaves return early before
        reaching this method (see is_branch_leaf short-circuit in _execute_phase).
        DO NOT add `branch_convergence_phase` / `_pending_branch_siblings` —
        those were v4 per-sibling-counter artifacts that the v5 barrier pattern
        eliminates. See §0.3 "v6 → v7" for the history.
        """
        async with self.tracker_lock:
            available = self.tracker.get_available_next()
            pending_threads = self.tracker.get_pending_thread_spawns()

        next_nodes = []

        # __branch__ expansion (§5.3): produce N leaves + 1 barrier.
        # Note: handled inline in _execute_phase for the *next* phase, not here.
        # This method only handles non-branch forward expansion.
        for phase in available:
            if phase.branch:
                # Defer to barrier-pattern construction (§5.3 code block).
                next_nodes.extend(self._build_branch_leaves_and_barrier(phase))
            else:
                next_nodes.append(self._create_child_node(phase, iteration=0))

        # __goto__ __afterwards__ deferred thread spawns
        for spawn in pending_threads:
            target_phase = self.sop.phase_by_id(spawn.target_phase)
            iteration = self.tracker.goto_counts.get(target_phase.id, 0)
            node = self._create_child_node(target_phase, iteration=iteration)
            if spawn.wait_duration:
                node.min_repeat_wait = _parse_duration_seconds(spawn.wait_duration)
            next_nodes.append(node)

        if not next_nodes:
            return artifacts  # Terminal phase — no expansion

        return GraphExpansionResult(
            result=artifacts,
            subgraph=SubgraphSpec(nodes=next_nodes, entry_nodes=next_nodes),
            expansion_id=f"sop_phase_{self.phase.id}_iter_{self.iteration}",
            seed={"phase_id": self.phase.id, "tracker": self.tracker.to_dict()},
        )

    def _create_child_node(self, phase, *, iteration=0, branch_item=None, is_branch_leaf=False):
        child_workspace = self.workspace.parent / f"phase_{phase.id}_iter_{iteration}"
        return SOPWorkGraphNode(
            phase=phase, sop=self.sop, tracker=self.tracker, tracker_lock=self.tracker_lock,
            inferencer_factory=self.inferencer_factory, workspace=child_workspace,
            branch_item=branch_item, iteration=iteration, yolo_mode=self.yolo_mode,
            is_branch_leaf=is_branch_leaf,
        )

    def _build_branch_leaves_and_barrier(self, phase: SOPPhase) -> list[WorkGraphNode]:
        """Build N branch leaves + 1 BranchBarrierNode for a __branch__ phase.

        See §5.3 for the full pattern. The leaves carry is_branch_leaf=True so they
        skip self-completion; the BranchBarrierNode owns tracker.complete() exactly
        once after WorkGraph's multi-parent Queue collects all leaf outputs.

        DO NOT add per-sibling convergence fields here — that's v4 logic the v5
        barrier pattern eliminated. See §0.3 "v4 → v5".
        """
        items = self.tracker.get_branch_items(phase.id)
        leaves = [
            self._create_child_node(phase, iteration=i, branch_item=item, is_branch_leaf=True)
            for i, item in enumerate(items)
        ]
        barrier = BranchBarrierNode(
            phase_id=phase.id,
            tracker=self.tracker,
            tracker_lock=self.tracker_lock,
            sop=self.sop,
            inferencer_factory=self.inferencer_factory,
            workspace=self.workspace.parent / f"phase_{phase.id}_barrier",
            previous=leaves,
            expected_count=len(leaves),
        )
        return leaves + [barrier]
```

**Code volume: ~150 LoC** for the node + ~50 LoC for the factory = **~200 LoC total**, plus 6 lines of StateNode field additions in RichPythonUtils. **No new event loop. No new checkpoint format. No new graph reporter. No new safety limits.**

### §5.6 Factory (`build_sop_workgraph`, ~50 LoC)

```python
def build_sop_workgraph(
    definition: WorkflowDefinition,
    inferencer_factory: Callable,
    workspace: Path,
    *,
    yolo_mode: bool = False,
    max_concurrency: int = 1,
    max_expansion_depth: int = 200,
    max_total_nodes: int = 500,
    max_goto_iterations: int = 10,
    graph_reporter: Any = None,
) -> tuple[WorkGraph, StateGraphTracker]:
    """Construct a WorkGraph + StateGraphTracker pair for an SOP definition.

    Returns both because callers need the tracker for snapshots / persistence.
    """
    tracker = StateGraphTracker(
        graph=definition.sop,
        max_goto_iterations=max_goto_iterations,
    )
    tracker_lock = asyncio.Lock()

    initial_phases = [p for p in definition.sop.phases if "initial" in p.directives]
    if not initial_phases:
        raise InvalidSOPError(
            f"SOP {definition.id!r} has no [__initial__] phase. "
            f"Exactly one or more phases must be marked with [__initial__]."
        )

    start_nodes = [
        SOPWorkGraphNode(
            phase=p,
            sop=definition.sop,
            tracker=tracker,
            tracker_lock=tracker_lock,
            inferencer_factory=inferencer_factory,
            workspace=workspace / f"phase_{p.id}_iter_0",
            yolo_mode=yolo_mode,
        )
        for p in initial_phases
    ]

    graph = WorkGraph(
        start_nodes=start_nodes,
        use_async=True,
        max_concurrency=max_concurrency,
        max_expansion_depth=max_expansion_depth,
        max_total_nodes=max_total_nodes,
    )

    if graph_reporter:
        graph.set_graph_event_callback(graph_reporter.on_graph_topology)

    return graph, tracker
```

### §5.7 How every SOP construct maps to WorkGraph (definitive table)

| SOP construct | WorkGraph mechanism | How it works |
|---|---|---|
| **Sequential phases** (0 → 1 → 1b) | Chained `GraphExpansionResult` | Phase 0 node creates Phase 1 node; Phase 1 creates Phase 1b; etc. `_propagate_settings_to_subgraph` propagates `_expansion_depth + 1` to children. |
| **`__branch__`** (parallel fan-out) | `SubgraphSpec` with N entry nodes + branch barrier (§5.3) | Phase 3b creates `Phase4_A`, `Phase4_B`, `Phase4_C` in one SubgraphSpec. WorkGraph's `_arun()` runs them concurrently with `max_concurrency`. Last sibling to complete creates the convergence node. |
| **`__depends on__ A, B`** (multi-parent) | WorkGraph multi-parent Queue merge (`_merge_upstream_inputs:297`) | Node with multiple parents collects all inputs via Queue. Execution deferred until all parents submit results. |
| **Same-phase `__goto__`** (3b → 3b) | `NextNodesSelector(include_self=True)` | Self-loop via WorkGraph's iterative while-loop. Proven WorkGraph pattern. |
| **Backward `__goto__`** (3b → 3) | Tracker re-enable + forward expansion | `tracker.complete("3b")` → `truly_completed.discard("3")` → `get_available_next()` returns Phase 3 → expansion creates `Phase3_iter_1` node. **Graph stays a DAG.** |
| **`__goto__ ... __afterwards__ __wait__ 1h`** | Deferred child with `min_repeat_wait` | Tracker emits `ThreadSpawnRequest`; SOPWorkGraphNode creates child with `min_repeat_wait=3600`. Child sleeps then runs. Parent flow continues independently. |
| **`__if__` gate** (top of phase) | Tracker evaluation, no expansion | `tracker._check_condition(gate_var, gate_value)` evaluates. If false, phase not in `get_available_next()` → no expansion to that phase. |
| **`__goto__ ... __if__`** (conditional goto) | `goto_condition_var`/`goto_condition_value` on StateNode | Same `_check_condition()` evaluator; only re-enables target if condition holds. |
| **`__for each__`** | Similar to branch | Creates N nodes one per collection item. `foreach_sequential` → `max_concurrency=1` for the subgraph. |
| **`[__requires confirmation__]`** | Inferencer-level confirmation tool (§5.4) | yolo_mode propagated via `template_extra_feed`. Tool is a no-op stub in yolo mode; interactive in non-yolo. |
| **`complete_phase(**artifacts)`** | Bridge tool (§5.4) | The LLM signals phase completion via this tool. SOPWorkGraphNode extracts artifacts and returns from `_run()`. |
| **Pause / resume (`exit_workflow`)** | WorkGraph checkpoint + per-node `FlowNodeAdapter` checkpoint | Running nodes save per-turn checkpoints; WorkGraph reconstructs expansion records via `_reconstruct_graph_expansions()`. |
| **Runaway protection** | WorkGraph's `max_expansion_depth` + `max_total_nodes`; tracker's `max_goto_iterations` | Three independent caps compose: graph-depth, total-nodes, and per-phase-goto-count. |

### §5.8 Long-running ConversationalInferencer inside a WorkGraph node (load-bearing FAQ)

> *(Note: v6 had a §5.8 documenting v3.2's substrate-rejection reversal. That meta-narrative now lives in §0.3 "v3.2 → v4". This section was renumbered from v6's §5.9 in v7.)*

WorkGraph was designed for callables that complete in seconds; SOP phases involve multi-turn LLM chats, possibly hours of user interaction. This section documents why the substrate is appropriate so future readers don't re-raise the question (the v3.2 plan re-raised it and rejected the substrate — see §0.3).

| Concern | Resolution | Evidence |
|---|---|---|
| **`async def callable` can block indefinitely?** | Standard Python `await` works for any duration — there's no implicit timeout on a coroutine. WorkGraph awaits the callable with no timeout wrapper by default. | `workgraph.py` `_arun()` simply does `result = await node.value(...)`. |
| **What if we need a timeout?** | Wrap the inferencer call with `asyncio.wait_for(coro, timeout=…)`. `ConversationalFlowNodeAdapter` already does this at `flow_node_adapter.py:391`. | Production code. |
| **No signaling registry for external completion?** | Not needed in our design. The inferencer runs **inside** the node via `await inferencer.ainfer(...)`. The "external signal" is the LLM emitting `complete_phase(**artifacts)` — an in-process tool call, not a cross-node event. The tool sets `inferencer.pending_completion` which `_ainfer` checks each turn. | §5.4. |
| **No mid-node pause/resume across server restart?** | Solved by the `ConversationalFlowNodeAdapter` checkpoint pattern: per-turn checkpoint saved via `on_turn_complete` callback. On server restart, WorkGraph reconstructs the running graph via `_reconstruct_graph_expansions()`; each restarted node loads its last checkpoint and continues from the last completed turn. | `flow_node_adapter.py:221, 229, 247, 330`. |
| **Concurrency cap?** | `max_concurrency` semaphore on the WorkGraph. Default `max_concurrency=1` for SOP (sequential phases), bumped to N when `__branch__` fans out. | `workgraph.py` `_arun()` uses `asyncio.Semaphore(self.max_concurrency)`. |
| **BTA actually does this?** | Yes — at 4 instantiation sites. BTA's `breakdown_node = WorkGraphNode(value=lambda: await breakdown_inferencer.ainfer(...))` where `breakdown_inferencer` is a `ConversationalFlowNodeAdapter` in real integration tests. This is the exact SOP execution shape, already running in production for months. | `breakdown_then_aggregate_inferencer.py:1226, 1307, 1667, 1862`; `test_conversational_in_bta_breakdown.py:1, 70, 94`. |

**Conclusion:** The substrate is appropriate for our use case. No new infrastructure needed for long-blocking awaits.

---

## §6. First-class workflow framework

**New module:** `agent_foundation/common/workflow/`

```
agent_foundation/common/workflow/
├── __init__.py                  # re-exports
├── definition.py                # WorkflowDefinition (parsed SOP + metadata)
├── instance.py                  # WorkflowInstance (per-session runtime state)
├── registry.py                  # WorkflowRegistry (discovers workflows from resource dirs)
├── manager.py                   # WorkflowManager (per-session orchestrator)
├── sop_workgraph_node.py        # SOPWorkGraphNode (§5.5)
└── sop_workgraph.py             # build_sop_workgraph factory (§5.6)
```

### §6.1 `WorkflowDefinition` (definition.py)

```python
@attrs(slots=False, kw_only=True, frozen=True)
class WorkflowDefinition:
    """A discovered, parsed workflow. Immutable after construction."""
    id: str                                  # filename stem (e.g., "code_optimization")
    name: str                                # from frontmatter or filename
    description: str                         # from frontmatter or first paragraph
    source_path: Path                        # filesystem location for diagnostics
    sop: SOP                                 # parsed SOP AST (StateGraph)
    raw_markdown: str                        # full markdown for re-render in different modes
    frontmatter: dict                        # YAML frontmatter (max_goto_iterations, etc.)
    requires_tools: list[str] = attrib(factory=list)  # from frontmatter
    available_modes: list[str] = attrib(factory=lambda: ["default", "yolo"])
```

### §6.2 `WorkflowInstance` (instance.py)

```python
@attrs(slots=False, kw_only=True)
class WorkflowInstance:
    """A live workflow execution. One per `enter_workflow` call.

    Owns the WorkGraph + StateGraphTracker pair. Persists per-session.
    """
    instance_id: str                         # uuid8; stable across resume
    definition_id: str                       # references WorkflowDefinition.id
    status: Literal["active", "suspended", "completed", "aborted"] = attrib(default="active")
    yolo_mode: bool = attrib(default=False)
    workspace: Path                          # per-instance workspace (checkpoints land here)
    created_at: datetime
    last_active_at: datetime

    # Runtime objects — reconstructed on resume from workspace state
    graph: Optional[WorkGraph] = attrib(default=None, init=False)
    tracker: Optional[StateGraphTracker] = attrib(default=None, init=False)
    _graph_task: Optional[asyncio.Task] = attrib(default=None, init=False)

    def to_persistent_dict(self) -> dict:
        """Serializable shape stored in session JSON (graph state lives in workspace files)."""
        return {
            "instance_id": self.instance_id,
            "definition_id": self.definition_id,
            "status": self.status,
            "yolo_mode": self.yolo_mode,
            "workspace": str(self.workspace),
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "tracker_state": self.tracker.to_dict() if self.tracker else None,
        }

    @classmethod
    def from_persistent_dict(cls, data: dict, definition: WorkflowDefinition) -> "WorkflowInstance":
        # Reconstruct tracker from to_dict snapshot; WorkGraph is rebuilt lazily on resume.
        ...
```

### §6.3 `WorkflowRegistry` (registry.py)

Discovers workflow definitions from configured resource directories. Mirrors the existing `tools/registry.py` and `skills/registry.py` patterns.

```python
class WorkflowRegistry:
    """Discovers WorkflowDefinitions from one or more resource directories."""

    def __init__(self, search_paths: list[Path]):
        self._search_paths = search_paths
        self._definitions: dict[str, WorkflowDefinition] = {}

    def load_all(self) -> dict[str, WorkflowDefinition]:
        """Scan all search paths for `*.md` files; parse each as a WorkflowDefinition."""
        for path in self._search_paths:
            for md_file in path.rglob("*.md"):
                try:
                    definition = self._parse_definition(md_file)
                    if definition.id in self._definitions:
                        logger.warning("Duplicate workflow id %r in %s", definition.id, md_file)
                    self._definitions[definition.id] = definition
                except Exception as e:
                    logger.warning("Failed to parse workflow %s: %s", md_file, e)
        return dict(self._definitions)

    def get(self, definition_id: str) -> WorkflowDefinition:
        if definition_id not in self._definitions:
            raise WorkflowNotFound(definition_id)
        return self._definitions[definition_id]

    def list_all(self) -> list[WorkflowDefinition]:
        return list(self._definitions.values())
```

**Default search paths (in order):**
1. `agent_foundation/resources/prompt_templates/conversation/main/_variables/workflow_sop/` (built-ins)
2. `~/.agent_foundation/workflows/` (user-installed)
3. Anything in `AGENT_FOUNDATION_WORKFLOW_PATH` env var (colon-separated)

### §6.4 `WorkflowManager` (manager.py) — per-session orchestrator

```python
@attrs(slots=False, kw_only=True)
class WorkflowManager:
    """Per-session orchestrator for enter/exit/resume/list of workflow instances.

    Held by ConversationalInferencer. Exposes the 4 conversation tools listed in §6.5.
    """
    registry: WorkflowRegistry
    session_workspace: Path
    active_instances: dict[str, WorkflowInstance] = attrib(factory=dict)
    focused_instance_id: Optional[str] = attrib(default=None)
    inferencer_factory: Callable                                # injected by ConversationalInferencer
    graph_reporter: Any = attrib(default=None)

    async def enter_workflow(self, definition_id: str, *, yolo_mode: bool = False) -> str:
        """Create a new WorkflowInstance and begin executing it. Returns instance_id."""
        definition = self.registry.get(definition_id)
        instance = WorkflowInstance(
            instance_id=_new_uuid8(),
            definition_id=definition_id,
            yolo_mode=yolo_mode,
            workspace=self.session_workspace / "workflows" / definition_id / _new_uuid8(),
            created_at=datetime.utcnow(),
            last_active_at=datetime.utcnow(),
        )
        instance.workspace.mkdir(parents=True, exist_ok=True)

        graph, tracker = build_sop_workgraph(
            definition=definition,
            inferencer_factory=self.inferencer_factory,
            workspace=instance.workspace,
            yolo_mode=yolo_mode,
            graph_reporter=self.graph_reporter,
        )
        instance.graph = graph
        instance.tracker = tracker
        instance._graph_task = asyncio.create_task(graph.arun())

        self.active_instances[instance.instance_id] = instance
        self.focused_instance_id = instance.instance_id
        return instance.instance_id

    async def exit_workflow(self, instance_id: Optional[str] = None) -> None:
        """Suspend the focused (or specified) instance. WorkGraph checkpoints; task cancelled."""
        instance = self._resolve_instance(instance_id)
        instance.status = "suspended"
        if instance._graph_task and not instance._graph_task.done():
            instance._graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await instance._graph_task
        if self.focused_instance_id == instance.instance_id:
            self.focused_instance_id = None

    async def resume_workflow(self, instance_id: str) -> None:
        """Resume a suspended instance. WorkGraph reconstructs from checkpoints."""
        instance = self.active_instances.get(instance_id)
        if not instance:
            raise WorkflowInstanceNotFound(instance_id)
        if instance.status != "suspended":
            raise WorkflowInstanceNotResumable(instance_id, instance.status)
        # Rebuild graph from definition + tracker snapshot
        definition = self.registry.get(instance.definition_id)
        graph, tracker = build_sop_workgraph(
            definition=definition,
            inferencer_factory=self.inferencer_factory,
            workspace=instance.workspace,
            yolo_mode=instance.yolo_mode,
            graph_reporter=self.graph_reporter,
        )
        # Restore tracker state from snapshot
        tracker.load_from_dict(instance.tracker.to_dict())
        # WorkGraph reconstructs from per-node checkpoints in workspace
        instance.graph = graph
        instance.tracker = tracker
        instance.status = "active"
        instance._graph_task = asyncio.create_task(graph.arun())
        self.focused_instance_id = instance_id

    def list_instances(self) -> list[WorkflowInstance]:
        return list(self.active_instances.values())

    def get_focused_instance(self) -> Optional[WorkflowInstance]:
        if self.focused_instance_id:
            return self.active_instances.get(self.focused_instance_id)
        return None
```

### §6.5 Six new conversation tools

| Tool | Type | Purpose | Scope |
|---|---|---|---|
| `enter_workflow(workflow_id: str, yolo_mode: bool = False) -> str` | Workflow lifecycle | Returns new `instance_id`; activates and focuses the instance | Always registered (when `WorkflowManager` is attached) |
| `exit_workflow(instance_id: Optional[str] = None) -> None` | Workflow lifecycle | Suspends focused (or specified) instance; checkpoints to disk | Always registered (when at least one instance exists) |
| `resume_workflow(instance_id: str) -> None` | Workflow lifecycle | Re-focuses a suspended instance; rebuilds graph from checkpoints | Always registered (when at least one suspended instance exists) |
| `complete_phase(**artifacts)` | SOP phase signal | Validates artifacts against `SOPPhase.outputs`; signals phase completion | **SOP-scoped:** Registered only when running inside SOPWorkGraphNode (§5.4) |
| `confirm_action(prompt: str, default: bool = False) -> bool` | SOP confirmation gate | Prompts user (non-yolo); returns boolean. **In yolo mode: no-op stub returning `True` immediately** so prompts in non-yolo SOPs still parse but never block. | **SOP-scoped:** Registered only when running inside SOPWorkGraphNode (§5.4) |
| `abort_phase(reason: str)` | SOP escape hatch | Raises `PhaseAbortedByUser` from SOPWorkGraphNode; emits `AbstainResult` (§5.4 step 3) | **SOP-scoped:** Registered only when running inside SOPWorkGraphNode (§5.4) |

These live under `agent_foundation/resources/tools/{enter,exit,resume}_workflow/` (workflow lifecycle, always-on) and `agent_foundation/resources/tools/{complete_phase,confirm_action,abort_phase}/` (SOP-scoped) with `tool.json` + `executor.py` per existing tool convention. The SOP-scoped subset is registered/unregistered around each `SOPWorkGraphNode._execute_phase` invocation; see §5.4.

**No `complete_workflow` tool by design.** A workflow instance reaches `status="completed"` automatically when WorkGraph drains (all phases done; no expansions outstanding). Manual close uses `exit_workflow` followed by archival; instances are not destroyed, only suspended. This matches the user-stated semantics ("workflow is stateful, you can exit half way and re-enter") — there is no terminal "completed and unrecoverable" state.

### §6.6 Prompt rendering — three sections

`conversational_inferencer.py::_render_prompt()` extended to compose three template sections:

| Section | When rendered | Source |
|---|---|---|
| **Available workflows** | Always | `WorkflowRegistry.list_all()` summarized as `id + name + description` (≤ 60 tokens each) |
| **Ongoing workflow instances** | Always (even when none) | `WorkflowManager.list_instances()` summarized as `instance_id + definition_id + status + current_phase` |
| **Focused workflow rich context** | Only when `focused_instance_id` is set | Full SOP rendered via `SOPManager.render_for_mode(sop, mode="yolo" if yolo_mode else "default")` + tracker state for next-step guidance |

**Yolo-mode behavior:** When `yolo_mode=True`, the focused context strips all `[__requires confirmation__]` markers and trailing text.

**Token budget:** ≤ 1500 tokens total for the three sections with 10 workflows + 5 instances + 1 focused (verified by AC §11.3).

---

## §7. Move `task` tool from OpenStartup to AgentFoundation

### §7.1 Why move

`task`'s executor at `openteam/.../task/executor.py` has exactly **one** OpenStartup-specific import: `workspace_allocator`. All other deps are AgentFoundation/RichPythonUtils. It is framework-level, not server-specific. Moving aligns with the new `sop` tool (§8), which is also framework-level.

### §7.2 Mechanical move plan

1. **Move `workspace_allocator` first** to `agent_foundation/common/workspace/allocator.py` (it's a small ~80-line file already neutral of OpenStartup logic).
2. **Move `task/executor.py` + topologies** from `openteam/.../task/` to `agent_foundation/resources/tools/task/`. Topologies (10 YAMLs) move alongside.
3. **Replace OpenStartup's `task/executor.py`** with a thin compatibility shim:
   ```python
   from agent_foundation.resources.tools.task.executor import main as af_task_main
   def main(*args, **kwargs):
       return af_task_main(*args, **kwargs)
   ```
4. **Preserve CLI surface exactly.** All existing OpenStartup task CLI tests must pass unchanged before merge (AC §11.4).
5. **One-release deprecation:** OpenStartup `task` import path emits `DeprecationWarning` pointing at AgentFoundation; remove the shim in the release after.

---

## §8. New `sop` tool

### §8.1 Surface

```bash
af-sop list                              # list discovered workflows
af-sop inspect <file-or-id>              # lint + AST dump
af-sop run --workflow-id <id> [--yolo]   # run end-to-end
af-sop run --file <path> [--yolo]        # run ad-hoc from file
af-sop resume --instance-id <id>         # resume a suspended instance from disk
```

### §8.2 Layout

```
agent_foundation/resources/tools/sop/
├── tool.json
├── executor.py            # CLI entrypoint; wires WorkflowManager + UI
└── lint.py                # static SOP linting (used by `inspect`)
```

### §8.3 Yolo vs non-yolo

- **Yolo:** `WorkflowInstance.yolo_mode=True`; confirm-action tool is no-op stub. Runs without user input.
- **Non-yolo:** Confirm-action tool prompts via `agent_foundation/ui/cli/` (§9). User can refuse → phase aborts per §5.4.

### §8.4 Standalone persistence

Per open question §13.Q1: standalone instance state persists to `~/.agent_foundation/workflows/<instance_id>/`. Each instance directory holds: `instance.json` (the `to_persistent_dict()` blob) + per-node checkpoint files.

---

## §9. CLI UI library — `agent_foundation/ui/cli/`

### §9.1 Why it exists

`af-sop run` (non-yolo) needs interactive prompts: confirmation, single-choice, multi-choice, free-text. The existing `InteractiveBase` / `RichInteractiveBase` infrastructure is the right base — but lacks CLI transports. v4 adds them.

### §9.2 Layout (extension, not replacement)

```
agent_foundation/ui/cli/
├── __init__.py
├── rich_terminal.py       # RichTerminalInteractive(RichInteractiveBase) — Rich-based
├── prompts.py             # high-level helpers: ask_confirm, ask_single_choice, ask_text
├── streaming.py           # token-streaming display (per-node aware via NodeStreamInteractive)
└── theme.py               # Rich theme + color tokens for SOP UI
```

### §9.3 Dependency policy

`agent_foundation[ui]` extras-only. Core agent imports do NOT depend on `rich` or `prompt_toolkit`. Plain `agent_foundation` install remains unchanged.

### §9.4 References (existing patterns to reuse)

| Existing | Purpose |
|---|---|
| `InteractiveBase` protocol | The base interface for all UI transports |
| `RichInteractiveBase` | Adds structured input modes (FREE_TEXT, SINGLE_CHOICE, etc.) |
| `InputModeConfig`, `InputMode`, `ChoiceOption` | The structured-input schema |
| `NodeStreamInteractive` | Per-node tagged streaming (used by BTA, applicable here) |
| `NamespacedGraphReporter` | Per-instance scoping for multi-instance UI |

### §9.5 Reference repos (for design only — NOT dependency)

- `atlassian-packages/rankevolve/src/cli` — Rich-based CLI patterns
- `atlassian_packages/acra-python` — interactive CLI structure

### §9.6 Conversation tool → CLI component mapping (v7.1 — was thin in v7)

`RichTerminalInteractive._send_response(response)` dispatches to a Rich component based on `_current_input_mode` (inherited from `RichInteractiveBase`). Mapping for the 6 conversation tools introduced in §6.5:

| Tool (caller) | Input mode set on `RichInteractiveBase` | Rich component | Async wrapper |
|---|---|---|---|
| `confirm_action(prompt, default)` | `InputMode.CONFIRMATION` | `ConfirmationGate` (Rich `Confirm.ask` styled with theme) | `await rich_terminal.aget_input(InputModeConfig(mode=CONFIRMATION, default=default))` |
| `enter_workflow` (when called via UI helper) | `InputMode.SINGLE_CHOICE` | `RichSelectPrompt` (arrow-key navigation over `ChoiceOption[]` from registry) | `await rich_terminal.aget_input(InputModeConfig(mode=SINGLE_CHOICE, options=workflows))` |
| `resume_workflow` (when called via UI helper) | `InputMode.SINGLE_CHOICE` | `RichSelectPrompt` over suspended-instance list | same |
| `abort_phase(reason)` | `InputMode.FREE_TEXT` | `RichInputPrompt` (multi-line allowed) | `await rich_terminal.aget_input(InputModeConfig(mode=FREE_TEXT, label="reason"))` |
| `complete_phase(**artifacts)` | n/a (LLM-emitted, no user input) | — | — |
| `exit_workflow` | n/a | — | — |

### §9.7 Streaming surface

`RichTerminalInteractive` implements `stream_token_batches(node_id: str, tokens: AsyncIterator[str]) -> None` (the signature expected by `conversational_inferencer.py:159-162`). Implementation:

- Wraps `NodeStreamInteractive` to scope output to the focused workflow instance.
- Emits per-token batches into a `rich.live.Live` panel labelled with `node_id` (= `Phase X (instance abc12345)`).
- Concurrent nodes (from `__branch__` fan-out) render in side-by-side panels via `rich.layout.Layout` split horizontally.
- On panel completion, the panel collapses into a one-line summary preserving artifacts.

### §9.8 `asend_response`/`aget_input` async wrapper pattern

Both methods are thin `asyncio.to_thread(...)` wrappers around the synchronous Rich blocking calls (Rich is not native-async). The wrapper preserves the existing `InteractiveBase` synchronous semantics for backward compatibility while exposing async surface for `ConversationalInferencer.ainfer`. Errors propagate via `RichTerminalUserAbort` → caught at the inferencer boundary → emitted as `abort_phase` tool call when running inside `SOPWorkGraphNode`.

---

## §10. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | `__branch__` source variable type mismatch (not iterable) | MED | Runtime raises `BranchSourceTypeError` with the actual type; lint catches statically when SOP declares output schema |
| 2 | `__branch__` source variable missing from upstream outputs | MED | Runtime raises `BranchSourceMissing` with parent phase's actual outputs; lint catches statically |
| 3 | Backward `__goto__` cycles (Phase 3b → Phase 3 → Phase 3b → Phase 3 → …) | HIGH | `max_goto_iterations` (default 10) raises `GotoBoundExceeded`. AC §11.2 covers a 12-iteration test that asserts the bound fires |
| 4 | `__goto__ __afterwards__ __wait__ 1h` creates orphan thread on session crash | MED | Per-thread checkpoint includes `wait_until` timestamp; on resume, expired waits run immediately, others rescheduled |
| 5 | OpenStartup `task` CLI breaks after move | HIGH | Thin wrapper preserves exact CLI surface; full task CLI test suite must pass unchanged before merge |
| 6 | Yolo filter strips non-confirmation lines (false positives) | HIGH | Filter operates ONLY on lines matching `_REQUIRES_CONFIRMATION_RE` (bracket-delimited); 6 unit tests cover edge cases (marker in code block, in markdown link, mid-line, etc.) |
| 7 | Unknown tags break the parser | MED | Per §4.1 principle 5: unknown tags parse + survive round-trip; warning emitted once; RED test pins this |
| 8 | `rich`/`prompt_toolkit` pulled into core package | LOW | Optional extra in pyproject (`agent_foundation[ui]`); core agent code does not import `ui/cli/` |
| 9 | `WorkflowInstance` count grows unbounded with repeated `enter_workflow` | MED | Per-session instance cap (default 16); `InstanceLimitExceeded` error if exceeded |
| 10 | Template feed order: `prior_context` overrides workflow sections | MED | Workflow sections added as explicit keys AFTER `**self.prior_context` spread, matching existing feed merge pattern |
| 11 | ~~Branch convergence race: two siblings observe `pending_siblings == 0` simultaneously~~ **MOOT under v5 BranchBarrierNode design** — barrier uses WorkGraph's existing multi-parent Queue merge which has zero race window. Risk preserved here only for historical traceability of the v4 design. | N/A (v5) | Replaced by `BranchBarrierNode` (§5.3); WorkGraph's `_merge_upstream_inputs` Queue is the proven primitive |
| 12 | Confirmation refusal abort cascade is too aggressive | MED | `AbstainResult` only blocks downstream phases that `depends_on` the aborted phase; parallel branches unaffected. AC test |
| 13 | `complete_phase(**artifacts)` validation rejects valid artifacts due to over-strict schema | MED | Initial validation is loose (presence-only); strict schema enforcement is opt-in via SOP frontmatter `strict_outputs: true` |
| 14 | WorkGraph `max_total_nodes=500` too small for long-running SOPs with many goto iterations | LOW | Configurable per-SOP via frontmatter `max_total_nodes`; logged warning at 80% utilization |
| 15 | Complex condition expressions (boolean composition, regex, etc.) not supported | LOW | §4.2 condition_expr intentionally restrictive (truthy / `==` / `!=`); complex logic stays in inferencer/code, not SOP grammar |

---

## §11. Acceptance criteria

### §11.1 SOP v2 parser
- [ ] All grammar rules from §4.2 covered by passing tests
- [ ] `code_optimization.md` parses without warnings
- [ ] `model_optimization.md` (post-Phase-8 migration) parses and round-trips
- [ ] Unknown tag preserved through parse/serialize cycle
- [ ] Duplicate tag in same scope warns exactly once
- [ ] Both old format (inline) and new format (separate-line) parse correctly from same test suite
- [ ] **NEW v4:** Zero `[__initial__]` markers → `InvalidSOPError` with clear message; lint catches statically
- [ ] **NEW v4:** `__goto__ Phase X __if__ var` maps to existing `goto_condition_var` (verified by tracker behavior test)
- [ ] **NEW v4:** `__goto__ Phase X __if__ var == "value"` maps to existing `goto_condition_value` (verified by tracker test)
- [ ] **NEW v4:** Top-of-phase `[__if__ var == "value"]` maps to existing `gate_var`/`gate_value`

### §11.2 Two-layer execution engine
- [ ] `enter_workflow` produces instance with unique `instance_id` and creates WorkGraph
- [ ] `exit_workflow` checkpoints WorkGraph, sets `status="suspended"`, instance remains in `active_instances`
- [ ] `resume_workflow` reconstructs WorkGraph from checkpoints, re-focuses suspended instance
- [ ] Multiple concurrent instances of same definition coexist with distinct IDs
- [ ] Session round-trip preserves all instances (including phase checkpoints + expansion records)
- [ ] WorkGraph executes phases in dependency order (Phase 0 → expansion creates Phase 1 → etc.)
- [ ] **§5.2 correction:** Multi-`[__initial__]` SOPs spawn parallel start nodes; verified by 2-initial + 1-converge test
- [ ] **§5.3 correction:** 3-sibling branch with stagger (200ms/500ms/900ms delays) — convergence node fires exactly once, after the 900ms sibling completes
- [ ] **§5.3 correction:** Branch convergence node receives all 3 sibling outputs (no missing parents)
- [ ] **§5.4 correction:** Yolo-mode SOP runs end-to-end with no prompts (confirm_action is no-op stub)
- [ ] **§5.4 correction:** Non-yolo SOP pauses on `[__requires confirmation__]`; user refuse twice → `PhaseAbortedByUser` → downstream phases skipped, parallel branches continue
- [ ] **§5.4 correction:** `complete_phase(**artifacts)` validates against `SOPPhase.outputs` (presence-only by default; strict if frontmatter says so)
- [ ] `__goto__ ... __afterwards__` — SOPWorkGraphNode creates deferred child node after phase completes
- [ ] `__goto__ ... __wait__ 1h` — child node has `min_repeat_wait=3600`; verified by mocked-clock test
- [ ] Backward `__goto__` (Phase 3b → Phase 3) — tracker re-enables Phase 3, expansion creates `Phase3_iter_1`. Graph stays DAG.
- [ ] 12-iteration goto loop (Phase A → A → A …) raises `GotoBoundExceeded` at iteration 11
- [ ] Graph reporter receives per-phase node status + streaming events
- [ ] `max_expansion_depth` prevents unbounded graph growth from goto cycles
- [ ] `asyncio.Lock` on StateGraphTracker prevents concurrent mutation during branch fan-out
- [ ] **NEW v4:** SOPWorkGraphNode supports checkpoint/resume — kill mid-phase, restart process, verify resume from last per-turn checkpoint

### §11.3 Prompt rendering
- [ ] "Available workflows" section renders when no workflow focused
- [ ] "Ongoing workflows" lists all suspended/active instances
- [ ] Focused-instance rich context renders ONLY for focused instance
- [ ] Yolo mode strips all `[__requires confirmation__]` lines
- [ ] Token budget: ≤ 1500 tokens with 10 workflows + 5 instances + 1 focused

### §11.4 Task tool relocation
- [ ] All existing OpenStartup task CLI tests pass without modification
- [ ] Topology loading from new location works
- [ ] No circular imports between openteam and agent_foundation
- [ ] OpenStartup shim emits `DeprecationWarning` exactly once per process

### §11.5 SOP tool + UI
- [ ] `af-sop run --workflow-id code_optimization --yolo` runs end-to-end without prompts
- [ ] `af-sop run` in non-yolo pauses on every `[__requires confirmation__]`
- [ ] `af-sop list` discovers workflows from default + custom directories
- [ ] `af-sop inspect <file>` reports lint errors with file:line precision
- [ ] `agent_foundation[ui]` is the only extra needed
- [ ] Core agent install (no `[ui]`) does not import `rich` or `prompt_toolkit`

---

## §12. Phased rollout

| Phase | Scope | Effort | Dependencies | Risk |
|---|---|---|---|---|
| **0** | Write all RED tests (~25 tests covering §11.1 + §11.2 + §11.3 + §11.4 + §11.5) | 2-3 days | None | LOW |
| **1** | RichPythonUtils: add 6 new StateNode fields (§4.7) + 2 new tracker methods (`get_pending_thread_spawns`, `get_branch_items`) + 2-line polarity inversion in `_check_condition`/`_check_gate` for `!=` support | 0.5 day | Phase 0 | LOW |
| **2** | SOP grammar v2 parser extensions (§4.5) in RichPythonUtils sop_manager.py — two-pass scanning, new regexes | 1.5 days | Phase 1 | LOW |
| **3** | SOPWorkGraphNode + factory (§5.5, §5.6) in AgentFoundation | 2 days | Phase 2 | MED |
| **4** | WorkflowDefinition, WorkflowInstance, WorkflowRegistry (§6.1-6.3) | 1.5 days | Phase 3 | LOW |
| **5** | WorkflowManager + 4 conversation tools (§6.4, §6.5) | 2 days | Phase 4 | MED |
| **6** | ConversationalInferencer integration (§6.6 prompt rendering, yolo_mode threading) | 1.5 days | Phase 5 | MED |
| **7** | Move `task` tool + `workspace_allocator` (§7) | 1 day | None (independent) | MED |
| **8** | New `sop` tool (§8) | 1.5 days | Phase 6, Phase 7 | LOW |
| **9** | `agent_foundation/ui/cli/` (§9) | 2 days | Phase 8 | LOW |
| **10** | Migrate `model_optimization.md` to v2 (§4.9) | 0.5 day | Phase 2 | LOW |
| **11** | Integration tests + AC verification | 1.5 days | All above | MED |
| **TOTAL** | | **~18 days** | | |

**Critical path:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 8 → 11. Phases 7, 9, 10 parallelizable.

---

## §13. Open questions

1. **Standalone persistence root.** `~/.agent_foundation/workflows/<instance_id>/` proposed. Confirm vs. `${XDG_DATA_HOME:-~/.local/share}/agent_foundation/...` for XDG-compliance?
2. **SOP linter in CI.** Should `af-sop inspect` block CI? Proposed: yes for SOPs under `resources/`, no for user-supplied.
3. **Multi-workflow concurrent advancement.** If two instances are both `active` and the agent emits actions touching both, which advances? **Proposed spec:** only the **focused** instance advances per turn; others are suspended-by-default after `enter_workflow` if there's an existing focused. Confirm.
4. **Branch convergence phase resolution ambiguity.** §5.3 `_resolve_branch_convergence` picks "the downstream phase that lists this phase in depends_on AND is not branch-internal." If multiple downstreams match, which one is the convergence? **Proposed:** the one whose `depends_on` includes *only* this phase (single-parent); if multiple, lint error.
5. **`agent_foundation[ui]` default install.** Should `[ui]` be in default extras (so `pip install agent_foundation` gets it)? **Proposed:** No — extras-only; keeps core install lean.
6. **`__include__ another_workflow` for v5.** Should v5 SOP grammar support inlining one workflow inside another? If yes, WorkGraph's `subgraph_registry` (`workgraph.py:1829`) is the right substrate. **v4 defers** — pre-registers the design pointer.

---

## §14. File-level change inventory

| File | Type | Change |
|---|---|---|
| `RichPythonUtils/.../stategraph.py` | Modify | +6 StateNode fields (§4.7); +2 tracker methods (`get_pending_thread_spawns`, `get_branch_items`); +2-line polarity inversion in `_check_condition`/`_check_gate`; +`load_from_dict`/`to_dict` round-trip parity |
| `RichPythonUtils/.../sop_manager.py` | Modify | +5 regex patterns (§4.5); +`_TAG_LINE_RE` two-pass scanner; +`render_for_mode` |
| `RichPythonUtils/.../workflow/__init__.py` | None | (unchanged — both StateGraph and WorkGraph already exported) |
| `AgentFoundation/src/agent_foundation/common/workflow/__init__.py` | New | Re-exports |
| `AgentFoundation/.../common/workflow/definition.py` | New | `WorkflowDefinition` (§6.1, ~40 LoC) |
| `AgentFoundation/.../common/workflow/instance.py` | New | `WorkflowInstance` + persistence (§6.2, ~80 LoC) |
| `AgentFoundation/.../common/workflow/registry.py` | New | `WorkflowRegistry` (§6.3, ~80 LoC) |
| `AgentFoundation/.../common/workflow/manager.py` | New | `WorkflowManager` (§6.4, ~150 LoC) |
| `AgentFoundation/.../common/workflow/sop_workgraph_node.py` | New | `SOPWorkGraphNode(WorkGraphNode)` (§5.5, ~150 LoC); v5 adds `is_branch_leaf: bool` flag |
| `AgentFoundation/.../common/workflow/branch_barrier_node.py` | New | **v5** — `BranchBarrierNode(WorkGraphNode)` (§5.3, ~50 LoC); convergence node for `__branch__` fan-out, mirrors BTA's workers→aggregator pattern |
| `AgentFoundation/.../common/workflow/sop_workgraph.py` | New | `build_sop_workgraph` factory (§5.6, ~50 LoC) |
| `AgentFoundation/.../inferencers/.../conversational_inferencer.py` | Modify | +`workflow_manager` attr; +yolo_mode threading; +3-section prompt rendering (§6.6) |
| `AgentFoundation/.../resources/prompt_templates/conversation/main/initial.jinja2` | Modify | +three new sections (available/ongoing/focused) |
| `AgentFoundation/.../resources/tools/enter_workflow/` | New | tool.json + executor.py (~30 LoC each) |
| `AgentFoundation/.../resources/tools/exit_workflow/` | New | tool.json + executor.py |
| `AgentFoundation/.../resources/tools/resume_workflow/` | New | tool.json + executor.py |
| `AgentFoundation/.../resources/tools/complete_phase/` | New | tool.json + executor.py (SOP-scoped) |
| `AgentFoundation/.../resources/tools/confirm_action/` | New | tool.json + executor.py (no-op in yolo mode) |
| `AgentFoundation/.../resources/tools/abort_phase/` | New | tool.json + executor.py |
| `AgentFoundation/.../common/workspace/allocator.py` | New | Moved from OpenStartup (§7) |
| `AgentFoundation/.../resources/tools/task/executor.py` | New | Moved from OpenStartup (§7) |
| `AgentFoundation/.../resources/tools/task/topologies/*.yaml` | New | 10 YAMLs moved (§7) |
| `OpenStartup/.../tools/task/executor.py` | Replace | Thin shim re-exporting from AgentFoundation (§7) |
| `AgentFoundation/.../resources/tools/sop/tool.json` | New | (§8.2) |
| `AgentFoundation/.../resources/tools/sop/executor.py` | New | CLI entrypoint (§8, ~200 LoC) |
| `AgentFoundation/.../resources/tools/sop/lint.py` | New | Static SOP linting (~150 LoC) |
| `AgentFoundation/.../ui/cli/__init__.py` | New | (§9.2) |
| `AgentFoundation/.../ui/cli/rich_terminal.py` | New | `RichTerminalInteractive(RichInteractiveBase)` (~100 LoC) |
| `AgentFoundation/.../ui/cli/prompts.py` | New | `ask_confirm`, `ask_single_choice`, etc. (~80 LoC) |
| `AgentFoundation/.../ui/cli/streaming.py` | New | Per-node token streaming (~60 LoC) |
| `AgentFoundation/.../ui/cli/theme.py` | New | Rich theme tokens (~30 LoC) |
| `AgentFoundation/pyproject.toml` | Modify | +`[ui]` extras: `rich`, `prompt_toolkit` |
| `AgentFoundation/.../resources/prompt_templates/conversation/main/_variables/workflow_sop/model_optimization.md` | Modify | Migrate to v2 format (§4.9) |

**Code volume estimate:** ~200 LoC (SOPWorkGraphNode + factory) + ~350 LoC (workflow framework: definition/instance/registry/manager) + ~200 LoC (sop tool) + ~270 LoC (ui/cli) + ~6 LoC (RichPythonUtils field additions) ≈ **~1,030 LoC of new code, 0 modifications to RichPythonUtils logic** (only additive field/method extensions).

---

## §15. Closing note

The architectural evolution that produced this plan (8 iterations across Rovo Dev v1 → v6 + Claude integrated, 3 wrong-substrate rounds, 1 dead-code integration bug, 2 substantive cross-pollinations with Claude's plan) is fully documented in **§0 Change history & lessons learned**. The three load-bearing lessons distilled from that evolution (§0.2) — *verify dispatch sites by reading code; `WorkGraphNode` extension uses `value=callable`, not method override; backward `__goto__` is forward expansion, not graph cycle* — are surfaced at point-of-use throughout §4 and §5 with concise `DO NOT` markers and back-references to §0.3.

v7 (this version) made no spec change; it reorganized the document so execution sections describe only the latest correct state and audit narrative lives in one place. See §0.1 row "v7" for the consolidation rationale.

---


