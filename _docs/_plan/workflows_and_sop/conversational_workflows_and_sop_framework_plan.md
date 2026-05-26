# Conversational Workflows + Enhanced SOP Framework — Integrated Plan

**Author:** Tony Chen (with Rovo Dev investigation)
**Date drafted:** 2026-05-24
**Status:** Draft v1 — pending review
**Scope:** Multi-domain plan spanning SOP grammar v2, first-class workflow runtime in `ConversationalInferencer`, `task`-tool relocation, new `sop` tool, and a new `agent_foundation/ui/` CLI library.

---

## §1. Why this plan exists — the four threads we are weaving together

The user surfaced four related but historically uncoupled ideas. They are coupled in practice and must be designed together:

1. **SOP format v2** (markdown-friendly tags on separate lines; new orchestration directives `__goto__` / `__afterwards__` / `__wait__` / `__branch__`; explicit `[__requires confirmation__]`).
2. **First-class workflows in `ConversationalInferencer`** — workflows become a peer to tools and skills: enter / exit / re-enter, stateful, identified by `workflow_id`, multiple concurrent workflows possible.
3. **Move `task` tool** from `OpenStartup/openteam/server/resources/tools/task/` to `AgentFoundation/src/agent_foundation/resources/tools/task/` (it is conceptually framework-level, not server-specific).
4. **New `sop` tool** (peer to `task`) that initializes a `ConversationalInferencer` and runs an SOP end-to-end in yolo or non-yolo mode. Non-yolo mode requires CLI UI components, motivating a new `agent_foundation/ui/` module.

These four threads share a single architectural pivot: **the SOP stops being free-form text inside a prompt variable and becomes a parsed, structured, stateful object that the runtime understands**. Everything else follows from that.

---

## §2. Empirical baseline — what exists today (verified)

| Area | Current state | Source |
|---|---|---|
| **SOP parsing** | No structured parser. `WorkflowContext.load_workflow_description()` reads the markdown file and stuffs the raw text into a prompt variable. | `agent_foundation/server/workflow_context.py:24` |
| **SOP state** | `WorkflowContext` has `current_phase: str`, `workflow_id: str`, `phase_records: list[WorkflowPhaseRecord]`, but the inferencer doesn't enforce the state machine — it's advisory text in the prompt. | `workflow_context.py:85+` |
| **Workflow re-entry** | None — `current_phase` is a single string; no stack, no concurrent workflows, no enter/exit semantics. | (absence) |
| **Yolo mode** | Single boolean on `RovoDevCliInferencer.yolo` (default True) — affects subprocess CLI invocation, NOT prompt-side instruction filtering. | `rovodev_cli_inferencer.py:75+` |
| **Tools registration** | Declarative in the SOP markdown via `**Tools**[__must__]:` lists. No code-level registry of "available tools." | (SOP markdown only) |
| **Skills registration** | Similar — declarative inside templates; no first-class registry. | (template variables) |
| **`task` tool** | Lives at `OpenStartup/src/openteam/server/resources/tools/task/`. Heavy AgentFoundation deps already; bound to OpenStartup via `cli.py` + `bootstrap.ensure_siblings_on_path()` + `tool_cli.run_cli()`. | (verified) |
| **AgentFoundation tools tree** | `src/agent_foundation/resources/tools/` exists but is sparse — no heavyweight common tools yet. | (verified) |
| **AgentFoundation `ui/`** | Directory exists but has no general CLI UI code; only `graph_reporter_factory.py` and similar artifact-rendering utilities. | (verified) |
| **CLI UI references** | `rankevolve/src/cli` uses `rich` + `prompt_toolkit` + `click` with a clean `ui/` module split. `acra-python` minimal. | (verified) |

---

## §3. Goals and non-goals

### Goals
- Define a precise, parseable, extensible **SOP grammar v2** with formalized tag scoping.
- Make `Workflow` a first-class runtime concept with `workflow_id`, enter/exit/re-entry, multi-instance support.
- Decouple `ConversationalInferencer`'s prompt from any single workflow — the prompt always advertises **available workflows** (like tools/skills) AND **ongoing workflow instances** (resumable).
- Move `task` tool to AgentFoundation; leave a thin shim in OpenStartup so existing CLI invocations keep working unchanged.
- Add a new `sop` tool (peer to `task`) that can run an SOP end-to-end with yolo / non-yolo modes.
- Lay foundational `agent_foundation/ui/` (rich + prompt_toolkit) as a reusable CLI UI toolkit; the `sop` tool is its first consumer.
- Migrate `model_optimization.md` → SOP v2 grammar; `code_optimization.md` already uses v2.

### Non-goals (explicit)
- Not rewriting `WorkflowContext` from scratch — extend it, preserve its serialization shape where possible.
- Not building a full TUI framework — start with a small, focused CLI component library.
- Not migrating OpenStartup's other tools in this plan — only `task`.
- Not adding workflow scheduling/persistence to a database — in-memory + per-session persistence is sufficient for v1.
- Not designing parallel/distributed workflow execution — single-process, single-conversation scope.

---

## §4. SOP grammar v2 — formalized

### §4.1 Design principles
1. **Markdown-friendly first.** Tags on separate lines below titles render cleanly in GitHub/Confluence/Bitbucket previewers.
2. **Tag scope is unambiguous.** Every tag has exactly one owner (phase, subsection, or instruction line).
3. **Orchestration is declarative.** `__goto__` / `__branch__` / `__depends on__` describe state-machine intent; the runtime owns execution.
4. **Two tag categories.** **Phase / subsection tags** (drive runtime behavior) vs. **instruction tags** (advisory; some have filter semantics like `[__requires confirmation__]` under yolo mode).
5. **Forward compatible.** Unknown tags MUST parse and survive round-trip; the runtime emits a `warning` but does not error.

### §4.2 Production grammar (EBNF)

```ebnf
sop                = workflow_metadata? phase+

workflow_metadata  = "# Workflow:" workflow_name NEWLINE description_block?
                   ; Optional H1 line declaring the workflow's display name.
                   ; If absent, name is derived from the markdown filename stem.

phase              = phase_header tag_line? body subsection*
phase_header       = "##" "Phase" phase_id "--" title NEWLINE
phase_id           = integer suffix?
suffix             = letter+                     ; e.g. "b", "c" for "Phase 3b"

subsection         = subsection_header tag_line? body tools_block?
subsection_header  = "###" title NEWLINE

tag_line           = blank_line "[" tag (separator tag)* "]" NEWLINE blank_line
separator          = "]" whitespace "["         ; chained brackets, e.g. "[a] [b]"

tag                = orchestration_tag | semantic_tag | unknown_tag

orchestration_tag  = "__initial__"
                   | "__depends on__" phase_ref ("," phase_ref)*
                   | "__goto__" phase_ref afterwards?
                   | "__branch__" branch_arg?
                   | "__for each__" identifier "__in__" identifier
                   | "__if__" condition_text

semantic_tag       = "__requires confirmation__"
                   | "__must__"
                   | "__optional__"
                   | "__prioritize__"

afterwards         = "__afterwards__" wait_arg?
wait_arg           = "__wait__" duration
duration           = integer time_unit
time_unit          = "s" | "m" | "h" | "d"
branch_arg         = identifier                  ; identifier of the output list
                                                 ; e.g. "[__branch__ proposals]"
                                                 ; meaning the previous phase's
                                                 ; output named "proposals" — each
                                                 ; element triggers one child phase

phase_ref          = "Phase" phase_id
unknown_tag        = "__" identifier "__" (whitespace tag_arg)*
                   ; Preserved verbatim for forward compatibility.

tools_block        = "**Tools**" tag_line ":" NEWLINE tool_list
tool_list          = ("- " tool_name NEWLINE)+
tool_name          = identifier

body               = (text_line | blank_line)*
```

### §4.3 Scope rules (the "where does this tag apply?" question)

This is the parser ambiguity that the SOP-format-comparison investigation raised. Rules, in priority order:

1. **A `tag_line` directly under a `phase_header` (with at most one blank line between) belongs to that phase.**
2. **A `tag_line` directly under a `subsection_header` belongs to that subsection.**
3. **A `tag_line` directly under a `**Tools**` block belongs to the tools list (subsection tag).**
4. **Inline `[__requires confirmation__]` inside a sentence is an `instruction tag`** — it modifies only that sentence/bullet, not the phase.
5. **Duplicate tags within the same scope are deduplicated**; the parser warns once.
6. **Unknown tags are preserved** in `Phase.unknown_tags: list[str]` for forward-compat; the runtime ignores them.

### §4.4 `__goto__` and `__afterwards__` semantics (the user's spec, formalized)

> User's quote: *"`__goto__ Phase 3 __afterwards__ __wait__ 1h` is an orchestration tag with args 'afterwards' and 'min wait', meaning there is a thread of the SOP going back to phase 3 after the current phase 3b (hard condition), and also that thread starts with wait time 1hour."*

Formalization:

| Directive | Semantics |
|---|---|
| `__goto__ Phase X` | **After** the current phase completes successfully, **spawn a new thread** at Phase X. The current phase does NOT block on this new thread. |
| `__goto__ Phase X __afterwards__` | Same as above; `__afterwards__` is the temporal qualifier (the goto happens *after* current phase completes, not immediately). |
| `__goto__ Phase X __afterwards__ __wait__ 1h` | The spawned thread sleeps for the `__wait__` duration before its first step. |

Multiple `__goto__` directives on one phase → multiple threads spawned. Threads are siblings; they each carry the workflow's full context. They are tracked by `WorkflowThread(parent_workflow_id, thread_id, target_phase, wake_time)`.

### §4.5 `__branch__` semantics

> User's quote: *"'branch' means the previous phase will output a list of items, and each item triggers a separate next phase."*

Formalization:

```
## Phase 3b -- Proposals
[__must__ output: proposals: list[Proposal]]

## Phase 4 -- Implement Proposal
[__depends on__ Phase 3b] [__branch__ proposals]
```

Semantics: when Phase 3b completes, the runtime reads its declared output `proposals` (must be a list). For each element, the runtime spawns one Phase-4 thread, each carrying the full workflow context plus a `branch_item` variable bound to that element. Threads run independently.

`__branch__` MUST be paired with a `__depends on__` that points to a phase declaring a list output. If the parent phase's output is missing or not a list, the runtime fails the workflow with a clear `BranchSourceMissing` error.

### §4.6 `[__requires confirmation__]` filtering under yolo mode

This is an **instruction tag** with active runtime semantics:

- **Non-yolo mode:** every line marked `[__requires confirmation__]` triggers a UI confirmation prompt before the agent acts on it.
- **Yolo mode:** the SOP-rendering pass strips all `[__requires confirmation__]` markers AND removes the entire line they decorate (the user's stated semantics: "all instructions with `[requires confirmation]` tag will be removed").

This filter is applied at the rendering layer, NOT the parser. The parser produces a typed AST; the renderer takes `(ast, yolo: bool)` → markdown.

### §4.7 What the OLD format had that NEW must not lose

The sop-format-comparison investigation flagged **two real losses** in the current `code_optimization.md`:

| OLD directive | Status in v2 | Action |
|---|---|---|
| `[__for each__ X __in__ Y]` | Preserved in grammar above | NEW format must add this — currently absent from `code_optimization.md` |
| `[__if__ condition]` inline branching | Preserved in grammar above (with `__branch__` as the structured alternative) | Both kept; `__branch__` is preferred for list-fanout, `__if__` for boolean gates |

The `code_optimization.md` file is therefore **a partial v2 example**, not the full grammar. The migration of `model_optimization.md` must use the full grammar including `__for each__` and `__if__`.

---

## §5. Workflow runtime — first-class concept

### §5.1 The mental model the user described

> *"workflow maybe just like a tool or skill, it has its place in the prompt, so the agent decides on its own when to enter and when to exit; the workflow is stateful, so you can exit half way and you can re-enter; every running workflow needs to have a workflow id."*
>
> *"When we enter a workflow, the prompt will have all its current support of workflow things, including workflow description, workflow next step guidance etc. but once it exits, those things are gone, but prompt still shows available workflows (just like tools/skills), and ongoing workflows (so you can resume)."*

Translation to code shapes:

```python
# NEW — workflow as first-class peer to tools/skills

@attrs.define
class WorkflowDefinition:
    """Static: parsed from an SOP markdown file."""
    workflow_id: str                    # stable, derived from filename or H1
    name: str                           # display name
    description: str                    # body before first phase
    sop_ast: SOPProgram                 # the parsed AST from §4
    yolo_filtered: bool = False         # has yolo filter been pre-applied?

@attrs.define
class WorkflowInstance:
    """Stateful: one in-flight execution of a WorkflowDefinition."""
    instance_id: str                    # unique per enter() call
    definition: WorkflowDefinition
    current_phase: PhaseRef             # the phase the next turn will execute
    completed_phases: list[PhaseRef]
    threads: list[WorkflowThread]       # active __goto__ / __branch__ spawn threads
    outputs: dict[str, Any]             # declared phase outputs (used by __branch__)
    state: Literal["active", "paused", "completed", "failed"]
    enter_time: datetime
    last_resume_time: datetime

@attrs.define
class WorkflowThread:
    """A __goto__ or __branch__ spawned thread."""
    thread_id: str
    parent_instance_id: str
    target_phase: PhaseRef
    wake_time: Optional[datetime]       # set by __wait__
    branch_item: Optional[Any]          # set by __branch__
```

### §5.2 Workflow registry

A `WorkflowRegistry` is a peer to the tools/skills registry. It is discovered by scanning a configurable directory (default: `resources/prompt_templates/conversation/main/_variables/workflow_sop/*.md`). Each `.md` file → one `WorkflowDefinition`.

### §5.3 The two-layer prompt rendering

The `ConversationalInferencer` prompt template will have two new sections that are always rendered:

```jinja2
{# Always advertise available workflows, like tools/skills #}
## Available workflows
{% for wf in workflow_registry.list_definitions() %}
- **{{ wf.name }}** (id: `{{ wf.workflow_id }}`): {{ wf.description | truncate(120) }}
  To start: call `enter_workflow(workflow_id="{{ wf.workflow_id }}")`
{% endfor %}

{# Show ongoing instances so the agent can resume #}
## Ongoing workflows
{% if workflow_state.active_instances %}
{% for inst in workflow_state.active_instances %}
- **{{ inst.definition.name }}** (instance: `{{ inst.instance_id }}`)
  Currently at: {{ inst.current_phase }}
  To resume: call `resume_workflow(instance_id="{{ inst.instance_id }}")`
{% endfor %}
{% else %}
(none)
{% endif %}

{# Only when one instance is active in this turn, render its rich context #}
{% if workflow_state.focused_instance %}
{% include "_partials/workflow_focused_context.jinja2" %}
{% endif %}
```

The `_partials/workflow_focused_context.jinja2` renders the SOP markdown for **only the focused instance**, with yolo filtering applied. When no workflow is focused, the rich context section is absent — exactly the user's intent.

### §5.4 Enter / exit / re-enter — the new tool surface

Three new tools are added to the `ConversationalInferencer`'s tool registry:

| Tool | Effect |
|---|---|
| `enter_workflow(workflow_id, init_inputs?)` | Instantiate a `WorkflowInstance` from the definition; set focus; persist to `WorkflowState` |
| `exit_workflow(instance_id?)` | Clear focus from the currently focused instance (or the named one); instance remains in `active_instances` with `state="paused"` so the agent can resume |
| `resume_workflow(instance_id)` | Set focus to a paused instance |
| `complete_workflow(instance_id, outputs?)` | Mark an instance `completed`; remove from `active_instances` |

These are surfaced just like ordinary tools — the agent can choose to call them at any turn.

### §5.5 State persistence

`WorkflowState` (the container for all instances) is serialized to/from the session JSON. Existing `WorkflowContext` is **kept** but **renamed conceptually** to "focused-instance projection" — it is now a computed view over `WorkflowState.focused_instance`, not the source of truth. Backward-compat: `WorkflowContext.to_dict()` continues to emit the same shape for one release; a `DeprecationWarning` is emitted on `from_dict()` when the old shape is detected.

---

## §6. Move the `task` tool to AgentFoundation

### §6.1 Verdict (from investigation)
The task tool **can be moved**, but it currently has two OpenStartup-specific bindings: `bootstrap.ensure_siblings_on_path()` and `tool_cli.run_cli()`. The recommendation: move the **framework-agnostic parts** (`executor.py`, `topologies/*.yaml`, `tool.json`) and leave a **thin shim** in OpenStartup that imports the moved code + adds OpenStartup-specific bootstrap.

### §6.2 New layout

```
AgentFoundation/src/agent_foundation/resources/tools/
└── task/
    ├── __init__.py
    ├── tool.json                          # declarative metadata (moved as-is)
    ├── executor.py                        # core execution logic (moved)
    ├── topologies/                        # YAML topology fixtures (moved)
    │   ├── default.yaml
    │   ├── full.yaml
    │   └── ... (8 files)
    └── cli.py                             # NEW thin AgentFoundation-side CLI
                                           # (uses agent_foundation.ui from §8)

OpenStartup/src/openteam/server/resources/tools/
└── task/
    ├── __init__.py
    └── cli.py                             # KEPT as thin shim:
                                           #   - calls openteam.bootstrap
                                           #   - imports executor from
                                           #     agent_foundation.resources.tools.task
                                           #   - delegates to openteam.tool_cli.run_cli
```

### §6.3 Bridge shim in OpenStartup

```python
# openteam/server/resources/tools/task/cli.py (post-move)
from openteam.bootstrap import ensure_siblings_on_path
ensure_siblings_on_path()  # sets sys.path for OpenStartup-local sibling imports

from agent_foundation.resources.tools.task.executor import execute as _execute
from openteam.server.services.tool_cli import run_cli

def main():
    run_cli(execute_fn=_execute, tool_name="task")

if __name__ == "__main__":
    main()
```

### §6.4 Workspace allocator dependency
`executor.py` currently imports `openteam.server.resources.tools._shared.workspace_allocator`. That module's *content* is generic (path-aware tool workspace allocation) — it is wrongly placed under OpenStartup. **Phase 4** of this plan moves the allocator alongside the `task` tool into `agent_foundation/resources/tools/_shared/workspace_allocator.py`, and OpenStartup's `_shared/` becomes a re-export shim.

### §6.5 Side benefit
After the move, the OpenStartup `task` cli is **one file** (the shim). The framework-agnostic executor + topologies are reusable across any AgentFoundation-using consumer.

---

## §7. New `sop` tool

### §7.1 What it does
A peer to `task`: given an SOP markdown file (or registered workflow id), spin up a `ConversationalInferencer`, enter the workflow, and drive it to completion. Supports two modes:

- **Yolo mode** (`--yolo`): runs end-to-end without user prompts; `[__requires confirmation__]` markers are stripped.
- **Non-yolo mode** (default): for each `[__requires confirmation__]` line, the runtime pauses and asks the user via the CLI UI from §8.

### §7.2 CLI surface

```bash
# Run a registered workflow by id
af-sop run --workflow-id code_optimization --target-path /path/to/repo

# Run an ad-hoc SOP file
af-sop run --sop-file ./my_workflow.md --target-path /path/to/repo

# Yolo mode
af-sop run --workflow-id code_optimization --target-path /path/to/repo --yolo

# Resume a paused instance
af-sop resume --instance-id <id>

# List available workflows
af-sop list

# Inspect an SOP file (parse + show AST + lint)
af-sop inspect ./my_workflow.md
```

### §7.3 Implementation surface

```
agent_foundation/resources/tools/sop/
├── __init__.py
├── tool.json
├── executor.py              # main entry point — spins up ConversationalInferencer
├── cli.py                   # CLI front-end (uses ui module from §8)
└── lint.py                  # SOP linter (uses §4 parser; called by `inspect`)
```

### §7.4 Non-yolo confirmation flow

```python
# Pseudo:
for turn in conversation:
    response = inferencer.next_turn(...)
    for instr in response.confirmable_instructions:
        if not ui.confirm(instr.text, instr.preview):
            inferencer.feedback("user declined: " + instr.text)
            break
    apply(response.actions)
```

The `ui.confirm` call uses the §8 UI library.

---

## §8. New `agent_foundation/ui/` CLI module

### §8.1 Library choice
Per investigation: adopt **`rich`** + **`prompt_toolkit`** + **`click`** (the rankevolve stack). This is the established pattern across `atlassian-packages/rankevolve/src/cli`. `acra-python` has nothing notable to borrow.

### §8.2 Module structure

```
agent_foundation/ui/
├── __init__.py                          # exposes the top-level facade
├── console.py                           # shared rich.Console + theme
├── theme.py                             # RichTheme defaults (colors, styles)
├── components/
│   ├── __init__.py
│   ├── phase_header.py                  # render_phase(name, status)
│   ├── confirmation.py                  # ask_confirmation(prompt, preview) -> bool
│   ├── progress.py                      # PhaseProgress(total_phases)
│   ├── stream_panel.py                  # StreamingPanel(title) — Live() wrapper
│   ├── status_footer.py                 # FooterBar(workflow_state)
│   └── select_menu.py                   # choose_one(options) — workflow picker
├── runners/
│   ├── __init__.py
│   └── sop_runner_ui.py                 # SOPRunnerUI facade for §7
└── graph_reporter_factory.py            # EXISTING — kept in place
```

### §8.3 Component contracts (minimal)

```python
# console.py
console: Console                          # singleton

# components/confirmation.py
def ask_confirmation(prompt: str, preview: Optional[str] = None,
                     default: bool = False, timeout: Optional[float] = None) -> bool: ...

# components/phase_header.py
def render_phase(phase_id: str, title: str,
                 status: Literal["pending", "active", "complete", "failed"],
                 tags: list[str]) -> None: ...

# components/stream_panel.py
class StreamingPanel:
    def __init__(self, title: str): ...
    def __enter__(self): ...                   # starts Live()
    def __exit__(self, *exc): ...
    def append(self, text: str) -> None: ...

# runners/sop_runner_ui.py
class SOPRunnerUI:
    def show_workflow_start(self, wf: WorkflowDefinition) -> None: ...
    def show_phase(self, phase: Phase, status: str) -> None: ...
    def ask_confirmation(self, instruction: str) -> bool: ...
    def stream_response(self) -> StreamingPanel: ...
    def show_completion(self, instance: WorkflowInstance) -> None: ...
    def show_pause(self, instance: WorkflowInstance, reason: str) -> None: ...
```

### §8.4 Why a facade
The `SOPRunnerUI` facade insulates the `sop` tool from the underlying library choices. Other consumers (future `task` non-yolo mode, future `code_review` tool, etc.) can either reuse `SOPRunnerUI` or compose their own from the lower-level components.

---

## §9. Phased rollout

| Phase | What ships | Risk | LOC est | Depends on |
|---|---|---|---|---|
| **0** | RED tests for SOP v2 parser (10 tests covering each grammar rule + scope rules) | low | +400 test | — |
| **1** | SOP v2 parser + AST (§4) — pure module, no runtime hook-up yet | low | +600 + tests | 0 |
| **2** | Workflow types (`WorkflowDefinition`, `WorkflowInstance`, `WorkflowThread`, `WorkflowRegistry`, `WorkflowState`) + serialization (§5.1–5.2, §5.5) | medium | +500 + tests | 1 |
| **3** | `ConversationalInferencer` two-layer prompt rendering (§5.3) + 3 new tools (`enter_workflow`, `exit_workflow`, `resume_workflow`, `complete_workflow`) (§5.4) | medium-high | +400 source + +200 templates + tests | 2 |
| **4** | Move `_shared/workspace_allocator` from OpenStartup → AgentFoundation; add OpenStartup re-export shim | low | net 0 | — |
| **5** | Move `task` tool (§6); add OpenStartup bridge shim; verify all existing `task` CLI tests pass unchanged | medium | net 0 + +1 shim | 4 |
| **6** | New `agent_foundation/ui/` module (§8) — console + theme + components + facade | low | +600 source + tests | — |
| **7** | New `sop` tool (§7) — uses §1–§6 stack end-to-end | medium | +400 source + tests | 1, 2, 3, 6 |
| **8** | Migrate `model_optimization.md` → SOP v2 grammar (preserves `__for each__` + `__if__` per §4.7) | low | +1 file diff | 1 |
| **9** | Yolo-mode filter pass (§4.6) integration end-to-end test | low | tests only | 7, 8 |

Total estimated effort: **~3 engineer-weeks** (1 engineer; ~50% test authoring, ~25% review).

### §9.1 Phase ordering rationale
- Parser before runtime — runtime can't be tested without an AST source.
- Workflow types before prompt rendering — prompt template needs the new types in feed.
- `_shared/workspace_allocator` move before `task` move — task depends on it.
- UI before `sop` tool — sop tool's CLI uses UI components.
- `model_optimization` migration last — proves parser handles full grammar including `__for each__`.

---

## §10. Risks + mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Existing `WorkflowContext` serialization shape change breaks resumed sessions | 🔴 HIGH | Backward-compat `from_dict()` handles both shapes for 1 release; `DeprecationWarning`; permanent regression test for old-shape JSON |
| 2 | Two-layer prompt rendering inflates token count (always-rendered "Available workflows" + "Ongoing instances") | 🟡 MED | Truncate descriptions to 120 chars in always-on sections; rich context only when focused; measure with a token-budget RED test |
| 3 | `__goto__` spawned threads compete for tool execution turns | 🟡 MED | v1: threads queued and executed sequentially (no concurrent tool invocation); document as known limitation; spec for v2 |
| 4 | `__branch__` source-list missing or wrong type | 🟡 MED | Runtime raises `BranchSourceMissing` with the parent phase's actual outputs in the error; lint catches it statically in `af-sop inspect` |
| 5 | OpenStartup `task` cli breaks after move | 🔴 HIGH | Bridge shim (§6.3) preserves exact CLI surface; existing test suite must pass unchanged before merge; smoke test in OpenStartup CI |
| 6 | Yolo filter accidentally strips non-confirmation lines | 🔴 HIGH | Filter operates ONLY on lines containing the literal `[__requires confirmation__]` marker; 6 unit tests cover edge cases (marker mid-line, marker in code block, marker in markdown link) |
| 7 | Unknown tags break the parser (forward compat) | 🟡 MED | Per §4.1 principle 5: unknown tags parse + survive round-trip; warning emitted once per tag-name per session; RED test pins this |
| 8 | `agent_foundation/ui/` pulls heavy `rich` + `prompt_toolkit` into the core package | 🟢 LOW | Add as optional extra in pyproject (`agent_foundation[ui]`); core agent code does not import `ui/` |
| 9 | SOP file discovery cost grows with many `.md` files | 🟢 LOW | Cache parsed `WorkflowDefinition`s by file mtime; invalidate on change |
| 10 | `WorkflowInstance.threads` grows unbounded with repeated `__goto__` loops | 🟡 MED | Per-instance thread cap (default 100); workflow fails with `ThreadLimitExceeded` if exceeded; configurable via SOP metadata |

---

## §11. Acceptance criteria

### §11.1 SOP v2 parser
- [ ] All 10 §4 grammar rules covered by passing tests
- [ ] `code_optimization.md` parses without warnings
- [ ] `model_optimization.md` (post-Phase-8 migration) parses without warnings and round-trips
- [ ] Unknown tag preserved through parse + serialize cycle
- [ ] Duplicate tag in same scope warns exactly once

### §11.2 Workflow runtime
- [ ] `enter_workflow` produces a `WorkflowInstance` with unique `instance_id`
- [ ] `exit_workflow` sets `state="paused"`, instance remains in `active_instances`
- [ ] `resume_workflow` re-focuses the paused instance
- [ ] Multiple concurrent instances of the same `WorkflowDefinition` coexist with distinct `instance_id`s
- [ ] Session round-trip preserves all instances (serialize → deserialize → identical state)
- [ ] Old-shape `WorkflowContext` JSON loads with `DeprecationWarning` and produces equivalent state

### §11.3 Prompt rendering
- [ ] "Available workflows" section renders even when no workflow is focused
- [ ] "Ongoing workflows" section lists all paused/active instances
- [ ] Focused-instance rich context renders ONLY for the focused instance
- [ ] Yolo mode strips all `[__requires confirmation__]` lines
- [ ] Token-budget test: rendered prompt ≤ 1500 tokens when 10 workflows registered + 5 instances active

### §11.4 task tool relocation
- [ ] All existing OpenStartup `task` CLI tests pass without modification
- [ ] `af-task` CLI works as a standalone AgentFoundation entry point
- [ ] No circular imports between `openteam` and `agent_foundation`

### §11.5 sop tool + UI
- [ ] `af-sop run --workflow-id code_optimization --target-path X --yolo` runs end-to-end without prompts
- [ ] `af-sop run` in non-yolo mode pauses on every `[__requires confirmation__]` line
- [ ] `af-sop list` discovers workflows from default + custom directories
- [ ] `af-sop inspect <file>` reports lint errors (missing `__initial__`, unresolved `__goto__` ref, `__branch__` without `__depends on__`, etc.)
- [ ] `af-sop resume --instance-id <id>` works after process restart (state loaded from session JSON)
- [ ] `agent_foundation[ui]` is the only extra needed for the sop tool

---

## §12. Open questions

1. **Where does `WorkflowState` persist when there's no session?** For `af-sop` running standalone (no AgentFoundation conversation server), suggest `~/.agent_foundation/workflows/<instance_id>.json`.
2. **Should `__goto__` threads run in parallel?** v1 plan says no (sequential queue). Confirm acceptable.
3. **Is the `WorkflowContext` rename a hard break or a soft alias?** v1 plan says soft alias for 1 release. Confirm.
4. **Should the SOP linter (`af-sop inspect`) block CI?** Suggest: yes for any SOP under `resources/prompt_templates/conversation/main/_variables/workflow_sop/`; no for user-supplied SOPs.
5. **Multi-workflow concurrent execution semantics.** If two instances are both "active" (not paused), and the agent's turn produces actions that touch both, which one's `current_phase` advances? Spec says: only the **focused** instance advances per turn; others are paused-by-default.

---

## §13. What I deliberately did not include

- **No database for workflow state.** JSON file persistence is sufficient for v1; the user did not ask for distributed/multi-process workflows.
- **No workflow versioning / migration.** If an SOP file changes mid-execution of an instance, behavior is "current instance keeps its snapshot of the AST, future enters see the new one." More elaborate versioning is out of scope.
- **No "approval workflow" / multi-user confirmation.** Non-yolo mode is single-user CLI; team-scale approval is a different design.
- **No web/REST UI.** CLI only. The §8 UI module is intentionally terminal-focused.
- **No migration of OpenStartup's other tools.** Only `task` per the user's request.
- **No first-class concept of "skill" alongside "workflow."** The user mentioned skills as a peer; skills already have a partial implementation. Touching them here would widen scope without clear ROI.

---

## §14. Implementation checklist (for the engineer who picks this up)

1. Read §4 thoroughly. Implement the grammar AS A `lark` or hand-written PEG parser — your call. Tests in `test/sop/test_parser.py`.
2. Add typed AST (`SOPProgram`, `Phase`, `Subsection`, `Tag` subclasses).
3. Implement renderer with `yolo: bool` parameter.
4. Add `WorkflowRegistry` that scans `resources/prompt_templates/conversation/main/_variables/workflow_sop/*.md`.
5. Implement `WorkflowState` + `WorkflowInstance` + serialization round-trip.
6. Modify `ConversationalInferencer` prompt template (§5.3).
7. Register 4 new tools (§5.4).
8. Move `_shared/workspace_allocator` (Phase 4).
9. Move `task` tool (Phase 5). Add OpenStartup shim.
10. Build `agent_foundation/ui/` (Phase 6).
11. Build `sop` tool (Phase 7).
12. Migrate `model_optimization.md` (Phase 8).
13. Run all acceptance criteria checks (§11).

---

*End of plan. Reviewers: please challenge §4.4 (`__goto__` thread semantics), §4.5 (`__branch__` source-list contract), §5.3 (token-budget impact of always-on workflow sections), and §6 (task move correctness for downstream OpenStartup CLI tests) most carefully.*
