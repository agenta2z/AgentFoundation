# Multi-Workflow-as-Skill Architecture for ConversationalInferencer

**Status**: Implementation-ready design proposal
**Author**: Drafted with @tchen7 (deep-investigation, critical-thinking iteration)
**Created**: 2026-05-09
**Scope**: `agent_foundation/common/inferencers/agentic_inferencers/conversational/*` and `agent_foundation/server/workflow_context.py`, with consumer updates in `OpenTeam` (`openteam/server/services/{conversation_service,session_store}.py`) and prompt templates under `prompt_templates/conversation/main/_variables/workflow*`.
**Risk**: Medium — touches the core prompt assembly path and the singular workflow state model that downstream consumers (OpenTeam, rankevolve) rely on. Backwards compatibility with the existing single-SOP path is required.
**Estimated effort**: ~7–10 active engineering days across 7 phases.

---

## §1 Problem Statement (User Intent → Engineering Restatement)

### §1.1 What @tchen7 asked
Three intertwined asks (paraphrased and verified against the SOP file at `OpenTeam/.../prompt_templates/conversation/main/_variables/workflow/sop.jinja2`):

1. **Investigate**: How does the conversational inferencer support SOP flows today? Trace the path from the SOP file through `WorkflowContext` → `prior_context` → the rendered prompt seen by the LLM.
2. **Re-architect** so the conversation is **not fixated on one workflow**:
   - Workflows should behave like first-class catalog citizens (parallel to **tools** and **skills**) — they have an entry in the prompt, a discoverable name, and the LLM/agent decides when to enter and when to exit.
   - Workflows are **stateful**: the agent can suspend mid-workflow, do unrelated work, and resume.
   - Each running instance of a workflow has a **`workflow_id`**.
   - The current variable-substitution mechanism (`_variables/workflow/sop.jinja2`) is **probably insufficient** for this; workflows likely deserve their own first-class subsystem.
3. **Prompt shape changes**:
   - When a workflow is **active**, the prompt should still surface workflow description / next-step guidance / status — exactly as today.
   - When **no workflow is active**, those blocks disappear, but the prompt still shows (a) the **available workflows** catalog (like the tools catalog) and (b) any **ongoing workflows** the agent could resume.

### §1.2 Engineering restatement (precise)
- Replace the implicit `1 session ↔ 1 SOP` assumption with an explicit **per-session workflow registry** and **workflow runtime** that:
  - Catalogs **available** workflows (declarative templates) that the agent can `ENTER`.
  - Tracks **ongoing** workflow instances (each keyed by `workflow_id`) that the agent can `RESUME` or `EXIT` (suspend/close).
  - Renders **only the active workflow's** description / next-step guidance / status blocks in the prompt — never two at once (V1 simplification; future-extension noted).
- The agent (LLM) is the **authoritative decision-maker** for ENTER/RESUME/EXIT, via three new **conversation/system meta-tools** (or one tool with a `verb` parameter) that mutate workflow runtime state.
- The existing `WorkflowContext` becomes a **per-instance** struct; the session holds a **stack/list** of these.

---

## §2 Discovery Summary — How SOP Flows Through the Stack Today

### §2.1 Files that matter (pinned references)

| Concern | File | Key symbols |
|---|---|---|
| **Inferencer turn loop** | `agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | `ConversationalInferencer.run_agentic_loop` (L132), `_render_prompt` (L567), `set_prior_context` (L520), `prior_context` dict |
| **Prompt renderer + variable manager** | `.../conversational/prompt_rendering.py` | `JinjaPromptRenderer.render` (L67), `template_variables` (L148), `variable_manager` (L110), `find_sop_file` (L187) |
| **Workflow state struct** | `agent_foundation/server/workflow_context.py` | `WorkflowContext` (L84), `WorkflowPhaseRecord` (L58), `to_status_text()` (L355), `to_dict/from_dict` (L464/L481), `STRATEGY_FILE_MAP` (L19), `load_workflow_description()` (L24) |
| **OpenTeam consumer (per-turn glue)** | `OpenTeam/src/openteam/server/services/conversation_service.py` | `_compute_session_context` (L352), `_persist_workflow_updates` (L392), `_load_workflow_description` (L380) |
| **OpenTeam session persistence** | `OpenTeam/src/openteam/server/services/session_store.py` | `_default_workflow_context` (L458), `update_workflow_context` (L231), `create_session` (L158) |
| **Prompt template + variable layout** | `OpenTeam/.../prompt_templates/conversation/main/` | `initial.jinja2` (top-level), `_variables/workflow/sop.jinja2` (the SOP itself), `_variables/workflow_description/default.jinja2`, `.initial.config.yaml` |

### §2.2 The current dataflow for ONE workflow (verified, end-to-end)

```
session_state.json
  └─ workflow_context: dict   ← single instance, persisted between turns
     ├─ strategy: "default"
     ├─ workflow_description: <jinja-rendered text from default.jinja2>
     ├─ current_phase, phase_status
     ├─ completed_phases: [WorkflowPhaseRecord, ...]
     ├─ phase_outputs: { … }
     ├─ task_queue, max_parallel_tasks, tool_phase_map, …
     └─ state_tracker (transient — not persisted)

           │
           ▼   conversation_service._compute_session_context()
prior_context (per-turn dict)         ─────────────────────────────►  ConversationalInferencer.set_prior_context(...)
  ├─ session_root_path
  ├─ workflow_status         (= WorkflowContext.to_status_text())
  ├─ workflow_description    (full text — multi-KB)
  ├─ strategy
  ├─ current_phase / phase_status
  ├─ completed_phases / phase_outputs
  └─ (later: _sop, tool_phase_map, _confirmation_gate_passed, _completed_gate_phases, …)

           │
           ▼   ConversationalInferencer._render_prompt() each turn
feed dict for Jinja2:
  { **template_variables (incl. _variables/workflow/sop expanded into workflow_var),
    workflow_nextstep_guidance:  SOPManager.render_guidance(tracker, sop, …),
    action_tools: <markdown>, conversation_tools: <markdown>,
    **prior_context,  ← merges in workflow_status etc.
    completed_actions, conversation_history, current_turn }

           │
           ▼   Jinja2 renders initial.jinja2
final prompt → LLM
```

### §2.3 Critical observations

1. **`workflow_description` text travels TWICE**: it's stored verbatim inside `session.workflow_context.workflow_description` AND re-injected into `prior_context.workflow_description` every turn. Multi-KB duplication.
2. **The SOP file is *singular*** — `JinjaPromptRenderer.find_sop_file()` looks at exactly one path (`_variables/workflow/sop.{jinja2,j2,md,yaml,yml}`). There is no concept of multiple SOPs being known to the system.
3. **`prior_context` is flat**. Phase fields (`current_phase`, `phase_status`, `completed_phases`, `phase_outputs`, `tool_phase_map`) live at the **top level**, with **no namespacing by workflow**. Two concurrent workflows would collide on every key.
4. **`workflow_description` is the single source of truth for phase IDs** — the comment in `conversational_inferencer.py:638-650` validates SOP `phase_ids` against regex matches in `workflow_description`. This 1:1 coupling assumes a single workflow.
5. **`STRATEGY_FILE_MAP` is a `default → default.jinja2` shim** — there is *one* default strategy. The plumbing for multiple strategies exists (`set_strategy()`) but the registry is empty.
6. **Phase advancement is LLM-driven** but state mutation is achieved by:
   - `_execute_tool_call()` consulting `tool_phase_map` and calling `start_phase`/`complete_phase` on the live `WorkflowContext` via the running `prior_context` dict.
   - Confirmation-gate phases: a `confirmation` widget result sets `_confirmation_gate_passed=True` in `prior_context`; on the next turn, `_render_prompt` auto-completes the gate phase.
7. **`OpenTeam.conversation_service` mediates state**: it copies `session.workflow_context → prior_context` *before* the turn, lets the inferencer mutate `prior_context` *during* the turn, then `_persist_workflow_updates()` rebuilds a `WorkflowContext` *from* `prior_context` and writes it back. This is the seam where multi-workflow state must be plumbed.
8. **`variable_manager` is per-template, not per-session** — it caches YAML defaults plus widget-collected overrides, but it is not a session/workflow-scoped state store. So overloading it for workflow state is a category error.

### §2.4 Why the variable mechanism is the wrong layer for workflows

| Variable mechanism (today) | Workflows (desired) |
|---|---|
| Selects ONE variant per variable per render | Need MANY workflows known concurrently |
| Statically discovered from filesystem (`_variables/<var>/<choice>.jinja2`) | Need an addressable registry with per-instance state |
| Pure templating; no lifecycle | Lifecycle: enter → progress → suspend → resume → exit |
| No identity (no `id`) | Each running instance needs `workflow_id` |
| No persistence in session | Must persist across turns AND across server restarts |
| Conflicts validated against folder names (`prompt_rendering.py:175-182`) | Workflows must coexist with normal variables and tools/skills |

**Conclusion**: Workflows belong in a **dedicated registry + session-stack subsystem**, parallel to (not on top of) tools, skills, and template variables.

---

## §3 Target Architecture — Workflows as First-Class Citizens

### §3.1 The mental model (parallel to tools & skills)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Prompt (initial.jinja2) sections                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  • System / role preamble                                              │
│  • Available Tools          ← from tool_registry (existing)            │
│  • Available Skills         ← from skill_registry (existing-ish)       │
│  • ★ Available Workflows    ← NEW — from workflow_registry             │
│  • ★ Ongoing Workflows      ← NEW — list[(workflow_id, name, phase)]   │
│  • ★ ACTIVE Workflow Block  ← only when one is "active" (resumed/      │
│       — Description           current). Identical to today's blocks.   │
│       — Status                                                          │
│       — Next-step guidance                                              │
│  • Conversation history                                                 │
│  • Current turn                                                         │
│  • Decision Procedure (extended with ENTER/RESUME/EXIT semantics)      │
│  • Response format                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### §3.2 New domain objects

#### `WorkflowDefinition` (immutable, per-template, registered at server start)
```python
@dataclass(frozen=True)
class WorkflowDefinition:
    name: str                    # e.g. "openstartup_orchestrator"
    display_name: str            # "OpenStartup AI-Employee Onboarding"
    description: str             # short blurb shown in catalog
    sop_path: Path               # absolute path to the SOP file
    description_path: Path       # workflow_description template
    tags: tuple[str, ...] = ()   # for filtering / categorization
    enter_aliases: tuple[str, ...] = ()  # natural-language hooks
    # Future: triggers (regex / classifier hints), preconditions, etc.
```

Backed by a filesystem **convention** that mirrors how skills are organized today:

```
prompt_templates/conversation/main/_variables/workflows/
  ├─ openstartup_orchestrator/
  │    ├─ workflow.yaml           ← name, display_name, description, tags…
  │    ├─ sop.jinja2              ← (== today's _variables/workflow/sop.jinja2)
  │    ├─ description.jinja2      ← (== today's _variables/workflow_description/default.jinja2)
  │    └─ .sop.config.yaml        ← (existing per-SOP config)
  ├─ team_onboarding/
  │    ├─ workflow.yaml
  │    └─ sop.jinja2
  └─ … more workflows …
```

A migration shim (Phase 2) keeps the legacy single-folder layout working for one release.

#### `WorkflowInstance` (mutable per-turn, persisted in session)
```python
@dataclass
class WorkflowInstance:
    workflow_id: str              # uuid (e.g., "wf-<unix>-<6hex>")
    definition_name: str          # FK → WorkflowDefinition.name
    status: Literal["active", "suspended", "completed", "errored"]
    started_at: float
    last_active_at: float
    context: WorkflowContext      # ← the existing dataclass, scoped per-instance
    parent_workflow_id: str | None = None   # for nested workflows (V2)
```

**Key insight**: `WorkflowContext` (today) becomes the *body* of a `WorkflowInstance`; the new wrapper adds identity, lifecycle, and timestamps.

#### `WorkflowRegistry` (server-side singleton)
```python
class WorkflowRegistry:
    def list_definitions(self) -> list[WorkflowDefinition]: ...
    def get(self, name: str) -> WorkflowDefinition: ...
    @classmethod
    def from_directory(cls, root: Path) -> "WorkflowRegistry": ...
```
Loads `workflow.yaml` files at startup; idempotent and reload-safe (parallel to the existing tool registry).

#### `SessionWorkflowState` (session-scoped runtime state)
```python
@dataclass
class SessionWorkflowState:
    instances: dict[str, WorkflowInstance] = field(default_factory=dict)  # keyed by workflow_id
    active_workflow_id: str | None = None   # which instance is "in focus" this turn
    # Stack semantics (V2): list[str] of workflow_ids; head = active
```

Persisted under `session["workflow_state"]` (replaces the old `session["workflow_context"]`; migration in §6).

### §3.3 Lifecycle verbs (LLM-controllable)

The agent (LLM) controls workflow lifecycle through three meta-actions. Two design choices, with a recommendation:

**Option A (recommended)**: Three explicit **system tools** (registered in the tool registry with `tool_type="System"` so they show up in the existing tool catalog and the existing `ToolsToInvoke` JSON contract handles them — no new format):
- `workflow_enter(definition_name, params={...}) → workflow_id`
- `workflow_resume(workflow_id)`
- `workflow_exit(workflow_id=None, reason="...")`  *(if id omitted, exits the active one)*

**Option B**: A single `workflow(verb=enter|resume|exit, …)` tool. Slightly fewer tool entries but worse discoverability and harder to schema-validate.

Decision: **A**, because (i) it slots cleanly into the existing tool-call parsing path, (ii) the LLM already groks "use the X tool to do X", and (iii) per-tool docs are clearer.

Each meta-tool is implemented as a **tool executor** in AgentFoundation that mutates `SessionWorkflowState` via the inferencer's `prior_context`, mirroring how `start_phase`/`complete_phase` work today.

### §3.4 Re-shaping `prior_context`

Today (flat):
```python
prior_context = {
    "workflow_status": ..., "workflow_description": ...,
    "current_phase": ..., "phase_status": ...,
    "completed_phases": [...], "phase_outputs": {...},
    "tool_phase_map": {...}, "_sop": <SOP>, …
}
```

Target (namespaced + catalog):
```python
prior_context = {
    # ── Catalog (always present, cheap) ─────────────────────
    "workflow_catalog": [
        {"name": "...", "display_name": "...", "description": "..."},
        ...
    ],
    "ongoing_workflows": [
        {"workflow_id": "wf-...", "name": "...", "display_name": "...",
         "current_phase": "Phase 1", "phase_status": "running",
         "last_active_at": "2026-05-09T18:00:00Z"},
        ...
    ],

    # ── Active workflow (present iff active_workflow_id is set) ──
    "active_workflow": {
        "workflow_id": "wf-...",
        "name": "openstartup_orchestrator",
        "display_name": "...",
        "description": "<rendered description>",
        "status": "<rendered to_status_text()>",
        "nextstep_guidance": "<rendered SOPManager.render_guidance()>",
        # plus the legacy keys aliased for in-place template compat:
        "current_phase": ..., "phase_status": ...,
        "completed_phases": [...], "phase_outputs": {...},
    },

    # ── Misc plumbing ──
    "session_root_path": ...,
}
```

Templates get **two ergonomic forms**:
- New canonical names (`active_workflow.status`, etc.) for new templates.
- Top-level aliases (`workflow_status`, `workflow_description`, `workflow_nextstep_guidance`) injected **only when an active workflow exists** so today's `initial.jinja2` keeps rendering the same as before.

This satisfies the user's requirement: "*once it exits, those things are gone, but prompt still shows available workflows and ongoing workflows*."

### §3.5 Updated prompt template (sketch — not full)

```jinja2
{# … role preamble … #}

## Available Workflows
{% if workflow_catalog %}
You can ENTER any of these structured workflows when appropriate:
{% for w in workflow_catalog %}
- **{{ w.display_name }}** (`workflow_enter("{{ w.name }}")`): {{ w.description }}
{% endfor %}
{% else %}
(no workflows configured)
{% endif %}

{% if ongoing_workflows %}
## Ongoing Workflows
You may RESUME any of these in-progress workflows:
{% for w in ongoing_workflows %}
- `{{ w.workflow_id }}` — **{{ w.display_name }}** at *{{ w.current_phase }}* ({{ w.phase_status }}, last touched {{ w.last_active_at }}). Resume via `workflow_resume("{{ w.workflow_id }}")`.
{% endfor %}
{% endif %}

{% if active_workflow %}
## Active Workflow — {{ active_workflow.display_name }}  (id={{ active_workflow.workflow_id }})

<WorkflowDescription>
{{ active_workflow.description }}
</WorkflowDescription>

<WorkflowStatus>
{{ active_workflow.status }}
</WorkflowStatus>

<WorkflowNextStepGuidance>
{{ active_workflow.nextstep_guidance }}
</WorkflowNextStepGuidance>

You may EXIT this workflow at any time via `workflow_exit("{{ active_workflow.workflow_id }}", reason=...)`.
{% endif %}

## Available Tools
{{ action_tools }}
… etc …
```

Decision Procedure update (in `initial.jinja2`):
- Step 0 (NEW): **Decide workflow stance** —
  - If user message clearly fits an *available* workflow's domain and no workflow is active → consider `workflow_enter(...)`.
  - If user message references / clearly continues an *ongoing* workflow → consider `workflow_resume(...)`.
  - If currently active workflow doesn't apply → consider `workflow_exit(...)` first, then handle ad-hoc.
- Existing steps 1–6 remain, but step 2 ("diff status against guidance") only applies when `active_workflow` is set.

---

## §4 Component-by-Component Changes

### §4.1 `agent_foundation/server/workflow_context.py`
- Keep `WorkflowContext` and `WorkflowPhaseRecord` largely unchanged; they're the per-instance body.
- **Remove** `STRATEGY_FILE_MAP` and `load_workflow_description()` from this module — these belonged to the singleton model. Move equivalents to the new `WorkflowRegistry`.
- Move `_WORKFLOW_DESC_PHASE_RE` to a small helper module (`workflow_phases.py`); it's needed by SOP/desc cross-validation regardless of how workflows are registered.
- Add `WorkflowInstance` and `SessionWorkflowState` dataclasses (or co-locate in a new `agent_foundation/server/session_workflow_state.py` to keep `workflow_context.py` slim).

### §4.2 New: `agent_foundation/server/workflow_registry.py`
- `WorkflowDefinition` dataclass.
- `WorkflowRegistry` with `from_directory(root)` discovery.
- `load_description(definition)` and `load_sop(definition)` helpers (read once per process; cache by mtime).
- Backwards-compat shim: if directory `_variables/workflows/` does not exist but `_variables/workflow/` does, synthesize a single `WorkflowDefinition` named `"default"` from the legacy files.

### §4.3 `agent_foundation/.../conversational/conversational_inferencer.py`
- `set_prior_context()` is unchanged (still receives a dict).
- `_render_prompt()` changes:
  - **Catalog branch** (always): if `prior_context["workflow_catalog"]` is set → pass through.
  - **Active branch** (conditional): if `prior_context.get("active_workflow_id")` is set, perform the existing SOP loading / `SOPManager.render_guidance(...)` work, but using the **active workflow's** SOP path (looked up via the registry — passed in via `prior_context["active_workflow"]["sop_path"]`), not via `JinjaPromptRenderer.find_sop_file()`.
  - The current `find_sop_file()` codepath remains as fallback for legacy single-SOP templates.
- New: helper method `_compute_workflow_block(active_instance) -> dict` that returns the `active_workflow` dict (description, status, nextstep_guidance) — reuses `WorkflowContext.to_status_text()` and `SOPManager.render_guidance()` exactly as today.
- Tool execution path (`_execute_tool_call`) updates: when a tool's `phase_map` membership matters, look it up against the **active instance's** `tool_phase_map`, not a global one. Tools fired while no workflow is active simply don't advance any phase (today's behavior — already a no-op for unmapped tools).

### §4.4 New: `agent_foundation/resources/tools/workflow_meta_tools/`
Three system tool definitions following the existing `tool.json` registration convention discovered in §2:

- `workflow_enter/tool.json` (+ executor)
- `workflow_resume/tool.json` (+ executor)
- `workflow_exit/tool.json` (+ executor)

Each executor mutates `prior_context["_workflow_state_updates"]` (a queued list of mutations applied at the end of the turn before persistence) — **avoids in-place mutation of `SessionWorkflowState` from inside a tool**, which would race with the prompt-render snapshot taken at turn-start.

### §4.5 `agent_foundation/.../conversational/prompt_rendering.py`
- `find_sop_file()` is **deprecated but kept** for backwards-compat with templates that don't yet declare a `workflows/` folder.
- Add `find_workflow_directory()` returning the `_variables/workflows/` path if present; the inferencer prefers this.
- `template_variables` continues to handle non-workflow variables unchanged. Conflict detection (L175-182) gets one more reserved name: `workflows`.

### §4.6 `OpenTeam/src/openteam/server/services/conversation_service.py`
- `_compute_session_context()` rewrites:
  - Pulls the new `session["workflow_state"]` (a `SessionWorkflowState` dict).
  - Asks the registry for `workflow_catalog` and builds `ongoing_workflows`.
  - If `active_workflow_id` is set, builds `active_workflow` block (description text via `registry.load_description(def)`; status via `WorkflowContext.to_status_text()`).
  - Returns flat dict ready for `set_prior_context()`.
- `_persist_workflow_updates()` rewrites:
  - Reads `prior_context["_workflow_state_updates"]` (the queued mutations from meta-tools) and applies them to `SessionWorkflowState`.
  - For the active instance, rebuilds its `WorkflowContext` from the mutated `prior_context` (legacy keys at top-level), exactly like today but stored under `active_workflow.context`.
  - Calls `data_service.update_workflow_state(session_id, state.to_dict())`.

### §4.7 `OpenTeam/src/openteam/server/services/session_store.py`
- Replace `_default_workflow_context()` with `_default_workflow_state()` returning a `SessionWorkflowState.empty().to_dict()`.
- `update_workflow_context()` → renamed `update_workflow_state()`; backwards-compat alias kept for one release.
- `_backfill_workflow_context()` becomes `_migrate_workflow_state()`:
  - If session has the **old** key `workflow_context` and not the new `workflow_state`, wrap it as a single `WorkflowInstance` named `"default"` with `status = "active"` (preserves existing behavior).
  - Persist once and continue.
- `create_session()` initializes the empty `workflow_state` (no instances, no active id).

### §4.8 OpenTeam prompt templates
- Move `prompt_templates/conversation/main/_variables/workflow/` to `prompt_templates/conversation/main/_variables/workflows/openstartup_orchestrator/` and add `workflow.yaml`.
- Move `prompt_templates/conversation/main/_variables/workflow_description/default.jinja2` into the same folder as `description.jinja2`.
- Update `initial.jinja2` per §3.5 sketch. Keep the existing `<WorkflowDescription>` etc. tags so the WebUI's structural-XML escape list (`.initial.config.yaml`) keeps working.

---

## §5 Edge Cases & Decision Table

| Scenario | Behavior |
|---|---|
| User starts new session, sends greeting | No active workflow. Prompt shows catalog + (empty) ongoing list. LLM responds in natural language; may suggest entering a workflow. |
| User says "let's onboard a new role" + `openstartup_orchestrator` workflow exists | LLM emits `workflow_enter("openstartup_orchestrator")` → registry instantiates a `WorkflowInstance` with new `workflow_id` → next render shows `active_workflow` block + Phase 0 guidance. |
| User mid-Phase-1 says "what's the weather" | LLM detects **ad-hoc** turn (existing Decision Procedure step 3); answers without invoking workflow tools; the active workflow remains active and unchanged. |
| User mid-Phase-1 says "actually let's switch — set up Slack instead" | LLM may either (a) call `workflow_exit("wf-…", reason="user pivot")` then `workflow_enter("slack_setup")`, or (b) call `workflow_enter("slack_setup")` which **suspends** (not exits) the previous one. Recommendation: enter-while-active suspends the prior; explicit exit ends it. |
| User returns 3 days later and says "back to the role onboarding" | LLM sees `ongoing_workflows` with the prior `workflow_id`; emits `workflow_resume("wf-…")`; the next render resurrects the active block. |
| Two workflows partly progressed, user neither resumes nor enters | Both stay in `ongoing_workflows`; neither is active; no active block rendered. |
| LLM tries `workflow_enter("nonexistent")` | Tool executor returns error string; turn continues; LLM may apologize. (No state mutation.) |
| Workflow completes (final phase) | Tool that finalizes the workflow (or an internal SOP signal) sets the instance to `status="completed"`. Stays in `ongoing_workflows` for one turn marked completed, then promoted to `completed_workflows` (or pruned per a configurable retention). |
| Two simultaneously active workflows? (V2) | Out of scope for V1. Stack semantics in `SessionWorkflowState.instances` keep the door open; V1 enforces "one active at a time" in `workflow_enter` and `workflow_resume`. |
| Tool defined in workflow A's `tool_phase_map` is invoked while workflow B is active | Today's behavior already gracefully no-ops phase tracking when a tool isn't in the active map. We preserve this. We will additionally **log a warning** so cross-workflow tool leakage is visible. |
| Confirmation gate (existing `_confirmation_gate_passed`) while workflow inactive | Unreachable: gates only exist inside an active workflow. The `_render_prompt` autocomplete code becomes scoped to the active instance. |
| Legacy session loaded after upgrade | `_migrate_workflow_state()` wraps `session.workflow_context` into one active `WorkflowInstance("default")`; user sees no behavioral change. |
| Workflow YAML missing `description` | Fail registry load loudly at startup (catch in tests). |
| Two workflow YAMLs share `name` | Registry raises `DuplicateWorkflowNameError` at load. |
| `prior_context["_workflow_state_updates"]` contains `enter` then `exit` of same id same turn | Apply in order; final state = exited. Persistence reflects this. |

---

## §6 Migration & Backwards Compatibility

### §6.1 Two-stage filesystem migration
- **Stage 1 (this PR)**: Add support for the new `_variables/workflows/<name>/` layout. The old `_variables/workflow/sop.*` and `_variables/workflow_description/<choice>.jinja2` continue to load (via the registry's compat shim) as a synthetic `default` workflow.
- **Stage 2 (next PR)**: OpenTeam moves its files to the new layout; legacy paths still work but emit a `DeprecationWarning` once.
- **Stage 3 (release after that)**: Remove the legacy shim.

### §6.2 Two-stage session-state migration
- On read: `_migrate_workflow_state()` upgrades `workflow_context` → `workflow_state` lazily on first access (idempotent).
- `update_workflow_context(session_id, dict)` → kept as a thin wrapper that writes to `workflow_state.instances[<active_id>].context`.
- On write: only the new `workflow_state` key is written; the old `workflow_context` key is removed once.

### §6.3 API/contract surface
- `ConversationalInferencer.set_prior_context(dict)` and `_render_prompt()` signatures unchanged. Internal feed dict gains keys but losing none.
- The existing `<WorkflowDescription>`, `<WorkflowStatus>`, `<WorkflowNextStepGuidance>` blocks in `initial.jinja2` are wrapped in `{% if active_workflow %} … {% endif %}`. Templates with no workflow active simply omit them — same as today's `{% if workflow_description is defined and workflow_description %}` guard, structurally.
- `data_service.update_workflow_context(...)` kept as a deprecated alias for one release.

---

## §7 Phased Implementation Plan

### Phase 0 — Pre-flight audit & spike (0.5 day)
- [ ] Greenfield branch from `main`.
- [ ] Snapshot a real OpenTeam session traversing 1+ phase, save the rendered prompt, the prior_context, and session_state.json. These become the golden artifacts for regression checks at every phase end.
- [ ] Confirm the WebUI MarkdownRenderer escape list in `.initial.config.yaml` covers any new XML tag pairs we introduce (e.g., `<AvailableWorkflows>` if we go XML-tagged; current sketch is markdown-headed, no XML tags needed — confirm).
- [ ] Verify `rich_python_utils.common_objects.workflow.stategraph.StateGraphTracker` and `SOPManager.render_guidance` already accept a SOP loaded from any path (not hard-coded). Spot-check by calling with a temp file. (Highly likely yes; failure here would change the design.)

### Phase 1 — Domain model + registry (1.5 days)
- [ ] Create `agent_foundation/server/workflow_registry.py` with `WorkflowDefinition`, `WorkflowRegistry`, and the directory-walk loader.
- [ ] Create `agent_foundation/server/session_workflow_state.py` with `WorkflowInstance` and `SessionWorkflowState` (incl. `to_dict`/`from_dict` for JSON persistence).
- [ ] Move `_WORKFLOW_DESC_PHASE_RE` and re-export from `workflow_context.py` for compat.
- [ ] Unit tests:
  - Registry discovers 0/1/N workflows; raises on duplicates; raises on malformed YAML; legacy-shim materializes a `default` workflow.
  - `SessionWorkflowState` round-trips through dict.
  - `WorkflowInstance.last_active_at` updates on `resume`.

### Phase 2 — Meta-tools (1 day)
- [ ] Add `workflow_enter`, `workflow_resume`, `workflow_exit` tool definitions and executors. Pure functions over a `SessionWorkflowState` snapshot returning `(new_state, result_text)`.
- [ ] Integration: `ConversationalInferencer._execute_tool_call` recognizes meta-tools by `tool_type="System"` (or namespace prefix) and routes their results into a turn-end mutation queue (`prior_context["_workflow_state_updates"]`).
- [ ] Tests: enter/resume/exit happy paths + invalid-id errors + enter-while-active suspension.

### Phase 3 — Inferencer prompt-render changes (1.5 days)
- [ ] Update `_render_prompt` to consult `prior_context["active_workflow"]` for SOP path; keep `find_sop_file()` fallback.
- [ ] Build `workflow_catalog` and `ongoing_workflows` views (just lift from `prior_context` — the heavy lifting happens in the consumer).
- [ ] Inject legacy aliases (`workflow_description`, `workflow_status`, `workflow_nextstep_guidance`) iff `active_workflow` exists.
- [ ] Unit tests using a fake `prior_context`:
  - No workflows → no active block, empty catalog rendered.
  - Catalog only → catalog rendered, no active block.
  - Active workflow → active block AND catalog rendered (catalog shown so agent can switch).

### Phase 4 — OpenTeam consumer wiring (1.5 days)
- [ ] Rewrite `conversation_service._compute_session_context()` to build the new `prior_context` shape from `session["workflow_state"]` + `WorkflowRegistry`.
- [ ] Rewrite `_persist_workflow_updates()` to apply `_workflow_state_updates` and write back `workflow_state`.
- [ ] Add `session_store._migrate_workflow_state()` and `update_workflow_state()`. Keep `update_workflow_context()` as deprecation alias.
- [ ] Move OpenTeam's SOP file to the new folder layout; add `workflow.yaml`. Update `initial.jinja2` per §3.5.
- [ ] Snapshot tests: render prompt for (a) fresh session, (b) session mid-phase-1, (c) session with two suspended workflows. Compare to golden files captured in Phase 0 for the migrated case (ensures backwards-compat output equivalence for the "single active workflow" case).

### Phase 5 — End-to-end behavioral tests (1 day)
- [ ] Headless OpenTeam test: simulate user → `workflow_enter` → couple of phases → `workflow_exit` → ad-hoc Q&A → `workflow_resume` → completion. Assert session_state.json transitions match expected.
- [ ] Failure modes: enter unknown workflow; resume unknown id; exit when none active.
- [ ] Concurrency / suspension: enter A, do partial; enter B (auto-suspends A); exit B; resume A; verify A's state is intact.

### Phase 6 — Documentation & polish (1 day)
- [ ] Update `agent_foundation` developer docs (parallel to existing skill/tool docs) with a "Workflows" section.
- [ ] Add an example workflow under `agent_foundation/examples/workflows/` demonstrating the new layout end-to-end.
- [ ] Add migration notes to OpenTeam README for any sessions older than the upgrade.
- [ ] Deprecation warnings: legacy paths and `update_workflow_context` alias both emit on first access.

---

## §8 Open Questions (require @tchen7 decision)

1. **Tool granularity**: Confirm Option A (three meta-tools `workflow_enter/resume/exit`) over Option B (single `workflow(verb=…)`). Recommendation: A.
2. **Auto-suspend vs auto-exit on `workflow_enter` while another is active**: Recommended **auto-suspend** (preserves user's earlier work). Confirm.
3. **Do we expose workflows via slash-commands** (`/workflow enter foo`)? Easy to add as syntactic sugar that the LLM expands; not in V1 scope unless desired.
4. **Per-workflow `tool_phase_map` namespacing**: V1 reads `tool_phase_map` from the active instance only. Should we also enforce that tools mapped to workflow A *cannot* fire while workflow B is active? Recommendation: warn (don't block) in V1; revisit if the warning fires often.
5. **Concurrent active workflows (true parallel)**: Hard NO for V1 (one active, others suspended). Future V2 could enable concurrent execution by interleaving SOP guidance — but that explodes prompt size and complicates the agent's mental model.
6. **Parent-child workflows**: `WorkflowInstance.parent_workflow_id` field is reserved but unused in V1. Confirm we want to ship the field for forward-compat.
7. **Pruning policy**: when do completed/errored workflows leave `ongoing_workflows`? Default: keep for the rest of the session, hide from UI; confirm.
8. **`workflow_catalog` filtering by role**: should `WorkflowDefinition.tags` interact with the current employee's role to filter the catalog? V2.

---

## §9 Out-of-Scope (explicitly)

- Concurrent / parallel workflow execution (only one *active* at a time in V1).
- Cross-session workflow state (workflows don't migrate between sessions).
- Workflow versioning (no `WorkflowDefinition.version` semantics; YAML changes apply to new instances only).
- Visual workflow builder UI (server-side only in V1).
- Per-step retry policy / time budgeting at the workflow layer (still SOP-managed).
- Auto-resume on session reopen (the agent decides; we don't auto-resume).
- Replacing the SOP framework itself — `SOPManager` and `StateGraphTracker` remain the in-flight execution model.

---

## §10 Residual Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Legacy shim mis-detects a non-default SOP folder | Low | Med | Phase-2 explicit migration; snapshot tests |
| Prompt size growth from catalog + ongoing list | Med | Low-Med | Cap ongoing list at N (default 5); shorten descriptions; test with realistic registries |
| LLM gets confused between tools and workflows | Med | High | Distinct prompt sections (§3.5); decision-procedure update in `initial.jinja2`; behavioral tests in Phase 5 |
| `_workflow_state_updates` mutation queue races with mid-turn re-renders | Low | Med | Apply queue **only** after the agentic loop returns, in `_persist_workflow_updates`; mid-turn renders see only the start-of-turn snapshot |
| Existing `prior_context` consumers (rankevolve via `load_workflow_description`) break | Low | High | Keep the function as a thin wrapper that calls into the registry's default workflow; deprecation warning |
| `tool_phase_map` collisions across workflows | Med | Low | Per-instance maps; warn-don't-block in V1 (§8 Q4) |
| Snapshot tests over-specify and cause noise | Med | Low | Use structural assertions (catalog has K items; active block present) rather than full string compare where possible |

---

## §11 TL;DR

- **Today**: 1 session ↔ 1 SOP, baked through `WorkflowContext`, `prior_context`, and `_variables/workflow/sop.jinja2`.
- **Tomorrow**: A `WorkflowRegistry` of declarative workflow definitions; each session holds a `SessionWorkflowState` of `WorkflowInstance`s with unique `workflow_id`s; the LLM enters/resumes/exits via three system tools; the prompt always shows the catalog + ongoing list, and additionally shows the active workflow's description/status/next-step guidance only when one is active.
- **Compatibility**: legacy single-SOP layout and old session payloads are auto-upgraded for one release.
- **Effort**: ~7–10 days across 7 phases; medium risk; clear rollback (revert; legacy paths still work).
