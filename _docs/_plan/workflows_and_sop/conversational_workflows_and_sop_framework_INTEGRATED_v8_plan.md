# Conversational Workflows + SOP Framework — Enhancement Plan v8

**Author:** Tony Chen (drafted by Claude under his direction)
**Date drafted:** 2026-05-26
**Status:** Draft v8 — applies five focused enhancements on top of v7.2; pending review.
**Builds on:** [`conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md`](./conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md)

---

## §0. What this plan adds (delta from v7.2)

v7.2 specified the **substrate** (SOP-as-parsed-AST, WorkGraph as runtime, `SOPWorkGraphNode` bridge, enter/exit/resume semantics, `BranchBarrierNode`). It did **not** specify:

1. **How the SOP appears to the conversational orchestrator** — today the orchestrator sees one hard-coded SOP via a `workflow_description/default.jinja2` template; SOPs are not enumerable peers to skills/tools.
2. **How the orchestrator decides which SOP to enter** — there is no `keywords` / `example_requests` mechanism analogous to skill triggers.
3. **What "yolo mode" actually does at the conversation layer** — today it strips `[__requires confirmation__]` lines from rendered markdown (`SOPManager.render_for_mode` at `RichPythonUtils/.../sop_manager.py:567-584`), a behavior that is **dead code on the live OpenStartup path** because `WorkflowManager` is never wired into the conversational inferencer (`OpenStartup/.../factories.py:157-164` omits `workflow_manager=`).
4. **Where SOP runs live on disk** — v7.2 says "workspace under the WorkflowInstance" but the layout is not pinned, and there is no story for "an SOP run is itself a conversational session with per-turn artifacts".
5. **How synthetic responses are produced and labeled** — today there is no synthetic-response code path; auto-advance is a JS→server round-trip in `OpenStartup/src/openteam/ui/src/hooks/useManagerChat.js:187-222` that hard-codes the create-role flow.

This plan fills those five gaps. It does **not** revisit v7.2's substrate decisions.

### §0.1 The five enhancements (mapped to user asks 1-5)

| # | Ask | Phase in this plan |
|---|---|---|
| 1 | Refine the SOP description in `role_creation` to be skill-like (concise, info-dense, comprehensive) | [Phase B §3](#phase-b--keywords-example_requests-and-refined-sop-description) |
| 2 | Add `__keywords__` and `__example_requests__` SOP meta-tags | [Phase B §3](#phase-b--keywords-example_requests-and-refined-sop-description) |
| 3 | Replace `[__requires confirmation__]` stripping with conversational-inference auto-advance + per-tool `yolo_default` config | [Phase C §4](#phase-c--yolo-as-synthetic-default-responses) |
| 4 | Make SOPs top-level resources (folder + `.md` + `.json`), render in prompt like skills/tools; active SOPs hold their own prompt section | [Phase A §2](#phase-a--sops-as-first-class-resources) |
| 5 | Dedicated `sop/` runtime folder mirroring `tasks/`; each SOP run is a conversational session; synthetic turns labeled; nested task workspaces | [Phase D §5](#phase-d--sop-runtime-layout) |

---

## §1. Background: what currently exists (concise restate)

| Component | Status | Citation |
|---|---|---|
| `SOPManager` (parses markdown, extracts phases/directives/tags) | EXISTS | `RichPythonUtils/.../sop_manager.py:1-200` |
| `[__requires confirmation__]`, `[__depends on__]`, `[__for_each__]`, `[__goto__]`, `[__if__]`, `[__initial__]`, `[__branch__]` parsers | EXISTS | `sop_manager.py:53-93` |
| `WorkflowRegistry` (rglob `*.md` from `workflow_sop/`) | EXISTS | `AgentFoundation/.../workflow/registry.py:33-50` |
| `WorkflowManager.enter_workflow / exit_workflow / resume_workflow` | EXISTS | `workflow/manager.py:54-136` |
| `WorkflowInstance.yolo_mode` (persisted) | EXISTS | `workflow/instance.py:26, 41, 55` |
| `WorkflowManager.render_prompt_sections()` (produces `available_workflows` / `ongoing_workflows` / `workflow_description` / `workflow_status` / `workflow_nextstep_guidance`) | EXISTS but **dead code in OpenStartup** | `workflow/manager.py:146-188`; not wired at `OpenStartup/.../factories.py:157-164` |
| `SOPManager.render_for_mode("yolo")` strips `[__requires confirmation__]` lines | EXISTS but only caller is dead | `sop_manager.py:567-584` |
| `ConversationalInferencer.yolo_mode` attribute | EXISTS but **never read in body** | `conversational_inferencer.py:111` |
| Per-tool `tool.json` (clarification/single_choice/multiple_choice/confirmation each have one) | EXISTS | `AgentFoundation/.../resources/tools/*/tool.json` |
| Conversation tool handler round-trip (`_handle_conversation_tool` blocks on `aget_input`) | EXISTS | `conversational_inferencer.py:970-1097` |
| `SessionStore` with per-turn folders (`turn_NNN/`), `session.jsonl`, `<session>/tasks/...` | EXISTS | `OpenStartup/.../services/session_store.py:368-481` |
| JS-driven auto-advance (`is_auto_advance: true` round-trip) | EXISTS but **hard-coded for create-role**, JS only | `OpenStartup/.../routes/manager_websocket_routes.py:354-355`, `ui/src/hooks/useManagerChat.js:187-222` |
| `__keywords__`, `__example_requests__` parsers or fields | **DOES NOT EXIST** | — |
| `available_sops` prompt block / SOP discovery on the live path | **DOES NOT EXIST** | `initial.jinja2` has `available_workflows` block (line 7-10) but it never renders today |
| Per-tool `yolo_default` config | **DOES NOT EXIST** | — |
| Synthetic-response generation in `_handle_conversation_tool` | **DOES NOT EXIST** | — |
| `<session>/sop/` runtime folder | **DOES NOT EXIST** | — |
| `source: "human" \| "synthetic"` turn label | **DOES NOT EXIST** | turn metadata has no provenance field; `is_auto_advance` is a UX hint, not provenance |

### §1.1 The architectural pivot already taken in v7.2

v7.2 already decided: WorkGraph is the runtime substrate; the `WorkflowInstance` owns the SOP run; the conversational inferencer renders SOP-aware prompts via `WorkflowManager.render_prompt_sections()`. **What v7.2 left unfinished** is wiring this into OpenStartup's live conversational path and giving SOPs the discovery surface, runtime folder, and yolo behavior the user is now asking for.

---

## §2. Phase A — SOPs as first-class resources

### §2.1 Goal

Promote SOPs from "one file under `_variables/workflow_sop/`" to "a top-level `resources/sop/` folder with one subfolder per SOP, each containing `SOP.md` + `sop.config.json`". Mirror the **skill** layout (per-folder, body file, structured metadata) but use a separate JSON sidecar instead of YAML frontmatter (per user's explicit request in ask 4). Both `AgentFoundation` and `OpenStartup` get their own `resources/sop/` directory; the registry merges them like `load_all_skills(extra_dirs=...)` does for skills (`AgentFoundation/.../resources/skills/registry.py:94-119`).

### §2.2 Folder layout (canonical)

**Framework-generic SOPs (AgentFoundation):**

```
CoreProjects/AgentFoundation/src/agent_foundation/resources/sop/
├── __init__.py
├── registry.py                     # SOPRegistry — replaces (eventually) the WorkflowRegistry rglob
├── code_optimization/
│   ├── SOP.md                      # phase definitions (moved from workflow_sop/code_optimization.md)
│   └── sop.config.json
└── model_optimization/
    ├── SOP.md
    └── sop.config.json
```

**App-specific SOPs (OpenStartup):**

```
CoreProjects/OpenStartup/src/openteam/server/resources/sop/
└── role_creation/
    ├── SOP.md                      # phase definitions (moved + refined from role_creation.jinja2)
    ├── sop.config.json
    └── references/                 # optional: deep-dive notes the SOP body links to
        └── role_categories.md
```

The `references/` subdirectory mirrors how `twg` skill carries reference documents (`OpenStartup/.../skills/twg/references/*.md`). Phase bodies in `SOP.md` can link to these via relative paths.

### §2.3 `sop.config.json` schema

```jsonc
{
  // ─── identity ──────────────────────────────────────────────────────
  "name": "role_creation",                       // unique key; matches folder name
  "display_name": "AI Role Creation",            // human-friendly
  "version": "1.0.0",

  // ─── activation surface (used by orchestrator prompt) ──────────────
  "description": "...",                          // see §3 — refined skill-like description
  "keywords": ["create role", "hire", ...],
  "example_requests": ["I want to create a new Program Manager role", ...],
  "labels": ["onboarding", "role-management"],   // tag bag, matches skill `labels`

  // ─── runtime ───────────────────────────────────────────────────────
  "available_modes": ["default", "yolo"],
  "requires_tools": [                             // tools this SOP invokes; checked at registry-load
    "create_role", "role_setup", "team_onboard",
    "multiple_choice", "single_choice", "confirmation", "clarification"
  ],
  "max_goto_iterations": 5,                      // forwarded to WorkflowDefinition.frontmatter (v7.2 §4)
  "max_total_nodes": 100,
  "max_concurrency": 1,

  // ─── yolo behavior (per-SOP overrides; tool defaults live in tool.json) ──
  "yolo_overrides": {
    "multiple_choice": { "mode": "select_all" },
    "single_choice":   { "mode": "first_choice" },
    "confirmation":    { "mode": "fixed", "value": "yes" },
    "clarification":   { "mode": "fixed", "value": "Follow your best judgment based on the role context." }
  },

  // ─── persistence ───────────────────────────────────────────────────
  "preserve_workspace": true,                    // keep <sop_run>/ on completion
  "checkpoint_on_phase_complete": true
}
```

Fields are **all optional except `name`**; sensible defaults apply when omitted (e.g., `keywords: []`, `yolo_overrides: {}` falls back to tool-level defaults).

### §2.4 `SOPRegistry` — new loader (~80 LoC)

New file: `CoreProjects/AgentFoundation/src/agent_foundation/resources/sop/registry.py`.

```python
"""SOPRegistry — discovers SOPs from resources/sop/ trees.

Mirrors skills/registry.py: one folder per SOP, body + sidecar config.
Multiple search paths supported; later paths override earlier paths on name collision
(so OpenStartup SOPs can shadow AgentFoundation defaults).
"""

@dataclass(frozen=True)
class SOPInfo:
    name: str
    display_name: str
    description: str
    keywords: list[str]
    example_requests: list[str]
    labels: list[str]
    available_modes: list[str]
    requires_tools: list[str]
    yolo_overrides: dict[str, dict]
    config: dict[str, Any]          # raw sop.config.json (for runtime knobs)
    body_path: Path                  # absolute path to SOP.md
    body: str                        # cached body content
    folder: Path                     # absolute path to the SOP directory
    sop: SOP                         # parsed SOPManager.parse_markdown(body)


def load_sop(name: str, base_dir: Path) -> SOPInfo: ...
def load_all_sops(extra_dirs: list[Path] | None = None) -> dict[str, SOPInfo]: ...
def format_all_sops(sops: dict[str, SOPInfo]) -> str:
    """Render the catalog block for the conversation prompt."""
```

`format_all_sops` produces the same shape as `format_all_skills` (one-line-per-SOP) but with keywords + 1 example request inline:

```
- **role_creation** [onboarding, role-management] — Provision a new AI employee end-to-end…
  triggers: "create role", "hire", "onboard", "new role"  ·  e.g. "Hire a Data Scientist for the analytics team"
- **code_optimization** [perf] — Diagnose and remediate a performance bottleneck end-to-end…
  triggers: "speed up", "make faster", "reduce latency"  ·  e.g. "Make CodeOptimization/foo.py 5× faster"
```

### §2.5 Wiring into the live conversation prompt

**Three edits**, none of them invasive:

**(a) Prompt template:** rename `## Available Workflows` → `## Available SOPs` in `AgentFoundation/.../prompt_templates/conversation/main/initial.jinja2:7-10`. Same for `## Ongoing Workflows` → `## Active SOPs` at lines 12-16. The variables `available_workflows` / `ongoing_workflows` remain (back-compat) but we add `available_sops` / `active_sops` as the new canonical names; the Jinja block prefers the new names and falls back to the old.

**(b) Inferencer factory wiring (OpenStartup):** at `OpenStartup/.../backends/factories.py:157-164`, add `workflow_manager=` argument to the `ConversationalInferencer(...)` constructor. The `workflow_manager` is instantiated once per session via `WorkflowManager(registry=SOPRegistry().load_all(), session_workspace=<session_dir>, inferencer_factory=...)` and stored on the session-scoped service. This single change makes the existing `if self.workflow_manager is not None:` block at `conversational_inferencer.py:700-704` activate — `available_sops`, `active_sops`, `workflow_description`, `workflow_status`, `workflow_nextstep_guidance` all start populating from `SOPRegistry` + `WorkflowManager` instead of from the legacy `_variables/workflow_description/default.jinja2`.

**(c) Bridge from `WorkflowRegistry` to `SOPRegistry`:** during transition, `WorkflowManager` accepts EITHER registry. v8 ships the `SOPRegistry` as the canonical loader; `WorkflowRegistry` is kept as a thin adapter (`registry.py:60-89` becomes `SOPRegistry().to_workflow_registry()`) for the `/sop` tool's existing `enter_workflow` call path. After Phase A lands, `WorkflowRegistry` becomes a deprecation shell that simply delegates.

### §2.6 Active vs available SOPs in the prompt

```jinja2
{# Available SOPs — catalog, always rendered when SOPs exist #}
{% if available_sops is defined and available_sops %}
## Available SOPs
{{ available_sops }}
{% endif %}

{# Active SOPs — one block per WorkflowInstance currently active in this session #}
{% if active_sops is defined and active_sops %}
## Active SOPs
{% for sop in active_sops %}
### {{ sop.display_name }} (`{{ sop.instance_id }}`) — phase {{ sop.current_phase }} ({{ sop.status }})
<SOPDescription>
{{ sop.description }}
</SOPDescription>
<SOPStatus>
{{ sop.status_text }}
</SOPStatus>
<SOPNextStepGuidance>
{{ sop.nextstep_guidance }}
</SOPNextStepGuidance>
{% endfor %}
{% endif %}
```

**Multiple active SOPs are supported** (mirrors the existing `WorkflowManager.list_instances()` shape) but ONE is "focused" (matches `WorkflowManager.focused_instance_id`). Unfocused active SOPs render only their `SOPStatus` block to keep prompt size bounded; the focused SOP renders all three.

`WorkflowManager.render_prompt_sections()` is refactored to return a list of dicts (one per active SOP) plus the catalog string. The `_render_prompt()` block at `conversational_inferencer.py:700-704` consumes this directly.

### §2.7 Acceptance criteria for Phase A

- AC-A1: `SOPRegistry().load_all()` discovers `role_creation`, `code_optimization`, `model_optimization` from the new layout (and any extras under `~/.agent_foundation/sop/`, `$AGENT_FOUNDATION_SOP_PATH`).
- AC-A2: On a fresh OpenStartup session with no active SOP, the rendered prompt contains a `## Available SOPs` block listing all three with their description + keywords + one example.
- AC-A3: Entering an SOP (via `/sop role_creation` or LLM-initiated; see §6) produces an `## Active SOPs` block with phase-0 guidance; the `## Available SOPs` block continues to render so the user/LLM can see what else is available.
- AC-A4: SOP name collision logs a warning; OpenStartup SOPs win (matches the skill loader semantics at `skills/registry.py:115`).
- AC-A5: `sop.config.json` fields propagate to `WorkflowDefinition.frontmatter` so the existing `manager.py:77-78` constructor consumes `max_goto_iterations` / `max_total_nodes`.

---

## §3. Phase B — Keywords, example_requests, and refined SOP description

### §3.1 Goal

Make SOP "triggers" first-class so the orchestrator can match user intent to an SOP via structured fields instead of pattern-matching prose. Mirrors how skills currently rely on trigger phrases embedded in their `description` text — but explicit and parseable.

### §3.2 Two layers: JSON sidecar (primary) + Markdown tag (optional override)

**Primary (always present):** `keywords` and `example_requests` arrays in `sop.config.json` (§2.3). These are the source of truth for the registry and the prompt rendering.

**Optional in-SOP override:** the SOP markdown can declare `__keywords__` / `__example_requests__` tags using the same separate-line v2 grammar as other directives (`sop_manager.py:78` `_TAG_LINE_RE`). When present, they're additive to (or override) the JSON values. Useful for keeping triggers next to phase content when iterating.

**Markdown syntax (added to v7.2 grammar):**

```markdown
# AI Role Creation

[__keywords__]: create role, new role, hire, onboard, provision employee, set up agent, AI employee, new AI hire
[__example_requests__]:
- I want to create a new Program Manager role
- Hire a Data Scientist for the analytics team
- Set up an SRE AI employee
- Onboard a customer support lead for the support team

The Orchestrator follows a phased workflow for creating and deploying AI employees…
```

Both single-line (comma-separated) and bullet-list forms are supported. Parser changes are isolated to `SOPManager._extract_top_level_tags()` (new) — phase-level parsing is untouched.

### §3.3 Parser changes

Add to `RichPythonUtils/.../sop_manager.py`:

```python
_KEYWORDS_RE = re.compile(
    r"^\[?__keywords__\]?\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_EXAMPLE_REQUESTS_RE = re.compile(
    r"^\[?__example_requests__\]?\s*:\s*(.+?)(?=^\[?__|^##|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
```

`SOP` gains two top-level fields:
```python
@attrs(...)
class SOP(StateGraph):
    keywords: list[str] = attrib(factory=list)
    example_requests: list[str] = attrib(factory=list)
```

`SOPInfo` (`§2.4`) merges the JSON sidecar values with the markdown values: JSON wins on collision unless a `_merge_with_markdown: true` flag is set in the JSON.

### §3.4 Refined description for `role_creation`

**Before (current `role_creation.jinja2:1-3`, 41 words):**
> The Orchestrator follows a phased workflow for creating and deploying AI employees. The user drives the interaction; the Orchestrator guides through each phase and ensures prerequisites are met.

**After (proposed, 71 words, info-dense, skill-like):**
> Provision a new AI employee end-to-end — from raw role description to deployed team member. Walks the user through (1) capturing role specification via a multiple-choice intent picker, (2) generating a comprehensive role responsibility document via deep research, (3) decomposing the role into reusable skills + tools, (4) specializing the generic role for a specific team's Jira/Slack/Confluence context. Produces a versioned role document, a `final_deliverables/` skill+tool bundle, and a team deployment config.

This description goes in **both** `sop.config.json` (`description` field) and the SOP markdown body (first paragraph after `# Title`). The registry prefers the JSON value when both exist; the markdown value is a fallback for SOPs without a sidecar.

### §3.5 Catalog rendering (worked example)

With Phase A + Phase B, the rendered `## Available SOPs` block reads:

```
## Available SOPs
- **role_creation** [onboarding, role-management, ai-employees] — Provision a new AI employee end-to-end — from raw role description to deployed team member. Walks the user through (1) capturing role specification via a multiple-choice intent picker, (2) generating a comprehensive role responsibility document via deep research, (3) decomposing the role into reusable skills + tools, (4) specializing the generic role for a specific team's Jira/Slack/Confluence context. Produces a versioned role document, a `final_deliverables/` skill+tool bundle, and a team deployment config.
  triggers: "create role", "hire", "onboard", "new role", "provision employee", "set up agent", "AI employee"
  e.g. "Hire a Data Scientist for the analytics team"

- **code_optimization** [perf] — …
- **model_optimization** [ml] — …
```

This is the LLM's only signal for "should I propose entering this SOP for the user's request?". Format is intentionally similar to how Claude Code surfaces skills (name + 1-line summary + trigger list).

### §3.6 Acceptance criteria for Phase B

- AC-B1: `role_creation/sop.config.json` declares `keywords` (≥8 entries) and `example_requests` (≥5 entries).
- AC-B2: Markdown-form `__keywords__` and `__example_requests__` tags parse correctly when present; merge with JSON values per §3.3.
- AC-B3: `SOPInfo.description` is the refined version (§3.4); legacy `role_creation.jinja2` is removed once the new file is in place.
- AC-B4: `format_all_sops()` output for `role_creation` matches the §3.5 rendering (one-line description, triggers line, example line).

---

## §4. Phase C — Yolo as synthetic default responses

### §4.1 Goal

Replace v7.2's "strip `[__requires confirmation__]` lines from rendered markdown" with a clean separation: **the LLM still emits conversation tools as needed; the conversational inferencer auto-synthesizes the human response when yolo mode is on**, using a per-tool default that's configurable in each tool's `tool.json` and overridable per-SOP. Synthetic turns are logged identically to human turns with a `source: "synthetic"` provenance label.

This makes yolo mode work uniformly for all conversation tools — not just `confirmation` gates — without leaking yolo logic into the SOP markdown grammar.

### §4.2 Per-tool `yolo_default` in `tool.json`

Each of the four conversation tools gets one new top-level field. Schemas:

```jsonc
// AgentFoundation/.../resources/tools/multiple_choice/tool.json — add:
"yolo_default": { "mode": "select_all" }

// .../single_choice/tool.json — add:
"yolo_default": { "mode": "first_choice" }
// alternatively: { "mode": "fixed", "value": "<the value string>" }

// .../confirmation/tool.json — add:
"yolo_default": { "mode": "fixed", "value": "yes" }

// .../clarification/tool.json — add:
"yolo_default": { "mode": "fixed", "value": "Follow your best judgment." }
```

**`mode` enum:**
| mode | applies to | behavior |
|---|---|---|
| `fixed` | all four | use the literal `value` string as the synthesized response |
| `select_all` | multiple_choice | select every offered choice |
| `first_choice` | single_choice, multiple_choice | pick the first choice's `value` |
| `recommended` | single_choice, multiple_choice | pick the choice flagged `"recommended": true` in the LLM-emitted choices list (falls back to `first_choice` if none) |
| `prompt_llm` | all four | a second LLM call that, given the conversation history + the question, produces the answer (advanced; opt-in only — see §4.6) |
| `none` | all four | no default; require human reply even in yolo mode (escape hatch) |

### §4.3 Per-SOP overrides in `sop.config.json`

The `yolo_overrides` map in `sop.config.json` (§2.3) lets a specific SOP override tool defaults. Worked example for `role_creation`:

```jsonc
"yolo_overrides": {
  // role_creation's multiple_choice should pick the FIRST option (recommended category)
  // instead of selecting all, because phase-0 is asking about role focus
  "multiple_choice": { "mode": "first_choice" },

  // role_creation's confirmation auto-approves
  "confirmation": { "mode": "fixed", "value": "yes" },

  // role_creation's clarification uses a role-context-aware fallback
  "clarification": { "mode": "fixed",
                     "value": "Follow your best judgment based on the role context." }
}
```

### §4.4 Wiring in `ConversationalInferencer`

**Currently** the inferencer's yolo gate is at `conversational_inferencer.py:289`:
```python
if conv_response.has_conversation_tool and effective_interactive:
    collected = await self._handle_conversation_tools(...)
```

**Change** (~30 LoC):
```python
if conv_response.has_conversation_tool:
    if effective_interactive and not self.yolo_mode:
        collected = await self._handle_conversation_tools(...)   # existing path
    else:
        # NEW: synthesize from yolo defaults; identical return shape
        collected = self._synthesize_yolo_responses(
            conv_response.conversation_tools,
            yolo_overrides=self.prior_context.get("active_sop_yolo_overrides", {}),
        )
        # mark provenance so on_new_turn / save_turn_data can label it
        if on_new_turn:
            await on_new_turn(turn_number, user_input, source="synthetic")
```

New method `_synthesize_yolo_responses` on the inferencer (~50 LoC) does, per tool:
1. Look up the tool definition in `self.tool_registry` by `tool.tool_type` to get its `tool.json.yolo_default`.
2. Apply the per-SOP override from `prior_context["active_sop_yolo_overrides"]` if present.
3. Resolve to a concrete string per the mode enum (§4.2). For `select_all`/`first_choice`/`recommended`, read from the LLM-emitted `tool.choices` list.
4. Return `{output_var: value}` in the same shape `_handle_conversation_tools` returns, so downstream code at `:310-319` doesn't change.

### §4.5 Removing the legacy `render_for_mode("yolo")` strip

Now that synthesis happens at the conversation-tool layer, `SOPManager.render_for_mode` at `sop_manager.py:567-584` can be deleted. Its single caller at `WorkflowManager.render_prompt_sections():174-177` becomes unconditional:
```python
sections["workflow_description"] = definition.raw_markdown   # no mode-based filtering
```

The `[__requires confirmation__]` markers remain in SOP source — they're still useful as documentation and for the SOPPhase-level `requires_confirmation: bool` flag (`sop_manager.py:354-357`). But they no longer drive any prompt-time text manipulation. The flag becomes an LLM-readable hint ("this phase needs explicit user sign-off") that the LLM honors or skips depending on `yolo_mode` in the active SOP's status block.

**The user's specific ask in (3):** "we no longer require yolo mode to remove `[__requires confirmation__]` related instructions from the phase guidance text" → satisfied: yolo no longer manipulates phase text at all. The text stays as-is; the conversation tool the LLM emits gets auto-answered.

### §4.6 `prompt_llm` mode (opt-in, deferred to Phase C.1)

For sophisticated SOPs that need context-aware synthetic answers (e.g., the user has already said "Program Manager" in turn 1, and now the SOP asks "Which scope?" in turn 3), `mode: "prompt_llm"` triggers a second small LLM call:

```
Given the conversation history below and the question, produce the answer
that a thoughtful user would give based on context. Keep it concise.

Conversation:
<...recent turns...>

Question: <conversation tool prompt>
Choices (if any): <choices list>

Your answer:
```

This is implemented in `_synthesize_yolo_responses` as a branch that calls `self.base_inferencer.ainfer(...)` with a temperature ≈ 0.2 and a 200-token budget. Disabled by default (set per-tool or per-SOP). Recommendation: ship Phase C without `prompt_llm`; add it in Phase C.1 once the simpler modes are stable.

### §4.7 Acceptance criteria for Phase C

- AC-C1: Each of the four conversation tool.json files has a `yolo_default` field validated by an updated `ToolDefinition.from_dict` (`AgentFoundation/.../resources/tools/models.py:95-180`).
- AC-C2: When `ConversationalInferencer.yolo_mode=True`, a multiple_choice tool emitted by the LLM is answered with all selected (or per-SOP override) without blocking on `aget_input`; the synthetic answer feeds back to the LLM identically to a human reply.
- AC-C3: Synthetic turns are persisted with `source: "synthetic"` in `turn_NNN/metadata.json` and in `session.jsonl` (`UserInput` record).
- AC-C4: `SOPManager.render_for_mode` is removed; SOP markdown is rendered as-is regardless of yolo mode.
- AC-C5: End-to-end smoke test exercises `role_creation` SOP from Phase 0 → Phase 3 in yolo mode with zero human input.

---

## §5. Phase D — SOP runtime layout

### §5.1 Goal

Confirm the user's mental model (which is correct) and pin the on-disk layout:
- Each SOP run is itself a conversational inferencer session.
- Logged as a session, sibling to the parent conversation, under `<parent_session>/sop/<sop_run_id>/`.
- Each SOP turn gets a `turn_NNN/` folder identical in shape to a regular conversation turn.
- Synthetic auto-advance human responses are labeled `synthetic` in turn metadata; otherwise indistinguishable from human turns to the LLM.
- Any task tool (e.g., `/role-setup`) invoked from inside an SOP allocates its workspace under `<sop_run>/tasks/`, recursively mirroring the existing Path-B layout (`tool_dispatcher.py:209-216`).

### §5.2 Canonical layout

```
<runtime_root>/servers/server_<TS>_<uuid8>/sessions/<session_id>_<TS>/
├── session_state.json
├── session.jsonl                                   # parent conversation
├── turn_001/, turn_002/, ...                       # parent conversation turns
├── tasks/                                          # existing: tasks invoked from parent conversation
└── sop/                                            # NEW
    └── role_creation__20260526_154500__a1b2c3d4/   # one folder per WorkflowInstance
        ├── sop_state.json                          # WorkflowInstance.to_persistent_dict() + extras
        ├── sop_definition_snapshot.json            # frozen copy of sop.config.json at run start
        ├── session.jsonl                           # SOP-run turn log, same schema as parent
        ├── turn_001/
        │   ├── rendered_prompt.txt
        │   ├── template_feed.json
        │   ├── template_config.json
        │   ├── api_payload.json
        │   ├── inference_response.txt
        │   ├── user_input.txt                      # the synthetic-or-human response
        │   └── metadata.json                       # {source: "synthetic"|"human", phase_id, instance_id, ...}
        ├── turn_002/, ...
        └── tasks/                                  # tasks invoked from THIS SOP run
            ├── create_role__20260526_154700__b5c6d7e8/
            │   ├── outputs/, _runtime/, children/, logs/, ...
            └── role_setup__20260526_160230__c9d8e7f6/
                ├── ...
```

### §5.3 SOP run ID and folder naming

Per the task-runtime agent's recommendation:
- **Folder name:** `<sop_name>__<YYYYMMDD_HHMMSS>__<uuid8>` (double-underscore separator to distinguish from task folders' single-underscore `_`).
- **In-context SOP run ID:** `sop-<uuid8>` (mirrors `task-<uuid8>` at `tool_dispatcher.py:186`).
- **WorkflowInstance.instance_id** retains its current 8-hex format (`workflow/manager.py:59`); the folder embeds it as the `<uuid8>` segment for back-reference.

### §5.4 Implementation: `SOPSession` wrapping `SessionStore` patterns

Reusing as much of `SessionStore` as possible. New module `AgentFoundation/.../server/sop_session.py`:

```python
class SOPSession:
    """A conversational session scoped to a single SOP run.

    Lives at <parent_session_dir>/sop/<sop_run_folder>/. Reuses SessionStore's
    atomic-write helpers, turn-folder allocation, and JSONL log conventions.
    The parent session retains a reference (sop_run_id, sop_folder) in its
    own session_state.json under a new `active_sops: [...]` field.
    """

    def __init__(self, parent_session_dir: Path, sop_name: str,
                 instance_id: str, sop_config: dict, yolo_mode: bool): ...

    def allocate(self) -> Path:
        """Create <parent>/sop/<sop_run_folder>/ and write sop_state.json."""

    def get_tasks_dir(self) -> Path:
        """Return <sop_run>/tasks/ for nested task allocation."""

    def save_turn_data(self, turn_idx: int, *, source: Literal["human","synthetic"],
                       prompt_data: dict, user_input: str, response: str,
                       phase_id: str, **kwargs) -> None:
        """Mirrors SessionStore.save_turn_data with the extra source/phase_id fields."""

    def get_jsonl_logger(self) -> JsonLogger:
        """Returns a JsonLogger writing to <sop_run>/session.jsonl."""
```

`WorkflowManager.enter_workflow` is extended to construct an `SOPSession` and route the per-phase inferencer's `on_new_turn` callback into `SOPSession.save_turn_data`. The `inferencer_factory` already receives a `workspace=` argument (per `sop/executor.py:71`); it's repurposed to be the SOPSession's directory so the inferencer's per-turn artifacts land in the right place.

### §5.5 Parent session linkage

The parent session's `session_state.json` gains:
```jsonc
{
  // ...existing fields...
  "active_sops": [
    {
      "sop_name": "role_creation",
      "instance_id": "a1b2c3d4",
      "sop_run_folder": "sop/role_creation__20260526_154500__a1b2c3d4",
      "status": "active",
      "focused": true,
      "yolo_mode": true,
      "entered_at_turn": 7,
      "entered_at_iso": "2026-05-26T15:45:00Z"
    }
  ]
}
```

This is what the orchestrator reads to populate the `## Active SOPs` prompt block (§2.6) on every turn. When an SOP completes/aborts, its entry stays but `status` flips to `"completed"` or `"aborted"` — the folder persists so the user can browse turn-by-turn artifacts.

The parent conversation's `session.jsonl` records SOP lifecycle events:
```json
{"timestamp": "...", "type": "SOPInvoked", "sop_name": "role_creation", "instance_id": "a1b2c3d4", "sop_run_folder": "sop/role_creation__20260526_154500__a1b2c3d4", "trigger_turn": 7}
{"timestamp": "...", "type": "SOPCompleted", "instance_id": "a1b2c3d4", "completed_phases": ["0","1","1b","2","2b","3"]}
```

These are the only parent-side artifacts; the SOP's own turn artifacts live entirely in its sub-folder. This avoids double-logging and keeps the parent session JSONL focused on user-facing conversation.

### §5.6 Synthetic vs human turn labeling

Add a `source` field everywhere a turn is written:

| Location | Field added | Default |
|---|---|---|
| `manager_websocket_routes.py:348-355` user_msg dict | `source: "human"` (or `"auto_advance"` if the existing flag is set) | `"human"` |
| `conversation_service.py:559` `UserInput` JSONL record | `source: <forwarded>` | `"human"` |
| `session_store.py:368-415` `save_turn_data` → `metadata.json` | `source: <forwarded>` | `"human"` |
| `SOPSession.save_turn_data` (§5.4) | `source: "synthetic"` for yolo-generated responses | `"synthetic"` if yolo, else `"human"` |
| `SOPSession.save_turn_data` for explicit SOPInvoked transitions | `source: "system"` for the initial prompt that boots phase 0 | — |

The label flows end-to-end so a downstream tool (UI replay, evaluation harness) can filter by provenance. The existing `is_auto_advance` boolean (`manager_websocket_routes.py:354-355`) becomes a special case of `source: "auto_advance"` (it's a synthetic JS-side response, distinct from server-synthesized yolo replies).

### §5.7 Acceptance criteria for Phase D

- AC-D1: Entering `role_creation` via `/sop role_creation` (or LLM-driven) creates `<session>/sop/role_creation__<TS>__<uuid8>/` with `sop_state.json`, `sop_definition_snapshot.json`, empty `session.jsonl`, empty `tasks/`.
- AC-D2: First SOP turn produces `turn_001/` with rendered_prompt.txt, template_feed.json, inference_response.txt, user_input.txt (synthetic value), metadata.json with `source: "synthetic"`, `phase_id: "0"`, `instance_id: "<id>"`.
- AC-D3: When the SOP invokes `/create-role`, the task workspace lands at `<session>/sop/role_creation__<TS>__<id>/tasks/create_role__<TS>__<task_id>/`, with its own `outputs/`, `_runtime/`, etc.
- AC-D4: Parent session's `session_state.json` gains an `active_sops[]` entry with the SOP run folder, status, focused flag, entered_at_turn.
- AC-D5: Parent session's `session.jsonl` contains exactly one `SOPInvoked` record per SOP entry; no SOP-internal turns leak into the parent log.

---

## §6. SOP entry mechanism (cross-cutting)

The user said in ask 4: "the SOP list on the conversational inferencer prompt just like skills/tools, and after entering an SOP, the workflow description, next step guidance will render just like the current way". This implies two entry pathways:

**(a) LLM-driven (primary, implicit):** The LLM reads `## Available SOPs`, matches the user's request to keywords/example_requests, and emits a new conversation tool **or** action tool to enter the SOP. Recommendation: a dedicated `enter_sop` action tool (added to AgentFoundation tools registry) that takes `{sop_name: "role_creation", yolo: true|false, params: {...}}`. The orchestrator can invoke this without user friction once the user's intent is clear.

**(b) User-driven (explicit, slash command):** Keep the existing `/sop <name>` slash command (`AgentFoundation/.../resources/tools/sop/tool.json`) but reframe it: it's no longer a "run end-to-end and exit" tool; it's a "enter and stay active" tool that adds the SOP to `active_sops` on the parent session. The existing executor at `sop/executor.py:24-128` is refactored to use the new `SOPSession` + `WorkflowManager.enter_workflow` without `await instance._graph_task` (the SOP runs across multiple parent turns, not synchronously).

Both pathways converge on `WorkflowManager.enter_workflow(...)`, which:
1. Allocates the `<session>/sop/<sop_run>/` folder via `SOPSession.allocate()`.
2. Snapshots `sop.config.json` into the folder.
3. Adds the entry to the parent session's `active_sops[]`.
4. Sets `focused_instance_id` (defocusing any prior focused SOP).
5. Renders `## Active SOPs` from the next turn onward.

### §6.1 `exit_sop` and SOP switching

Symmetric `exit_sop` action tool (or just `/sop --exit <instance_id>` flag). LLM can also switch focus between active SOPs via `/sop --focus <instance_id>` without entering a new one. The user is encouraged to keep ≤2 active SOPs at a time; the system warns above 3 but doesn't block.

---

## §7. Migration: from `role_creation.jinja2` to `resources/sop/role_creation/`

### §7.1 File moves and creation

| Action | From | To |
|---|---|---|
| Move + rename | `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_sop/role_creation.jinja2` | `OpenStartup/src/openteam/server/resources/sop/role_creation/SOP.md` |
| Move + rename | `AgentFoundation/.../workflow_sop/code_optimization.md` | `AgentFoundation/.../resources/sop/code_optimization/SOP.md` |
| Move + rename | `AgentFoundation/.../workflow_sop/model_optimization.md` | `AgentFoundation/.../resources/sop/model_optimization/SOP.md` |
| Create | (new) | `OpenStartup/.../resources/sop/role_creation/sop.config.json` |
| Create | (new) | `AgentFoundation/.../resources/sop/code_optimization/sop.config.json` |
| Create | (new) | `AgentFoundation/.../resources/sop/model_optimization/sop.config.json` |
| Create | (new) | `AgentFoundation/src/agent_foundation/resources/sop/__init__.py` |
| Create | (new) | `AgentFoundation/src/agent_foundation/resources/sop/registry.py` |
| Delete after Phase A lands | `AgentFoundation/.../_variables/workflow_sop/` (entire folder) | — |
| Delete after Phase C lands | `AgentFoundation/.../_variables/workflow_description/default.jinja2` | — (replaced by `WorkflowManager.render_prompt_sections()`) |

### §7.2 Refining `role_creation/SOP.md` content (additional cleanup)

Three small fixes the investigation surfaced:

1. **Fix the typo** at current `role_creation.jinja2:34`: `[__requires_confirmation__]` (underscore) → `[__requires confirmation__]` (space). Current regex `_REQUIRES_CONFIRMATION_RE` requires whitespace and silently no-ops on the typo'd form.
2. **Remove `Tools[__requires_confirmation__]:`** scaffolding — phase tool lists no longer need a per-tool confirmation marker because yolo synthesis handles this at the tool layer. The phase-level `[__requires confirmation__]` directive on the phase heading remains (it's parsed to `SOPPhase.requires_confirmation`).
3. **Wrap the top with the new `__keywords__` + `__example_requests__` tags** (§3.2 example) and the refined description (§3.4).

### §7.3 Test plan

Build the missing end-to-end test (yolo agent confirmed none exists today, §4 of yolo report). New file `AgentFoundation/test/.../test_role_creation_sop_e2e.py`:

```python
@pytest.mark.asyncio
async def test_role_creation_sop_runs_to_completion_in_yolo():
    session_dir = tmp_path / "session"
    parent_session = SessionStore(session_dir)
    sop_registry = SOPRegistry().load_all(extra_dirs=[OPENSTARTUP_SOP_DIR])
    manager = WorkflowManager(registry=sop_registry, session_workspace=session_dir, ...)

    instance_id = await manager.enter_workflow("role_creation", yolo_mode=True)
    # ... drives the parent conversation 4-6 turns ...
    assert manager.active_instances[instance_id].status == "completed"
    assert (session_dir / "sop" / f"role_creation__*__{instance_id[:8]}").exists()
    assert all(
        (turn_dir / "metadata.json").read_text().contains('"source": "synthetic"')
        for turn_dir in (session_dir / "sop" / ...).glob("turn_*")
    )
```

---

## §8. Open design questions for review

These three decisions are load-bearing enough that I want sign-off before implementation starts.

### §8.1 Should we deprecate the legacy `_variables/workflow_description/default.jinja2` path immediately?

**Context:** OpenStartup currently feeds `workflow_description` from `default.jinja2` because `workflow_manager` isn't wired. After Phase A wires it, `default.jinja2` becomes dead code.

**Options:**
- (a) **Recommended:** Delete it at the end of Phase A; document the migration in CHANGELOG.
- (b) Keep as fallback when no SOP is active (current behavior).

My recommendation is (a) because keeping the fallback creates a confusing "which source is authoritative" question every time the file changes.

### §8.2 Single focused SOP or multiple in parallel?

**Context:** The plan supports multiple active SOPs (each gets a section in `## Active SOPs`) but only ONE is "focused" at a time (its `SOPNextStepGuidance` is the one the LLM acts on by default). User wrote "the conversational inferencer might need to maintain active SOPs" (plural) but the wording is ambiguous between "many active, one focused" and "many in parallel, all driving simultaneously".

**Options:**
- (a) **Recommended:** Many active, one focused. Unfocused SOPs show status-only. User/LLM can `/sop --focus <id>` to switch.
- (b) Many parallel, all driving — every turn the LLM decides which SOP(s) advance based on user message classification. More powerful, much harder to predict.

My recommendation is (a) because (b) creates ambiguous tool-invocation routing and forces complex priority logic. (a) ports naturally to (b) later by allowing multiple focused.

### §8.3 `prompt_llm` yolo mode in v8 or defer?

**Context:** §4.6's `prompt_llm` mode produces context-aware synthetic answers via a second LLM call. Powerful but adds latency + cost per yolo turn and requires careful prompt design.

**Options:**
- (a) **Recommended:** Defer to Phase C.1 (separate PR after C lands). v8 ships only `fixed` / `select_all` / `first_choice` / `recommended` / `none`.
- (b) Include in v8.

My recommendation is (a) because the simple modes cover the user's "select all by default, follow-best-judgment for free text" requirement and are debuggable in isolation. Adding `prompt_llm` opens prompt-engineering questions that are best handled separately.

---

## §9. Implementation phases & dependencies

```
Phase A — SOPs as resources
├── §2.4 SOPRegistry (new file, ~80 LoC)
├── §2.5(a) Prompt template rename: available_workflows → available_sops
├── §2.5(b) Wire workflow_manager into OpenStartup factories.py
├── §2.5(c) WorkflowRegistry → SOPRegistry bridge
└── §7.1 Move role_creation/code_optimization/model_optimization to new layout
       ↓ (Phase A complete: SOPs discoverable; one rendered as Active when entered)

Phase B — Keywords + example_requests + refined description
├── §3.3 Parser additions to sop_manager.py (~30 LoC)
├── §3.4 Refined role_creation description
├── §3.5 format_all_sops rendering update
└── (§7.2 SOP.md edits — fix typo, remove Tools[...], add top-level tags)
       ↓ (Phase B complete: SOP catalog rich enough for LLM-driven entry)

Phase C — Yolo synthetic responses
├── §4.2 Per-tool yolo_default in tool.json + ToolDefinition.from_dict updates
├── §4.3 Per-SOP yolo_overrides plumbing
├── §4.4 _synthesize_yolo_responses in ConversationalInferencer (~50 LoC)
├── §4.5 Delete SOPManager.render_for_mode + dead caller
└── End-to-end yolo smoke test
       ↓ (Phase C complete: yolo SOPs run without human input)

Phase D — SOP runtime layout
├── §5.4 SOPSession class (~150 LoC)
├── §5.5 Parent session active_sops[] field + SOPInvoked/SOPCompleted JSONL records
├── §5.6 source: "human"|"synthetic" labeling end-to-end
├── §5.3 SOP run folder naming + tasks nesting
└── AC-D1..D5 acceptance tests
       ↓ (Phase D complete: SOP runs persistable, browsable, resumable)

Phase C, D dependencies: Phase A must land first (registry surface).
Phase C ↔ D can be parallelized but D needs C's source label.
```

### §9.1 Suggested PR sequence

1. **PR-1: Phase A + B (registry, layout, catalog, refined description).** Pure additive — wires SOPs into the prompt but doesn't change yolo behavior. Lowest risk.
2. **PR-2: Phase C (yolo synthesis).** Adds the synthetic response code path; deletes `render_for_mode`. Medium risk — touches the agentic loop.
3. **PR-3: Phase D (runtime layout).** Adds `SOPSession` + folder structure. Highest LoC but mostly new code.
4. **PR-4: End-to-end test + role_creation migration polish.** Validates AC-A1..D5.

---

## §10. File change index

### New files

| Path | Purpose | LoC |
|---|---|---|
| `AgentFoundation/.../resources/sop/__init__.py` | Package marker | ~5 |
| `AgentFoundation/.../resources/sop/registry.py` | `SOPRegistry`, `SOPInfo`, `load_all_sops`, `format_all_sops` | ~80 |
| `AgentFoundation/.../resources/sop/code_optimization/SOP.md` | Moved from `workflow_sop/code_optimization.md` | (existing) |
| `AgentFoundation/.../resources/sop/code_optimization/sop.config.json` | Sidecar config | ~30 |
| `AgentFoundation/.../resources/sop/model_optimization/SOP.md` | Moved | (existing) |
| `AgentFoundation/.../resources/sop/model_optimization/sop.config.json` | Sidecar config | ~30 |
| `OpenStartup/.../resources/sop/role_creation/SOP.md` | Moved + refined from `role_creation.jinja2` | ~100 |
| `OpenStartup/.../resources/sop/role_creation/sop.config.json` | Sidecar config — primary instance of new format | ~50 |
| `AgentFoundation/.../server/sop_session.py` | `SOPSession` class | ~150 |
| `AgentFoundation/test/.../test_role_creation_sop_e2e.py` | End-to-end yolo test | ~150 |

### Modified files

| Path | Change | Phase |
|---|---|---|
| `RichPythonUtils/.../sop_manager.py` | Add `_KEYWORDS_RE`, `_EXAMPLE_REQUESTS_RE`, `_extract_top_level_tags`, `keywords`/`example_requests` on `SOP` | B |
| `RichPythonUtils/.../sop_manager.py` | Remove `render_for_mode` | C |
| `AgentFoundation/.../workflow/manager.py` | Accept `SOPRegistry`; `render_prompt_sections` returns active-SOPs list; remove `mode = "yolo" if focused.yolo_mode` branch | A, C |
| `AgentFoundation/.../workflow/registry.py` | Become thin adapter delegating to `SOPRegistry` | A |
| `AgentFoundation/.../inferencers/.../conversational_inferencer.py` | Add `_synthesize_yolo_responses`; gate at line 289 splits on `self.yolo_mode`; on_new_turn forwards `source=` | C, D |
| `AgentFoundation/.../resources/tools/{clarification,single_choice,multiple_choice,confirmation}/tool.json` | Add `yolo_default` field | C |
| `AgentFoundation/.../resources/tools/models.py` (`ToolDefinition.from_dict/to_dict`) | Parse + emit `yolo_default` | C |
| `AgentFoundation/.../resources/prompt_templates/conversation/main/initial.jinja2` | Rename `Available Workflows` → `Available SOPs`; rename `Ongoing Workflows` → `Active SOPs`; loop over `active_sops` for per-SOP sections | A |
| `AgentFoundation/.../resources/tools/sop/executor.py` | Refactor to enter-and-stay-active (no `await instance._graph_task`); use `SOPSession` | A, D |
| `OpenStartup/.../backends/factories.py` | Pass `workflow_manager=WorkflowManager(SOPRegistry(...).load_all(), ...)` to `ConversationalInferencer` | A |
| `OpenStartup/.../services/conversation_service.py` | `_compute_session_context` reads `active_sops[]`; `_on_new_turn` forwards `source`; `_persist_workflow_updates` updates `active_sops[]` | A, D |
| `OpenStartup/.../services/session_store.py` | `save_turn_data` accepts + persists `source`; new `get_session_sop_dir` helper | D |
| `OpenStartup/.../routes/manager_websocket_routes.py` | `user_msg["source"] = "auto_advance"` if `is_auto_advance` else `"human"` | D |
| `OpenStartup/.../ui/src/hooks/useManagerChat.js` | Reframe `is_auto_advance` synthetic-message generation to use new SOP-driven path (cleanup; UI continues to hide such messages) | D |

### Deleted files (after corresponding phase lands)

| Path | When |
|---|---|
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_sop/role_creation.jinja2` | After Phase A migration |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_sop/code_optimization.md` | After Phase A migration |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_sop/model_optimization.md` | After Phase A migration |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_description/default.jinja2` | After Phase A (per §8.1 recommendation a) |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow/.sop.config.yaml` | After Phase A (replaced by per-SOP sop.config.json) |

---

## §11. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Wiring `workflow_manager` into OpenStartup factory breaks existing sessions that don't expect `available_sops` prompt block | Block only renders when SOPs exist (guarded `{% if available_sops %}`); existing prompt text unchanged when none. |
| R2 | Synthetic responses in yolo mode produce nonsensical answers for multiple_choice with ambiguous options | Per-SOP `yolo_overrides` lets each SOP pick the right strategy; default `select_all` is conservative; `none` escape hatch forces human reply. |
| R3 | `SOPSession` and `SessionStore` diverge over time, causing per-turn artifact format drift | Make `SOPSession` USE `SessionStore` helpers (composition) rather than duplicate. Specifically, factor out `save_turn_data` into a free function that both call. |
| R4 | Active-SOP prompt block balloons context with multiple SOPs | Cap to 3 active SOPs by default; unfocused SOPs render status only (no description, no nextstep_guidance); add a config knob `MAX_ACTIVE_SOPS_IN_PROMPT=3`. |
| R5 | `is_auto_advance` JS round-trip and new server-side synthesis race or duplicate | JS round-trip is deprecated by D §5.6 once parity verified; until then, server detects synthetic origin via `source` field and de-dups. |
| R6 | Refactoring `WorkflowRegistry` into a thin adapter breaks the `/sop` tool's existing `enter_workflow` call | Adapter exposes the same `get / list_all` surface; only internal wiring changes. Test `sop/executor.py` paths in PR-1. |
| R7 | Per-SOP `yolo_overrides` shadowing global `yolo_default` makes debugging tricky | Add a debug logger line at synthesis: `[yolo] tool=X mode=Y source=tool.json|sop.config|builtin value=Z`. |
| R8 | Deleting `default.jinja2` breaks any code that still imports it | Grep before deletion; v7.2 §2 already documents the only consumer. |

---

## §12. What this plan explicitly does NOT do

- Does NOT change v7.2's WorkGraph substrate, `SOPWorkGraphNode`, `BranchBarrierNode`, or per-phase inferencer factory. All v7.2 decisions stand.
- Does NOT alter SOPManager's parsing of existing v2 tags (`__depends on__`, `__for_each__`, `__goto__`, etc.) except to add `__keywords__` / `__example_requests__`.
- Does NOT change skill registry, skill format, or skill rendering. Skills remain at `<name>/SKILL.md` with YAML frontmatter; only SOPs use `.md + .json` per user's explicit ask.
- Does NOT introduce a new conversation-tool type. The four existing types (`clarification`, `single_choice`, `multiple_choice`, `confirmation`) plus `tool_argument_form` remain unchanged in schema; only `yolo_default` is added to their `tool.json`.
- Does NOT delete the JS-side auto-advance immediately — it's deprecated in D §5.6 with parity check, removed in a later PR once server-side synthesis is verified equivalent.
- Does NOT add database persistence. All SOP state remains filesystem-only (matches OpenStartup's existing model per CLAUDE.md).

---

## §13. Quick reference — what changes vs what stays

**Stays:**
- WorkGraph + SOPWorkGraphNode + BranchBarrierNode runtime substrate (v7.2 §4-5)
- WorkflowManager + WorkflowInstance API (enter/exit/resume, focused_instance_id)
- SOPManager markdown parser (existing tags)
- Conversation tool types and their core schemas (`ConversationTool` dataclass, the four handler classes)
- Session/turn folder conventions for parent conversation
- Skill format (YAML frontmatter under `<name>/SKILL.md`)

**Changes:**
- SOPs move from `_variables/workflow_sop/<file>` to `resources/sop/<name>/SOP.md + sop.config.json`
- Conversation prompt gains `## Available SOPs` and `## Active SOPs` blocks (the previously-dead `available_workflows`/`ongoing_workflows` blocks)
- Yolo mode produces synthetic per-tool defaults instead of stripping markdown
- SOP runs live under `<session>/sop/<sop_run>/` with their own JSONL + turn folders
- Synthetic vs human turns are explicitly labeled in metadata

**Net effect for the user:**
- They can type "I want to hire a Data Scientist" and the orchestrator recognizes this from `role_creation`'s `keywords` + `example_requests`, suggests entering the SOP.
- With yolo on, the SOP runs end-to-end with the orchestrator making sensible default choices, producing a navigable `<session>/sop/role_creation__.../` tree to audit.
- With yolo off, the existing interactive multiple-choice / confirmation flow works exactly as today, just with cleaner prompt structure.

---

## §14. Approval-required decisions (summary)

Before implementation, please confirm:

1. **§8.1 — Delete `_variables/workflow_description/default.jinja2` after Phase A?** (Recommended: yes.)
2. **§8.2 — Single focused SOP with many active, vs many parallel-focused?** (Recommended: single focused, many active.)
3. **§8.3 — Include `prompt_llm` yolo mode in v8, or defer to C.1?** (Recommended: defer.)
4. **Folder location for OpenStartup SOPs:** `OpenStartup/src/openteam/server/resources/sop/` — confirm this is where you want `role_creation` to live (mirrors `skills/`)?
5. **`SOP.md` extension** (vs `.jinja2`): the new layout uses `.md` consistently per your ask (4). Confirm you want the SOP body parsed by `SOPManager.parse_markdown` directly without Jinja templating? (Today `role_creation.jinja2` doesn't actually use any Jinja syntax — it's pure markdown — so this is mostly a rename.)

Once these are confirmed I'll begin PR-1 (Phase A + B).
