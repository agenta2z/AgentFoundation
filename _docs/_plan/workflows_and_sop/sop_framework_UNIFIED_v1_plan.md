# SOP Framework — Unified Plan (v1)

**Author:** Rovo Dev (drafted in conversation with Tony Chen)
**Date:** 2026-05-26 18:59
**Status:** Draft v1 — single source of truth for the SOP framework

### §-1. Provenance and audit history

This plan **supersedes and consolidates**:
- `sop_runtime_enablement_plan.md` v3.1 (5 concerns: refined description, keywords/example_requests, yolo synthesis, SOP-as-resource, runtime layout)
- `multi_sop_focus_and_tool_concurrency_plan.md` v1.1 (mode system + PendingConversationToolQueue + ToolDefinition.concurrency)
- External "v8" proposal pasted 2026-05-26 18:51 (tracker-driven substrate decision; 6-mode yolo enum; source-pipe end-to-end; role_creation typo catch; sop.config.json schema depth)

The two predecessor plans remain on disk for reference and **WILL be archived** once this unified plan is reviewed (§16). Companion plan that this builds ON TOP OF (not superseded):
- `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md` (overall SOP architecture; WorkGraph substrate decision; `__keywords__`/`__example_requests__`-aware grammar foundation). v7.2 stands; this plan layers runtime/registry/queue concerns on top.

| Round | Date | Source | What was integrated / corrected |
|---|---|---|---|
| 1.0 | 2026-05-26 18:59 | initial unified draft | Plan #1 + Plan #2 + v8 best parts; 3 architectural disagreements resolved with rationale |
| 1.1 | 2026-05-26 19:19 | empirical gap audit (Tony) | §6.9 added: `recommended` field on `ChoiceItem` is empirically absent (`conversation_tools.py:26-46` parser drops unknown fields); resolver `recommended` mode required schema+parser+UI changes that v1.0 silently assumed. Patches: 1 new subsection (~30 LoC); 4 new ACs (AC6.9-6.12); 1 new RED test (#15); 1 new risk (R17). |
| 1.2 | 2026-05-26 19:39 | subagent architecture concretization (Tony) | v1.0 §9 was vague on what runs an active SOP ("tracker-driven or asyncio.Task per SOP"). Empirical evidence: (a) `WorkflowInstance` (`manager.py:54-65`) is pure metadata — no inferencer reference; (b) BTA (`breakdown_then_aggregate_inferencer.py:1168, 1180, 1675`) already implements the subagent-with-shared-interactive pattern (`worker.interactive = self.graph_reporter.node_interactive(_node_name)`); (c) `RichInteractiveBase._current_input_mode` is instance state — concurrent SOPs calling `aget_input()` would clobber it; (d) `InteractiveBase` has zero built-in queue. v1.2 concretizes §9 with subagent model: each active SOP runs as its own `SOPInferencer(ConversationalInferencer)` instance, shares parent's `interactive` via per-SOP `RoutedInteractive` proxy (matches BTA `:1677`), parent owns `PendingConversationToolQueue` as session-coordinator router. Patches: §9.1 rewrites unit definition (P1); new §9.11 shared-interactive coordination (P2); §9.4 clarifies routing lives in parent (P3); §10 confirms per-SOP turns folder is per-inferencer message log (P4 — already correct, just restated); §8.1 restates queue ownership at parent/session level (P5); new AC9.11 (P6); new R18 (P7). |
| 1.3 | 2026-05-26 20:16 | v1.2 propagation audit (Tony) | v1.2 added subagent model to §9 but did NOT propagate the change to §1 (architectural pin) / §2 (disagreement resolution) / §8 (queue scope) — creating top-down contradictions. Reviewer found 3 HIGH inconsistencies + 3 MED specification gaps + 3 LOW polish issues; all 9 empirically verified. v1.3 patches: (Q1) §1 pin reframed — "active-SOP execution is tracker-driven *inside each per-SOP SOPInferencer*"; (Q2) §2 row 1 rewritten with parent/child clarification; (Q3) §8.1 + AC8.5 corrected; (Q4) §9.12 NEW; (Q5) §9.13 NEW; (Q6) folded into §9.12.2; (Q7) §3 description rephrased; (Q8) §9.4 2-LLM-call cost disclosure; (Q9) §9.11.3 wording fix; new ACs AC9.12 + AC9.13; new R19. |
| 1.5 | 2026-05-26 21:49 | v1.4 wiring-gaps audit (reviewer) | Reviewer found 3 real gaps in v1.4: (1) MED — `WebSocketInteractive._direct_send` override declared in narrative but not in the code block, so implementer would silently fall through to base-class `asyncio.to_thread(send_response)` → `NotImplementedError`; (2) **HIGH** — `input_queue → serializer.deliver_response()` bridge missing: with serializer active, `aget_input` no longer reads `input_queue`, so the React UI's `pending_input_response` messages (handled by `manager_websocket_routes.py:570-585` putting into `input_queue`) would never reach the serializer. Without this bridge, the serializer's per-caller futures never resolve — full deadlock; (3) LOW — `has_any_active()` called in §9.11.4 but never defined on `InteractionSerializer`. Also rejected the §3 "multiple-choice intent picker" claim as stale (already fixed in v1.3 Q7; empty grep confirms). v1.5 patches: V16 adds `_direct_send` override to `WebSocketInteractive` code block; V17 adds new §9.11.2.1 specifying the bridge inside `WebSocketInteractive.aget_input` (one-shot `_bridge_one_response` task, cancellation-safe, layered correctly — bridge NOT in route handler to preserve cross-package boundary); V18 defines `has_any_active()` on `InteractionSerializer`; AC9.15 NEW (bridge correctness under N concurrent callers); R21 NEW (bridge concurrency risk with mitigation). |
| 1.4 | 2026-05-26 21:25 | interactive-layer serialization redesign (Tony + reviewer) | v1.3 §9.11 introduced `RoutedInteractive` proxy + `PendingConversationToolQueue` as 2 new SOP-specific classes. Reviewer's deeper architectural critique: **the concurrent-callers-sharing-one-user-attention problem is a transport concern, not an SOP concern**; serialization belongs in `InteractiveBase`, not above it. Empirical evidence: (a) `InteractiveBase` (`ui/interactive_base.py:198-209`) already has `aget_input`/`asend_response` as overridable async wrappers with docstrings explicitly inviting subclass override; (b) 5 existing subclasses follow this pattern (rich, terminal, rich_terminal, queue, web); (c) `WebSocketInteractive` (`OpenStartup/server/services/websocket_interactive.py`) is the duck-typed outlier that should have inherited from day one; (d) existing `QueueInteractive` is a TRANSPORT (queue-as-IPC), orthogonal to the proposed in-process serializer — no naming collision. v1.4 design: (1) `WebSocketInteractive` inherits `InteractiveBase` (sync abstracts raise `NotImplementedError("use async methods")`); (2) `InteractiveBase` gains optional `_serializer: Optional[InteractionSerializer]` attribute; `enable_serialization(serializer)` activates it; (3) when active, `asend_response` enqueues + returns; `aget_input` awaits per-caller response via `ContextVar`-scoped caller identity; (4) `InteractionSerializer` is a single new class (NOT SOP-specific — reusable for any concurrent-caller scenario including BTA workers + background tasks); (5) caller identity flows implicitly via `asyncio.ContextVar` (no caller_id parameter pollution). Replaces 2 classes (RoutedInteractive + PendingConversationToolQueue) with 1 (InteractionSerializer). Strictly cleaner: zero-overhead when `_serializer=None`; SOPs share interactive directly; works beyond SOPs. Patches: §-1 audit row (V1); §1 architectural pin transport-layer note (V2); §9.11 fully rewritten (V3); §9.11.1-9.11.4 redesigned (V4-V6); §11 file inventory updated (V7-V9); §10 cross-ref for caller_id in turn logs (V10); AC9.11 tightened (V11); AC9.14 new for backward compat (V12); R18 mitigation reaimed (V13); R20 new for cross-package blast radius (V14); §12 effort table adjusted (V15). |

---

### §0. Why this plan exists

The `role_creation` SOP at `OpenStartup/.../prompt_templates/conversation/main/_variables/workflow_sop/role_creation.jinja2` is the canonical test bed. Today the SOP framework **cannot run it end-to-end** because:

1. The SOP has no skill-description-style activation surface (orchestrator can't match user intent to it).
2. The SOP parser doesn't understand `__keywords__` or `__example_requests__` meta-tags.
3. `ConversationalInferencer.yolo_mode` is declared at `conversational_inferencer.py:111` but **completely unused** — no synthetic auto-advance.
4. SOPs are orphaned variables (in `_variables/workflow_sop/`) rather than first-class registry entries.
5. There's no `<session>/sops/` runtime layout for SOP-as-conversational-sub-session.
6. Multi-SOP coordination is undesigned (queue, focus mode, tool-call attribution).
7. `ToolDefinition` has no concurrency labeling, so the runtime can't decide what needs queuing.

This plan addresses **all seven**, with empirical verification of every code citation.

---

### §1. Empirical baseline (verified 2026-05-26)

All claims verified by direct file inspection; line numbers are exact.

| Component | Path | Status & line |
|---|---|---|
| `SOPManager.parse_markdown` | `RichPythonUtils/.../template_manager/sop_manager.py` | ✅ EXISTS; parses phases, directives, requires_confirmation, depends_on, goto, foreach |
| `_REQUIRES_CONFIRMATION_RE` | sop_manager.py:91-93 | ✅ EXISTS; **requires whitespace** between words — `[__requires_confirmation__]` (underscore) silently skipped |
| `SOPManager.render_for_mode("yolo")` | sop_manager.py:567-584 | ✅ EXISTS; **dead code** — only caller is dead manager.py:174-177; **will be deleted** |
| `WorkflowRegistry.load_all()` | `AgentFoundation/.../common/workflow/registry.py:33-50` | ✅ EXISTS; `rglob`s `*.md` under `workflow_sop/` |
| `WorkflowManager.enter_workflow/exit_workflow/resume_workflow` | `AgentFoundation/.../common/workflow/manager.py:54-136` | ✅ EXISTS |
| `WorkflowManager.render_prompt_sections()` | manager.py:146-188 | ✅ EXISTS but **dead in OpenStartup** (factory at backends/factories.py:157-164 doesn't pass `workflow_manager=`) |
| `WorkflowInstance.yolo_mode` | `AgentFoundation/.../common/workflow/instance.py:26, 41, 55` | ✅ EXISTS; persisted but inert (never read in inferencer body) |
| `ConversationalInferencer.workflow_manager: Any` | conversational_inferencer.py:110 | ✅ EXISTS; default None |
| `ConversationalInferencer.yolo_mode: bool` | conversational_inferencer.py:111 | ✅ EXISTS; **NEVER READ** in body |
| `if self.workflow_manager is not None:` | conversational_inferencer.py:700-704 | ✅ EXISTS; gated dead code |
| `ConversationalInferencer.prior_context: dict[str, Any]` | conversational_inferencer.py:107 | ✅ EXISTS; the right surface for threading `sop_instance_id` |
| `ToolDefinition.asynchronous: bool` | `AgentFoundation/.../resources/tools/models.py:105` | ✅ EXISTS; "Fire-and-forget: tool runs in background, turn completes immediately" |
| Tool concurrency 3-value enum | (none) | ❌ MISSING; today `asynchronous: bool` collapses 2 of 3 cases |
| `PendingConversationToolQueue` | (none) | ❌ MISSING; no cross-SOP queue exists |
| `tool_call_parser.py` envelope format | `<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>` | ✅ verified; legacy `<Tools>` and `<ActionTools>` block forms also supported |
| `WorkflowContext.start_phase/complete_phase` | `AgentFoundation/.../server/workflow_context.py:99, 135, 170-218` | ✅ EXISTS; updates tracker via tool name → phase map |
| `WorkflowContext.tool_phase_map` | workflow_context.py:135 | ✅ EXISTS but **session-global, not per-instance** — must become per-instance for multi-focus |
| Session layout `<runtime>/servers/<server>/sessions/<sid>/{state.json, jsonl, turn_NNN/, tasks/}` | `OpenStartup/.../services/session_store.py:1-25, 215, 581` | ✅ EXISTS |
| `_runtime/sops/` (or singular) | (none) | ❌ MISSING |
| `resources/skills/` and `resources/tools/` | confirmed at `resources/{skills,tools}/` | ✅ BOTH use plural; `resources/sops/` (plural) matches convention |
| `_parse_skill_md` frontmatter parser | `resources/skills/registry.py:31` | ✅ EXISTS; reusable for SOP frontmatter |
| `tool.json` schema | many files under `resources/tools/*/tool.json` | ✅ JSON sidecar pattern is established |
| `JS is_auto_advance` round-trip | `OpenStartup/.../ui/src/hooks/useManagerChat.js:187-222`, `routes/manager_websocket_routes.py:354-355` | ✅ EXISTS; **NOT SOP-aware**; will be deprecated post-PR4 |
| `role_creation.jinja2:34` typo | uses `[__requires_confirmation__]` with underscore | 🐛 **BUG**; silently no-ops; fixed in PR-5 |
| `available_workflows` / `ongoing_workflows` Jinja slots | `prompt_templates/conversation/main/initial.jinja2:7-9, 13-15` | ✅ EXISTS but unpopulated; just need wiring |

**Architectural pin** (load-bearing for the rest of this plan; resolution of disagreement #1 below):

> **Active-SOP execution is tracker-driven, not WorkGraph-driven.** Each active SOP is a `WorkflowInstance` with its own `tracker: StateGraphTracker`, driven by **its own `SOPInferencer` (a `ConversationalInferencer` subclass spawned by the parent at `enter_sop` time, per §9.1 v1.2)**. The SOP inferencer issues tool calls (including `WorkflowContext.start_phase` / `complete_phase`) that advance its own tracker. The **parent** inferencer routes user input to the appropriate SOP inferencer (§9.4 hybrid routing); it does **not** drive trackers itself. The WorkGraph + per-phase inferencer machinery from v7.2 is **retained for `/sop --autonomous` end-to-end runs only**. The enter-and-stay-active path skips graph dispatch.

This decision is taken from external "v8" and refined by v1.2's subagent-model concretization. Rationale: linear SOPs (which is what `role_creation` is) don't need WorkGraph's concurrency; tracker + per-SOP inferencer is dramatically simpler; WorkGraph is the right tool when an SOP needs `__branch__` or `__for_each__` parallelism. v7.2 substrate isn't deleted — it's narrowed to one of two execution modes. See §9.1 for `SOPInferencer` definition, §9.11 for shared-interactive coordination, §9.12 for `run_one_turn` semantics, §9.13 for SOP-scoped prompt template.

**Transport-layer principle (v1.4 — V2):** any concurrent-callers-sharing-one-user-attention problem (multi-active SOP inferencers, autonomous WorkGraph parallel phases, BTA workers, background tasks) is solved at the **`InteractiveBase` transport layer** via `InteractionSerializer`, not at the SOP or BTA orchestrator layer. The serialization mechanism is opt-in (`interactive.enable_serialization(serializer)`); zero overhead and zero behavior change when not enabled. See §9.11 for the full design.

---

### §2. Three architectural disagreements between source plans — resolved

The 3 source plans disagreed on 3 architectural questions. Honest resolution table:

| # | Question | Plan #1 said | Plan #2 said | v8 said | **Unified decision** |
|---|---|---|---|---|---|
| 1 | Active-SOP execution substrate | WorkGraph (per v7.2) | (inherited v7.2) | **Tracker-driven** | **Tracker-driven inside each per-SOP `SOPInferencer`** for enter-and-stay-active (v1.2 subagent model — §9.1, §9.11); WorkGraph reserved for `/sop --autonomous`. Parent inferencer routes; child `SOPInferencer` drives its own tracker via tool calls. Rationale: simpler for linear SOPs; clean context isolation per SOP; WorkGraph available when truly needed. |
| 2 | Default focus mode | (not addressed) | `multi` (per Tony's 2026-05-26 16:11 direction) | `single_focus` (smaller prompt, predictable) | **`multi`** — honors Tony's stated direction. Cost of multi-as-default is mitigated by the conditional routing block (`only renders when len(active) >= 2`); single-active sessions pay zero overhead. v8's safety concern is captured as a kill-switch feature flag `OPENTEAM_SOP_FORCE_SINGLE_FOCUS`. |
| 3 | SOP config format | YAML frontmatter in SOP.md | (didn't take a stance) | JSON sidecar `sop.config.json` | **JSON sidecar `sop.config.json`** — matches existing `tool.json` convention; ergonomically distinct from SKILL.md's YAML frontmatter (different layer of the system); machine-validatable via JSON Schema. Plan #1's YAML-frontmatter call is reversed. |

Each decision is empirically defensible AND honors the user's stated direction where one was given.

---

### §3. Refined `role_creation` description (Concern #1)

Replaces current `role_creation.jinja2:1-3` (41 generic words). New description goes in **both** `sop.config.json.description` (primary) and as the first paragraph of `SOP.md` (fallback when sidecar absent):

> Provision a new AI employee end-to-end — from raw role description to deployed team member. Walks the user through (1) clarifying the role's responsibility categories, (2) generating a comprehensive role responsibility document via deep research, (3) decomposing the role into reusable skills + tools, (4) specializing the generic role for a specific team's Jira/Slack/Confluence context. Produces a versioned role document, a `final_deliverables/` skill+tool bundle, and a team deployment config.

**71 words; info-dense; names each phase + each deliverable.** Reads like a skill description, which is exactly its role — it becomes the SOP's row in the `## Available SOPs` prompt block and is what the LLM matches against user intent.

**AC0.1** Description ≤ 500 chars; contains no implementation-detail leakage (no mentions of `--yolo`, `tool_argument_form`, internal class names, or magic strings).
**AC0.2** Description begins with a verb ("Provision"); ends with a sentence naming deliverables.

---

### §4. SOP meta-tags `__keywords__` and `__example_requests__` (Concern #2)

#### §4.1 Sources of truth

Two sources, with explicit precedence:
1. **`sop.config.json` sidecar** — `keywords: list[str]`, `example_requests: list[str]` (primary; machine-readable; supports validation).
2. **In-markdown tags** — `[__keywords__]: ...` and `[__example_requests__]: ...` at top of SOP.md (fallback; ergonomic for quick edits; allows merge if sidecar declares `_merge_with_markdown: true`).

**Merge rule:** sidecar wins on collision unless `_merge_with_markdown: true` is set, in which case markdown values **append** (not replace) to sidecar values.

#### §4.2 Parser change in `SOPManager`

```python
# NEW: RichPythonUtils/.../template_manager/sop_manager.py
_KEYWORDS_RE = re.compile(
    r"^\[?__keywords__\]?\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_EXAMPLE_REQUESTS_RE = re.compile(
    r"^\[?__example_requests__\]?\s*:\s*(.+?)(?=^\[?__|^##|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
```

Both single-line (comma-separated) and bullet-list forms accepted:
```markdown
[__keywords__]: create role, new role, hire, onboard

[__example_requests__]:
- I want to create a new Program Manager role
- Hire a Data Scientist for the analytics team
```

#### §4.3 New fields on `SOP` class

```python
@attrs(slots=False, kw_only=True)
class SOP(StateGraph):
    # ... existing v7.2 fields ...
    keywords: list[str] = attrib(factory=list)          # NEW
    example_requests: list[str] = attrib(factory=list)  # NEW
```

**Per critical R5 audit on prior round:** these are **first-class fields, NOT a `meta: dict`**. Reasons: explicit attribute means IDE/typecheck support; parser dispatch can be type-strict; unknown meta-tags raise `SOPParseError` (fail-loud is safer than silent acceptance).

**AC4.1** SOP parser accepts both single-line and bullet-list forms.
**AC4.2** `sop.keywords` / `sop.example_requests` are populated correctly per the merge rule.
**AC4.3** Unknown meta-tag (e.g., `[__synonyms__]: foo`) raises `SOPParseError` with a clear message naming the unrecognized tag.

---

### §5. SOPs as first-class resources (Concern #4)

#### §5.1 Folder layout

```
AgentFoundation/src/agent_foundation/resources/sops/    # PLURAL (matches resources/skills/, resources/tools/)
├── __init__.py
├── registry.py                                        # new SOPRegistry (~80 LoC)
├── code_optimization/
│   ├── SOP.md                                         # moved from _variables/workflow_sop/code_optimization.md
│   └── sop.config.json
└── model_optimization/
    ├── SOP.md
    └── sop.config.json

OpenStartup/src/openteam/server/resources/sops/        # PLURAL
└── role_creation/
    ├── SOP.md                                         # moved + refined from role_creation.jinja2
    ├── sop.config.json
    └── references/                                    # optional deep-dive notes (mirrors skills/twg/references/)
        └── role_categories.md
```

OpenStartup wins on name collision (mirrors `load_all_skills(extra_dirs=...)` at `resources/skills/registry.py:99-115`).

#### §5.2 `sop.config.json` canonical schema

All fields optional except `name`:

```jsonc
{
  // identity
  "name": "role_creation",
  "display_name": "AI Role Creation",
  "version": "1.0.0",

  // activation surface (Concern #2)
  "description": "Provision a new AI employee end-to-end — …",  // ≤500 chars
  "keywords": ["create role", "new role", "hire", "onboard", "provision employee",
               "set up agent", "AI employee", "new AI hire"],
  "example_requests": [
    "I want to create a new Program Manager role",
    "Hire a Data Scientist for the analytics team",
    "Set up an SRE AI employee",
    "Onboard a customer support lead for the support team",
    "Create an AI assistant for backend reliability work"
  ],
  "labels": ["onboarding", "role-management", "ai-employees"],
  "_merge_with_markdown": false,

  // runtime
  "available_modes": ["default", "yolo"],
  "requires_tools": ["create_role", "role_setup", "team_onboard",
                     "multiple_choice", "single_choice", "confirmation", "clarification"],
  "max_goto_iterations": 5,
  "max_total_nodes": 100,
  "linear_only": true,                                  // if true, registry rejects __goto__/__branch__/__for_each__

  // yolo behavior (Concern #3)
  "yolo_overrides": {
    "multiple_choice": { "mode": "first_choice" },
    "single_choice":   { "mode": "first_choice" },
    "confirmation":    { "mode": "fixed", "value": "yes" },
    "clarification":   { "mode": "fixed", "value": "Follow your best judgment based on the role context." }
  },

  // persistence (Concern #5)
  "preserve_workspace": true,
  "checkpoint_on_phase_complete": true
}
```

#### §5.3 `SOPRegistry` — new file

Pattern: copy `resources/skills/registry.py` (`SkillInfo`, `load_skill`, `load_all_skills`, `format_all_skills`). Differences:
- (a) Read `sop.config.json` instead of YAML frontmatter (`json.loads(...)`).
- (b) Call `SOPManager.parse_markdown(body)` from RichPythonUtils to attach the parsed `SOP` AST.
- (c) Merge sidecar `keywords`/`example_requests` with in-markdown tags per §4.1.
- (d) Validate `linear_only`: if true and parsed SOP has any `__goto__`/`__branch__`/`__for_each__` → reject with clear error naming the offending phase.

```python
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
    config: dict                          # raw sop.config.json for runtime knobs
    body_path: Path
    body: str
    folder: Path
    sop: SOP                              # SOPManager.parse_markdown(body)

def load_sop(name: str, base_dir: Path) -> SOPInfo: ...
def load_all_sops(extra_dirs: list[Path] | None = None) -> dict[str, SOPInfo]: ...
def format_all_sops(sops: dict[str, SOPInfo]) -> str: ...
```

#### §5.4 Prompt wiring — three minimal edits

1. **Prompt template** `AgentFoundation/.../prompt_templates/conversation/main/initial.jinja2`: rename `## Available Workflows` (line 7-10) → `## Available SOPs`; rename `## Ongoing Workflows` (line 13-16) → `## Active SOPs`. The new `## Active SOPs` block loops over a list-of-dicts; rendering depth depends on mode (§9). Keep old variable names as Jinja aliases for one-release backward compat (`{% set available_sops = available_workflows %}` at top of include).

2. **OpenStartup factory** `OpenStartup/.../backends/factories.py:157-164`: pass `workflow_manager=` to `ConversationalInferencer`. The manager is built once per session:
```python
workflow_manager = WorkflowManager(
    registry=SOPRegistry().load_all(extra_dirs=[OPENSTARTUP_SOP_DIR]),
    session_workspace=<session_dir>,
    inferencer_factory=...,
    focus_mode=session_focus_mode_or_default(),  # §9
)
```
This single change activates the previously-dead `if self.workflow_manager is not None:` block at conversational_inferencer.py:700-704.

3. **`WorkflowManager.render_prompt_sections()`** at manager.py:146-188: change return shape from `dict[str, str]` to `dict[str, Any]` where `active_sops` is a list-of-dicts (one per active instance). **Drop the `mode = "yolo" if focused.yolo_mode` branch** (§6 makes this dead). Read `focus_mode` to decide per-active-SOP rendering depth.

**AC5.1** `SOPRegistry().load_all(extra_dirs=[OPENSTARTUP_SOP_DIR])` returns 3 entries (`role_creation`, `code_optimization`, `model_optimization`).
**AC5.2** Fresh OpenStartup session renders `## Available SOPs` block with all 3 entries; sending "hello" shows the block exactly once.
**AC5.3** Entering `role_creation` (via `enter_sop` action tool or `/sop role_creation` slash command) produces an `## Active SOPs` block; the `## Available SOPs` block remains visible.
**AC5.4** Name collision (e.g., `code_optimization` in both repos) logs a warning; OpenStartup wins.
**AC5.5** `sop.config.json` fields `max_goto_iterations` / `max_total_nodes` propagate to `WorkflowDefinition.frontmatter` and are honored by manager.py:77-78.
**AC5.6** `linear_only: true` SOP containing `__goto__` is rejected at registry load with error: `"SOP <name> is linear_only=true but contains __goto__ in phase <id>; remove the directive or set linear_only=false."`

#### §5.5 Migration

- Move `_variables/workflow_sop/{role_creation.jinja2, code_optimization.md, model_optimization.md}` to new layout. Drop `.jinja2` extension (these files don't contain Jinja syntax — they're pure markdown).
- Delete `_variables/workflow_sop/` directory after move.
- Delete `_variables/workflow_description/default.jinja2` after PR-1 ships (replaced by `WorkflowManager.render_prompt_sections()`).
- Delete `_variables/workflow/.sop.config.yaml` (replaced by per-SOP `sop.config.json`).

---

### §6. Yolo synthetic per-tool default responses (Concern #3)

**Goal:** When `yolo_mode` is on, the LLM still emits conversation tools as usual, but the inferencer auto-synthesizes the user's reply per tool type instead of blocking on `aget_input` (`conversational_inferencer.py:289-295`). The synthesized reply is fed back to the LLM identically to a human reply but is logged with `source: "synthetic"`.

#### §6.1 Per-tool `yolo_default` in `tool.json`

| Tool | yolo_default | File |
|---|---|---|
| `multiple_choice` | `{"mode": "select_all"}` | `AgentFoundation/.../resources/tools/multiple_choice/tool.json` |
| `single_choice` | `{"mode": "first_choice"}` | `.../single_choice/tool.json` |
| `confirmation` | `{"mode": "fixed", "value": "yes"}` | `.../confirmation/tool.json` |
| `clarification` | `{"mode": "fixed", "value": "Follow your best judgment."}` | `.../clarification/tool.json` |

#### §6.2 Mode enum — 6 modes (adopted from v8; strict superset of Plan #1)

| Mode | Applies to | Behavior |
|---|---|---|
| `fixed` | all four | Use literal `value` string as the synthesized response |
| `select_all` | multiple_choice | Select every offered choice |
| `first_choice` | single_choice, multiple_choice | Pick the first choice's `value` |
| `recommended` | single_choice, multiple_choice | Pick the choice flagged `"recommended": true` in the LLM-emitted choices list (falls back to `first_choice`) |
| `prompt_llm` | all four | Second low-cost LLM call (see §6.4) for context-aware answer |
| `none` | all four | No default; require human reply even in yolo mode (escape hatch) |

**Per-SOP override:** `sop.config.json.yolo_overrides{tool_type → {mode, value?}}` shadows the tool's `yolo_default` when that SOP is the active context for the call.

**Resolution order** (first hit wins):
1. Per-SOP `yolo_overrides[tool_type]` (from `sop.config.json` of the SOP attributed to the tool call)
2. Tool's `tool.json` `yolo_default`
3. Builtin fallback: `{"mode": "fixed", "value": "Follow your best judgment."}`

#### §6.3 Synthetic response format — verified against actual protocol

**Critical bug fix from Plan #1 v3.1:** earlier drafts used invented shapes like `{"selected": [...]}` and `{"confirmed": True}`. The **actual protocol** (verified at `_handle_conversation_tool` in conversational_inferencer.py and `_process_widget_response` in RichInteractiveBase) expects:

| Tool | Required response shape | Source key (in returned `{var_name: value}` dict) |
|---|---|---|
| `multiple_choice` | dict per tool's `output_variable`; value is comma-joined selected `choice` values, OR list of choice values, depending on tool config | `values: dict[str, str]` for compound dispatch |
| `single_choice` | string value of chosen option (NOT `{"choice_index": N}`) | `values: {var_name: value_str}` |
| `confirmation` | plain string `"yes"` or `"no"` | `values: {var_name: "yes" \| "no"}` |
| `clarification` | plain string (user's free-text response) | `values: {var_name: response_str}` |
| `tool_argument_form` | `{"param_overrides": dict}` | `values: {var_name: json.dumps(overrides)}` |

The compound `_handle_conversation_tools` (line 1099) returns `Optional[dict[str, str]]` where keys are `output_variable` names from each fired tool. The synthetic dispatcher must produce the same shape.

```python
def _synthesize_yolo_collected(self, tools: list[ConversationTool],
                               sop_attribution: Optional[str]) -> dict[str, str]:
    """Return {output_variable: value_str} for each fired conversation tool."""
    collected: dict[str, str] = {}
    for tool in tools:
        spec = self._resolve_yolo_spec(tool.tool_type, sop_attribution)  # §6.2 resolution order
        mode = spec["mode"]
        if mode == "fixed":
            collected[tool.output_variable] = spec["value"]
        elif mode == "select_all":
            # multiple_choice: all choice values, comma-joined
            collected[tool.output_variable] = ",".join(c["value"] for c in tool.choices)
        elif mode == "first_choice":
            collected[tool.output_variable] = tool.choices[0]["value"]
        elif mode == "recommended":
            recommended = next((c for c in tool.choices if c.get("recommended")), None)
            collected[tool.output_variable] = (recommended or tool.choices[0])["value"]
        elif mode == "prompt_llm":
            collected[tool.output_variable] = await self._yolo_prompt_llm(tool)  # §6.4
        elif mode == "none":
            raise YoloModeRequiresHumanResponse(tool=tool, message=
                f"Tool '{tool.tool_type}' has yolo mode='none'; cannot auto-advance.")
    return collected
```

#### §6.4 `prompt_llm` mode implementation

Triggers a single low-cost LLM call via `self.base_inferencer.ainfer(...)` with `temperature=0.2, max_tokens=200`.

New template `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/templates/yolo_synthesize.jinja2`:

```jinja
Given the conversation context and the orchestrator's question, produce the answer
that a thoughtful, decisive user would give. Match the question's expected format.

Conversation context (most recent turns):
{{ recent_history }}

Active SOP guidance (if any):
{{ active_sop_guidance }}

Question type: {{ tool_type }}
Question: {{ prompt }}
{% if choices %}Choices: {{ choices }}{% endif %}

Respond with ONLY the answer text. For choices, return the value string(s) comma-separated;
for confirmation, return "yes" or "no"; for clarification, return prose.
```

**Response parsing per tool type:**
- `multiple_choice` → comma-split, match values back to indices
- `single_choice` → value string
- `confirmation` → "yes"/"no" (defaults to "yes" on parse failure)
- `clarification` → raw text

**Caching:** keyed on `(tool_prompt_hash, last_3_turn_hashes)`; stored on `_dynamic_context` (cleared at session boundary). Avoids redundant calls on retry loops.

None of the four built-in tool defaults use `prompt_llm`; it's opt-in per-tool or per-SOP.

#### §6.5 Inner vs outer inferencer integration

**Critical wiring detail (carried over from Plan #2 v1.1):** there are two distinct inferencer instances in the live flow.

| Inferencer | Role | Method |
|---|---|---|
| **Outer** (user-facing, one per session) | Renders prompt; routes user response; handles yolo synthesis when no `aget_input` available | `_handle_conversation_tools` (modified to gate on `yolo_mode`) |
| **Inner** (per autonomous-SOP, only in `/sop --autonomous`) | Fires conversation tool inside `SOPWorkGraphNode._execute_phase`; enqueues + awaits via `PendingConversationToolQueue` | `_handle_conversation_tools` (calls into queue) |

**Outer inferencer** integration at `conversational_inferencer.py:289` (gate split):

```python
if conv_response.has_conversation_tool:
    if self.yolo_mode:
        # NEW: synthesize per §6.3
        sop_attribution = self._resolve_sop_attribution(conv_response)  # §9.4
        collected = await self._synthesize_yolo_collected(
            conv_response.conversation_tools, sop_attribution
        )
        if on_new_turn:
            await on_new_turn(turn_number, user_input=collected, source="synthetic")
    elif effective_interactive:
        collected = await self._handle_conversation_tools(...)  # existing path
    else:
        collected = None
```

**Inner inferencer** uses `self.prior_context.get("sop_instance_id")` to know which SOP to attribute its queue entries to (§8.4). `prior_context: dict[str, Any]` exists at conversational_inferencer.py:107; SOPWorkGraphNode sets it via `set_prior_context({..., "sop_instance_id": instance.id})` (line 525).

#### §6.6 Code changes

- `AgentFoundation/.../resources/tools/models.py` (`ToolDefinition.from_dict / to_dict`): parse + emit `yolo_default`. Today unknown JSON keys are silently dropped — need explicit field handling. New attribute: `yolo_default: Optional[dict] = None`.
- `AgentFoundation/.../inferencers/agentic_inferencers/conversational/conversational_inferencer.py:289` — gate split per §6.5.
- New method `_synthesize_yolo_collected` (~50 LoC; §6.3).
- New method `_resolve_yolo_spec(tool_type, sop_attribution)` — resolution order per §6.2.
- New method `_yolo_prompt_llm(tool)` — §6.4 LLM call with caching.
- New method `_resolve_sop_attribution(conv_response)` — reads `prior_context["active_sops"]` and the LLM's per-tool `sop_instance_id` field (§9.4) to determine the right SOP override map.

#### §6.7 Delete legacy

- `SOPManager.render_for_mode` body in `RichPythonUtils/.../sop_manager.py:567-584` — yolo no longer manipulates SOP markdown text.
- `WorkflowManager.render_prompt_sections:174-177` — the dead `mode = "yolo"` branch.

The `[__requires confirmation__]` tag stays in SOP markdown as LLM-readable documentation; the parsed `SOPPhase.requires_confirmation` bool is still useful for status-text rendering at `workflow_context.py:431-440`.

#### §6.8 Acceptance criteria

**AC6.1** Each of the four conversation `tool.json` files has a `yolo_default` field validated by `ToolDefinition.from_dict`.
**AC6.2** With `yolo_mode=True`, the LLM emitting a `multiple_choice` tool produces an immediate synthetic answer (no widget shown to UI, no `aget_input` block); the answer feeds back into the LLM next iteration identically to a human reply.
**AC6.3** Synthetic turns are persisted with `source: "synthetic"` in `turn_NNN/metadata.json` and in `session.jsonl` UserInput record.
**AC6.4** `SOPManager.render_for_mode` body is removed; SOP markdown renders as-is regardless of yolo mode.
**AC6.5** End-to-end yolo run of `role_creation` SOP completes Phase 0 → Phase 3 with zero human input.
**AC6.6** `prompt_llm` mode opt-in via per-tool `tool.json` or per-SOP `yolo_overrides` produces a context-coherent answer (asserted by example: ask "Which scope?" after user said "Program Manager for the analytics team" earlier → answer references "analytics" or "Program Manager").
**AC6.7** `none` mode falls through to human interactive widget even when `yolo_mode=True` (escape hatch verified).
**AC6.8** Resolution order verified: per-SOP override → `tool.json` default → builtin fallback (assert with a SOP whose override is `recommended` for `single_choice`).

#### §6.9 Prerequisite: `recommended` field on `ChoiceItem` (v1.1)

**Empirical fact (verified 2026-05-26):** the current `ChoiceItem` dataclass at `conversation_tools.py:26-46` has only `label`, `value`, `description` fields. `from_dict` silently drops any `recommended` key the LLM might emit. No `tool.json` example mentions `recommended`, so the LLM has no in-prompt signal to emit it. The `recommended` yolo mode in §6.2 therefore requires 4 prerequisite changes before it can work.

##### §6.9.1 Schema + LLM-prompt signal (`tool.json`)

```jsonc
// resources/tools/single_choice/tool.json (and multiple_choice/tool.json)
"parameters": [
  {"name": "prompt", "type": "string", "required": true, ...},
  {"name": "choices", "type": "string", "required": true,
   "description": "Array of {label, value, description, recommended?} objects. Set `recommended: true` on at most one choice (single_choice) or any subset (multiple_choice) to flag it as the suggested default."},
  {"name": "allow_custom", "type": "flag", "default": true, ...}
],
"examples": [
  "{\"tool_type\": \"single_choice\", \"prompt\": \"Which optimization strategy?\", \"choices\": [{\"label\": \"Memory\", \"value\": \"memory\", \"description\": \"Reduce memory footprint\"}, {\"label\": \"Latency\", \"value\": \"latency\", \"description\": \"Reduce inference time\", \"recommended\": true}], \"allow_custom\": true}"
]
```

##### §6.9.2 Parser change — `conversation_tools.py:26-46`

```python
@dataclass
class ChoiceItem:
    """A single choice option for single/multiple choice tools."""

    label: str
    value: str
    description: str = ""
    recommended: bool = False                          # NEW (v1.1)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "value": self.value}
        if self.description:
            d["description"] = self.description
        if self.recommended:                           # NEW: only emit when True (compact JSON)
            d["recommended"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceItem:
        return cls(
            label=data.get("label", ""),
            value=data.get("value", ""),
            description=data.get("description", ""),
            recommended=bool(data.get("recommended", False)),  # NEW
        )
```

##### §6.9.3 Resolver — `_synthesize_yolo_collected` `recommended` mode (typed access)

```python
elif mode == "recommended":
    recommended = next((c for c in tool.choices if c.recommended), None)
    collected[tool.output_variable] = (recommended or tool.choices[0]).value
```

Note: typed `c.recommended` (boolean attribute), not `c.get("recommended")` — `ChoiceItem` is a dataclass post-parse, not a dict.

##### §6.9.4 UI surface (optional but valuable for non-yolo human flow)

In OpenStartup React choice widgets (`MultipleChoice*.jsx` / `SingleChoice*.jsx`):

```jsx
{choice.recommended && <span className="badge-recommended" title="LLM recommended">⭐ Recommended</span>}
```

This lets human users see what the LLM is recommending even when not in yolo mode — useful context for human decision-making.

##### §6.9.5 ACs

**AC6.9** `ChoiceItem.from_dict({"label": "A", "value": "a", "recommended": True})` produces `ChoiceItem(label="A", value="a", recommended=True)`; missing `recommended` defaults to `False`.
**AC6.10** `ChoiceItem(label="A", value="a", recommended=True).to_dict() == {"label": "A", "value": "a", "recommended": True}`; `.to_dict()` for non-recommended choice omits the key (compact).
**AC6.11** `recommended` yolo mode picks the flagged choice when exactly one is flagged; falls back to `first_choice` semantics when none flagged; in `single_choice` with multiple flags, picks first in choice order + logs warning per R17.
**AC6.12** Updated `tool.json` examples are parsed by `ToolDefinition.from_dict` without error; LLM running on the new prompt emits a `recommended: true` choice in its multiple_choice output (verified against mock LLM).

##### §6.9.6 New RED test

Added as test #15 in §11.1: `test_choice_item_recommended_field.py` — parse/serialize round-trip + resolver behavior + LLM-emit-with-new-examples integration.

---

### §7. `ToolDefinition.concurrency` — 3-value enum (Concern #7)

#### §7.1 New field

```python
@dataclass
class ToolDefinition:
    # ... existing fields ...
    concurrency: Literal["blocking", "async_background", "async_awaitable"] = "blocking"
    yolo_default: Optional[dict] = None  # §6.1
    # Backward compat:
    # - asynchronous: bool   (deprecated; derived)

    @property
    def asynchronous(self) -> bool:
        """DEPRECATED — use concurrency."""
        return self.concurrency != "blocking"
```

#### §7.2 Semantics

| Value | Meaning | Turn semantics | Examples |
|---|---|---|---|
| `"blocking"` | Must complete before turn advances | Agentic loop awaits result | All conversation tools today; fast sync actions |
| `"async_background"` | Fire-and-forget; turn advances immediately; result delivered via callback or polled later | Loop continues; result joins via separate mechanism | `research_propose` (today: `asynchronous=True`); long-running subagents |
| `"async_awaitable"` | Concurrently runnable within a turn; loop fans out via `asyncio.gather`; turn advances when all complete | Loop awaits gather | NEW — not exercised today; future fan-out (e.g., parallel API calls in one turn) |

#### §7.3 Backward-compat migration

- Existing `tool.json` files with `"asynchronous": true` → derived as `"concurrency": "async_background"`
- Existing `tool.json` files without `"asynchronous"` → derived as `"concurrency": "blocking"`
- New `tool.json` should use `"concurrency"` explicitly
- `ToolDefinition.from_dict()` accepts both forms; emits `DeprecationWarning` on `"asynchronous"` in 2 releases

#### §7.4 Conversation tools

All conversation tools (`clarification`, `confirmation`, `multiple_choice`, `single_choice`, `tool_argument_form`) keep `concurrency: "blocking"` — they fundamentally require user response before the turn can complete.

#### §7.5 Interaction with multi-focus

Critical: only `blocking` tools need cross-SOP queuing.

| Concurrency | Multi-focus implication |
|---|---|
| `blocking` (conversation tools) | **MUST queue** across SOPs — shared user attention (only one prompt at a time) |
| `async_background` (research_propose et al) | **No queue needed** — each SOP's WorkGraph asyncio.Task runs the tool independently; results route back via the SOP's own task |
| `async_awaitable` (future) | **No cross-SOP queue needed** — each SOP gathers its own concurrent calls |

#### §7.6 ACs

**AC7.1** `ToolDefinition.concurrency` round-trips correctly from `tool.json` (both new `concurrency` and legacy `asynchronous` accepted; legacy emits `DeprecationWarning`).
**AC7.2** All existing conversation tools (`clarification`, `confirmation`, etc.) load with `concurrency == "blocking"` by default.
**AC7.3** Existing `research_propose` (today `asynchronous: True`) loads with `concurrency == "async_background"` via backward-compat derivation.
**AC7.4** A new `async_awaitable` tool can be loaded; integration test verifies `asyncio.gather` semantics (deferred — no v1 tool uses this).

---

### §8. `PendingConversationToolQueue` (only needed in `/sop --autonomous` + multi-focus)

#### §8.0 Ownership (v1.2 — P5; superseded by v1.4 §9.11)

**Status (v1.4):** the §8 "PendingConversationToolQueue" concept is **superseded** by the v1.4 `InteractionSerializer` (§9.11.3), which lives in `InteractiveBase` and serves the same purpose with a cleaner transport-layer design. **§8 remains documented here as historical context for the `--autonomous` mode** (where the WorkGraph path also benefits from interaction serialization across multiple per-phase inferencer tasks), but the unified solution is the `InteractionSerializer` — *one* serialization mechanism for all concurrent-caller scenarios (enter-and-stay-active multi-SOP, `--autonomous` WorkGraph parallel phases, BTA workers, etc.). New implementations should attach an `InteractionSerializer` to the shared `interactive` via `enable_serialization()`; they should NOT build a separate queue inside any inferencer or session coordinator.

The earlier v1.2/v1.3 framing of "parent owns the queue" is correct in spirit — under v1.4, the parent's session coordinator owns the `InteractionSerializer` instance and attaches it to the shared `interactive`. The mechanics differ (serializer is at the transport layer, not at the SOP coordinator layer), but the ownership boundary is the same.

#### §8.1 When the queue is needed

- **Enter-and-stay-active path, single-focus mode OR multi-focus with `len(active) <= 1`:** Only one `SOPInferencer` runs at a time. **Queue NOT needed** in this path.
- **Enter-and-stay-active path, multi-focus mode with `len(active) >= 2`** *(v1.3 → v1.4 redesigned)*: N `SOPInferencer` instances share the parent's `interactive` directly (no proxy class needed). The `ConversationService` attaches an `InteractionSerializer` (§9.11.3) to the shared `interactive` via `enable_serialization()`. Each SOP's `aget_input()`/`asend_response()` calls route through the serializer (single widget visible at a time) via `_CURRENT_INTERACTION_CALLER` ContextVar caller identity. **Serialization IS needed** here, regardless of `--autonomous`. This is the cross-SOP-inferencer serialization case — the `InteractionSerializer` is the unified mechanism for it and for the `--autonomous` parallel-phase case.
- **`/sop --autonomous` end-to-end run path (WorkGraph substrate):** each autonomous SOP runs its own inner inferencer in an asyncio.Task; multiple SOPs can fire conversation tools concurrently. **Queue REQUIRED** here.

*(v1.3)* The earlier framing "queue is autonomous-mode-only" was wrong — v1.2's subagent model means multi-focus enter-and-stay-active mode also needs the queue when ≥2 SOPs are active. The queue's role is more accurately: **"required whenever ≥2 inferencer instances share a single `interactive`"** — which happens in both autonomous mode AND multi-focus enter-and-stay-active mode.

#### §8.2 Single-head + multi-display semantics

Per Tony's direction (2026-05-26 16:14):
- **Single-head:** Only one tool is actively bound to user attention at any time — the queue head.
- **Multi-display:** The prompt's "AWAITING RESPONSE" line lists ALL P0 tools so the LLM and user know what's coming.
- When the user responds: LLM routing decision says which SOP the response is for; if it's the head, deliver and pop; if it's a non-head, **reorder the queue** (move that tool to head) and re-present.

#### §8.3 API

```python
@dataclass
class PendingTool:
    instance_id: str              # which WorkflowInstance fired this
    tool: ConversationTool
    future: asyncio.Future        # the WorkGraph node awaits this
    enqueued_at: datetime
    prompt_excerpt: str           # for "AWAITING RESPONSE" display

class PendingConversationToolQueue:
    """Per-session queue of blocking conversation tools across concurrent autonomous SOPs."""

    _queue: deque[PendingTool]
    _workflow_manager: "WorkflowManager"  # back-ref for last_user_input_advanced_at writes (§9.6)

    async def enqueue(self, instance_id: str, tool: ConversationTool) -> asyncio.Future: ...
    def head(self) -> Optional[PendingTool]: ...
    def all(self) -> list[PendingTool]: ...
    def reorder_to_head(self, instance_id: str) -> None: ...

    def deliver(self, instance_id: str, response: str) -> None:
        if not self._queue or self._queue[0].instance_id != instance_id:
            self.reorder_to_head(instance_id)
        head = self._queue.popleft()
        # writer for P1 priority + clear P0 marker
        instance = self._workflow_manager.active_instances.get(instance_id)
        if instance:
            instance.last_user_input_advanced_at = datetime.utcnow()
            instance.pending_conversation_tool = None
        head.future.set_result(response)
```

#### §8.4 Integration with inner inferencer

The inner inferencer (running inside `SOPWorkGraphNode._execute_phase`) calls into the queue:

```python
async def _handle_conversation_tools_inner(self, tools, ...) -> Optional[dict[str, str]]:
    if self.yolo_mode:
        return self._synthesize_yolo_collected(tools, sop_attribution=self.prior_context.get("sop_instance_id"))

    instance_id = self.prior_context.get("sop_instance_id")  # §6.5
    if self.workflow_manager and instance_id:
        collected = {}
        futures = []
        for tool in tools:
            f = await self.workflow_manager.pending_tool_queue.enqueue(instance_id, tool)
            futures.append((tool.output_variable, f))
        for var_name, f in futures:
            collected[var_name] = await f
        return collected

    return await self._original_interactive_handler(tools)
```

#### §8.5 ACs

**AC8.1** Two autonomous SOPs fire conversation tools in the same agentic-loop iteration: both are enqueued; only one is bound to user attention; the other's WorkGraph node awaits its future without racing.
**AC8.2** LLM routes user response to non-head queued tool: queue reorders; new head is presented; original head moves to position 1.
**AC8.3** On `deliver(instance_id, response)`: matching `WorkflowInstance.pending_conversation_tool` is cleared; `last_user_input_advanced_at` is set; SOP demotes from P0.
**AC8.4** Queue is per-session (not global); 2 sessions in same server have independent queues.
**AC8.5** *(v1.3 corrected)* Queue is NOT instantiated when `focus_mode="single"` OR (`focus_mode="multi"` AND `len(active_instances) <= 1`) — verified by `parent_inferencer.pending_tool_queue is None`. Queue IS instantiated as soon as a second `SOPInferencer` becomes active under `multi` mode (parent lazily creates it; reset to `None` when active count drops back to ≤1).

---

### §9. Mode system — single-focus vs multi-focus (Concern #6)

#### §9.1 Mode definitions

**`multi` mode (default per Tony's direction 2026-05-26 16:11):**
- N SOPs can be `status="active"` simultaneously. **Each runs as its own `SOPInferencer` instance** — a subclass of `ConversationalInferencer` spawned by the parent inferencer at `enter_sop` time (the subagent pattern; precedent: `BreakdownThenAggregateInferencer._iter_child_inferencers` at `breakdown_then_aggregate_inferencer.py:1168`). Each `SOPInferencer` has its own `messages` list, its own `prior_context` (SOP-scoped: `{phase_id, sop_meta, sop_instance_id, ...}`), and its own `_tracker: StateGraphTracker`. The parent inferencer remains the user-facing chitchat + routing surface; SOP inferencers are dispatched into by the parent. See §9.11 for shared-interactive coordination and §10 for per-SOP message-log persistence.
- No auto-suspension on `enter_sop`.
- Prompt renders all active SOPs with **priority-based budget allocation** (§9.3).
- Routing: when `len(active_instances) >= 2`, the prompt includes a routing instruction; the LLM emits a tool call with per-call `sop_instance_id` attribution (§9.4) and/or an explicit routing tool call.
- **Conditional cost:** when `len(active) <= 1`, the routing block is NOT rendered. Single-active sessions pay zero overhead.

**`single` mode:**
- At most 1 SOP can be `status="active"` at a time.
- Entering a new SOP via `enter_sop(new_id)` **auto-suspends** the currently-active one (calls `suspend_workflow`, NOT `exit_workflow` — suspended SOPs are resumable).
- Prompt rendering: one full block for the active SOP; suspended SOPs as one-line P3 entries.
- Routing: implicit — user input always belongs to the (single) active SOP.

**Kill-switch:** `OPENTEAM_SOP_FORCE_SINGLE_FOCUS=true` env var coerces `focus_mode` to `"single"` regardless of session/server config. Captured as `EnvOverride` event in `session.jsonl` for observability.

#### §9.2 Configuration sources (in resolution order; first hit wins)

1. Session-init payload: `{"focus_mode": "single" | "multi"}` (per-session override; persisted in `session_state.json`)
2. Env kill-switch `OPENTEAM_SOP_FORCE_SINGLE_FOCUS=true` → forces `"single"`
3. Server-side default in `sop_config.json` (per-deployment default; defaults to `"multi"`)
4. Hardcoded fallback: `"multi"`

#### §9.3 Priority-based rendering (`multi` mode only, only when `len(active) >= 2`)

| Priority | Condition | Rendering | Budget |
|---|---|---|---|
| **P0** | `pending_conversation_tool is not None` (a tool fired and is awaiting user response — only relevant in autonomous mode; in tracker-driven mode the rendering uses the current phase's pending state) | FULL context + highlighted "AWAITING RESPONSE: <tool_type>: '<prompt_excerpt>'" | Up to `max_p0_full_render` (default 3); excess P0 demote to compact with warning |
| **P1** | `last_user_input_advanced_this_sop is True` (the most recent user-input turn advanced this SOP — event-driven, not wall-clock) | FULL context | Share remaining budget proportionally |
| **P2** | Active but neither P0 nor P1 (running, e.g., long-async tool in progress, or just-entered) | COMPACT — name + phase + one-line status + `is_computing: bool` flag | One-line per SOP |
| **P3** | `status="suspended"` | ONE-LINE — `name (instance_id) — suspended; resume with resume_workflow('id')` | One-line per SOP |

**Deterministic budget algorithm:**

```python
def allocate_budget(sops: list[WorkflowInstance], total_budget: int = 2000,
                    max_p0_full_render: int = 3) -> dict[str, int]:
    p0 = [s for s in sops if s.priority == "P0"]
    p1 = [s for s in sops if s.priority == "P1"]
    p2 = [s for s in sops if s.priority == "P2"]
    p3 = [s for s in sops if s.priority == "P3"]

    # P0 cap: first `max_p0_full_render` by enqueued_at get FULL; rest COMPACT with warning
    p0_full = sorted(p0, key=lambda s: s.pending_conversation_tool["enqueued_at"])[:max_p0_full_render]
    p0_demoted = [s for s in p0 if s not in p0_full]
    if p0_demoted:
        log_warning(f"P0 cap reached: {len(p0_demoted)} SOP(s) demoted to compact; "
                    f"consider suspending less-urgent SOPs")
    p0_alloc = {s.id: estimate_full_context_tokens(s) for s in p0_full}
    p0_alloc.update({s.id: COMPACT_TOKEN_BUDGET for s in p0_demoted})
    used = sum(p0_alloc.values())
    remaining = max(0, total_budget - used)

    if p1 and remaining > 0:
        per_p1 = remaining // len(p1)
        p1_alloc = {s.id: per_p1 for s in p1}
    else:
        p1_alloc = {s.id: COMPACT_TOKEN_BUDGET for s in p1}

    return {**p0_alloc, **p1_alloc,
            **{s.id: COMPACT_TOKEN_BUDGET for s in p2},
            **{s.id: ONE_LINE_TOKEN_BUDGET for s in p3}}
```

`total_budget` and `max_p0_full_render` configurable in `sop_config.json`.

#### §9.4 Tool-call attribution + routing location (v1.2 clarified)

**Routing lives in the parent inferencer, NOT in any individual SOP inferencer** (v1.2 — P3). The flow:

1. User input arrives at the **parent** `ConversationalInferencer`.
2. Parent applies a **hybrid routing strategy**:
   - **(i) Rule-based fast path:** if exactly one active SOP has a pending conversation tool (`pending_tool_in_queue[sop_id] is not None`), the user input is delivered to that SOP's pending tool's queue slot. No LLM call. Zero overhead.
   - **(ii) Structural fast path:** if the user input is a slash-prefixed routing override (`/sop sop-abc12`), parent dispatches directly to that SOP inferencer.
   - **(iii) LLM-based fallback:** otherwise, parent's LLM call sees the routing block (rendered when `len(active) >= 2`) and emits a tool call. Attribution mechanism below.
3. Once routed, the chosen `SOPInferencer.run_one_turn(user_input)` advances exactly that SOP. Other SOP inferencers remain in `awaiting_response` state.
4. SOP inferencer's emitted conversation tools route through the `InteractionSerializer` attached to the shared `interactive` (§9.11). Serializer ensures one widget visible at a time across all callers; per-caller identity from `_CURRENT_INTERACTION_CALLER` ContextVar (set by parent before dispatching to each SOP inferencer).

**Why this matters:** the LLM never sees N SOP contexts merged into one prompt — it sees only the parent's routing summary + (after dispatch) the chosen SOP's full context. This is fundamentally cheaper and clearer than the "single inferencer with concatenated SOP blocks" framing v1.0 implied.

**Two-LLM-call cost** *(v1.3 — Q8 disclosure)*: when **route (iii) LLM-based fallback** fires (multi-focus, ≥2 active SOPs, no pending tool to match, no slash override), a user message incurs **2 LLM calls** per turn — one in the parent (routing), one in the chosen SOP (response generation). The rule-based (i) and structural (ii) fast paths cost **0 parent LLM calls** (just lookup + dispatch). Empirically: a single P0 SOP awaiting a confirmation always takes the fast path; full LLM routing engages only when 2+ SOPs are in ambiguous states. Captured in §13 as R19 (cost observability requirement).

##### Tool-call attribution mechanism (unchanged from v1.0)

**Two parallel mechanisms** (LLM may use either; runtime accepts both):

**Mechanism A (per-tool-call inline):** the LLM adds a top-level `sop_instance_id` field to each tool-call object:
```json
{"name": "role_setup", "arguments": {...}, "sop_instance_id": "a1b2c3d4"}
```

**EMPIRICAL VERIFICATION REQUIRED — PHASE 0 PREREQUISITE:** the current parser at `tool_call_parser.py` parses JSON-blob tool calls; if it preserves unknown top-level fields on the parsed object, mechanism A works as-is. If not, the parser needs a 1-line patch to retain unknown fields on `ParsedToolCall`. **The first test in Phase 0 must verify this.**

**Mechanism B (explicit routing tool call):** the LLM emits a dedicated routing call alongside the user-response tool calls:
```
<tool_call>{"name": "workflow_route", "arguments": {"ids": ["<instance_id>" | "none" | "<id_a>", "<id_b>"]}}</tool_call>
```

`workflow_route` is registered as a special pseudo-tool handled in-process by the inferencer; it does NOT dispatch to an external executor.

**Pre-dispatch parser hook** (peels off both mechanisms before regular dispatch):
```python
def split_routing_from_tools(parsed_calls: list[ParsedToolCall]) -> tuple[
        Optional[ParsedToolCall], dict[str, str], list[ParsedToolCall]]:
    """Find workflow_route (Mechanism B); collect per-call sop_instance_id (Mechanism A); return (route_call, per_call_attribution, remaining)."""
    routing = next((c for c in parsed_calls if c.name == "workflow_route"), None)
    per_call = {c.raw_id: c.extra.get("sop_instance_id")
                for c in parsed_calls
                if c.name != "workflow_route" and c.extra.get("sop_instance_id")}
    remaining = [c for c in parsed_calls if c.name != "workflow_route"]
    return routing, per_call, remaining
```

**Fallback logic:** if neither mechanism produces routing and `len(active) == 1`, attribute to the single active SOP. If neither mechanism produces routing AND `len(active) >= 2`, log warning + attribute to most-recently-progressed SOP + increment `unattributed_routing_fallback_count` metric.

#### §9.5 Per-instance `tool_phase_map`

Today `tool_phase_map` is session-global (`workflow_context.py:135`). For multi-focus to work correctly when two SOPs share a tool name (e.g., both have a `confirmation` tool that advances different phases), it must become per-instance:

```python
# workflow_context.py — modified
class WorkflowContext:
    tool_phase_maps_by_instance: dict[str, dict[str, str]]  # NEW: per-instance maps
    # legacy: tool_phase_map -> dict_view of focused instance for back-compat

    def start_phase(self, tool_name: str, instance_id: Optional[str] = None) -> None:
        if instance_id is None:
            instance_id = self._infer_instance_id_from_focus_or_attribution()
        phase_id = self.tool_phase_maps_by_instance[instance_id][tool_name]
        # ... rest unchanged
```

**Breaking change risk** — feature-flag this refactor (`OPENTEAM_PER_INSTANCE_TOOL_PHASE_MAP=true`) until parity verified. In single-focus mode and back-compat mode, the legacy single-map path is preserved.

#### §9.6 New `WorkflowInstance` fields

```python
@attrs(slots=False, kw_only=True)
class WorkflowInstance:
    # ... existing v7.2 fields ...
    pending_conversation_tool: Optional[dict] = None       # {tool_type, prompt_excerpt}; set by queue (autonomous only)
    last_user_input_advanced_at: Optional[datetime] = None # for P1 priority; written by queue.deliver() and routing handler
    is_computing: bool = False                             # true while an async_background tool is in flight
    focused: bool = False                                  # single-focus marker; advisory in multi mode
    sop_run_folder: Optional[str] = None                   # set by SOPSession.allocate(); recorded in session_state
```

#### §9.7 Mid-session toggle

Allowed via the explicit `set_sop_mode` action tool, UI button, or `/sop --mode <name>` slash command.

| From → To | Action | Suspended SOPs |
|---|---|---|
| single → multi | All current `active` and `suspended` stay as-is. Prompt re-renders with multi layout next turn. | Unchanged |
| multi → single | Pick the focus winner: `P0` → else `most recently advanced` → else `first by creation_ts`. Auto-suspend others. | Newly suspended (the non-winners) |

**Mode-switch notification** — next turn's prompt preamble includes:

```
[Mode switch] Focus mode is now {mode}. {role_creation} is active{; <others> are suspended (resume with `resume_workflow('id')`)}.
```

`session_state.json` persists `focus_mode` so resumed sessions restore correct mode.

#### §9.8 Routing scenarios (formalized as ACs)

| Scenario | Active SOPs | User input | Expected routing |
|---|---|---|---|
| A — Clear (single P0) | role_creation (P0: pending confirmation), code_opt (P2 running) | "yes, looks good" | Attribute to role_creation_<id> |
| B — Ambiguous (multi P0) | role_creation (P0), code_opt (P0) | "yes" | LLM responds with plain-prose disambiguation question; no tool call dispatched |
| C — New SOP entry | role_creation (P1 active in Phase 1) | "also optimize the pipeline at src/pipeline/" | `enter_sop("code_optimization")` → 2 active SOPs |
| D — Meta-query | (any) | "what's the status of all my workflows?" | No SOP attribution; LLM summarizes from rendered Active SOPs section |
| E — Invalid routing ID | role_creation (P1) | (LLM hallucinates ID) | Runtime validates `sop_instance_id` against `active_instances`; on miss, falls back to focus-winner OR most-recently-progressed + logs `routing_valid: false` |
| F — Single-active in multi mode | role_creation only | "yes" | No routing block in prompt (because `len(active) == 1`); user input goes directly to role_creation |

#### §9.9 Routing audit trail

Each turn appends a `RoutingDecision` event to `session.jsonl`:

```jsonc
{"type": "RoutingDecision", "turn": 7, "ts": "...",
 "active_sop_count": 3,
 "pending_tools_count": 1,
 "llm_routed_to": ["role_creation_abc12345"],
 "mechanism": "inline_field" | "workflow_route" | "fallback_focused" | "fallback_most_recent",
 "routing_valid": true,
 "routing_fallback": null}
```

#### §9.10 ACs

**AC9.1** Default `focus_mode == "multi"` on a fresh session unless overridden.
**AC9.2** With `focus_mode == "multi"` and exactly 1 active SOP, the rendered prompt contains NO routing block (zero token cost).
**AC9.3** With `focus_mode == "multi"` and 2+ active SOPs, the rendered prompt contains the routing block.
**AC9.4** Scenarios A, C, D, E, F per §9.8 — LLM's routing behavior matches expected.
**AC9.5** Scenario B (ambiguous multi-P0): LLM emits plain-prose disambiguation question; both SOP futures remain unresolved.
**AC9.6** `set_sop_mode("single")` mid-session: most-recent-P0/P1 stays active; others become suspended; next-turn prompt has mode-switch notification.
**AC9.7** `OPENTEAM_SOP_FORCE_SINGLE_FOCUS=true` coerces all sessions to single mode; `EnvOverride` recorded in `session.jsonl`.
**AC9.8** With 5 P0 SOPs (`max_p0_full_render` default = 3): first 3 by `enqueued_at` get FULL rendering; SOPs #4 and #5 demote to COMPACT with warning.
**AC9.9** Per-instance `tool_phase_map` correctly routes: two SOPs sharing tool name `confirmation` advance their own phases without cross-contamination (gated behind `OPENTEAM_PER_INSTANCE_TOOL_PHASE_MAP=true`).
**AC9.10** Phase 0 prerequisite: parser preservation of unknown `sop_instance_id` field on tool calls is empirically verified before Mechanism A is shipped.
**AC9.12** *(v1.3 — Q4/Q6; v1.4 reference updated)* `SOPInferencer.run_one_turn(user_input)` invokes inherited `run_agentic_loop(content=user_input)` exactly once, scoped to this SOP's `_tracker`. It returns when either (a) a conversation tool is emitted (enqueued via `InteractionSerializer.enqueue_send` on the shared interactive — §9.11), or (b) the agentic loop terminates naturally. After return, the parent invokes `SOPInferencer.check_completion()` which calls `self._tracker.is_completed()`; if True, parent invokes `WorkflowManager.complete_workflow(instance_id)` to set `status="completed"`, persist `<session>/sops/<sop_id>__TS__uuid/state.json`, and emit a `WorkflowCompleted` event to `session.jsonl`.
**AC9.13** *(v1.3 — Q5)* `SOPInferencer.prompt_renderer` uses `prompt_templates/sop/main/initial.jinja2` (NOT the parent's `prompt_templates/conversation/main/initial.jinja2`). Template includes: `{sop_description}`, `{tracker.phase_summary()}`, `{phase_specific_action_tools}`, `{sop_scoped_messages}`. Does NOT include: `{available_sops}`, `{ongoing_sops_routing_block}` (these belong to the parent's prompt only).
**AC9.11** *(v1.2 → v1.4 V11 retargeted)* Three active SOPs each emit a `confirmation` conversation tool within the same parent-inferencer turn. The shared `InteractionSerializer` attached via `interactive.enable_serialization()` (§9.11.1, §9.11.4) serializes them. The user sees **exactly one widget at a time** on the shared `interactive`. The second widget appears only after the user responds to the first; the third only after the second. Each `SOPInferencer`'s `_CURRENT_INTERACTION_CALLER` ContextVar is set correctly so `await_response_for_caller()` returns the right user response to the right caller. No `_current_input_mode` clobbering; no deadlock; no cross-SOP message bleed (each SOP inferencer's `messages` log records only its own confirmation exchange).
**AC9.15** *(v1.5 — V17 bridge correctness)* With `InteractionSerializer` attached to a `WebSocketInteractive` and 3 concurrent `aget_input` callers, the bridge mechanism (§9.11.2.1) drains `input_queue` correctly: (a) the first user response delivered via WS route handler → `input_queue` → first available `_bridge_one_response` task → `serializer.deliver_response()` → resolves the head-of-queue caller's future; (b) the second user response unblocks the next pending caller; (c) cancelled bridge tasks (when another bridge task wins the race) do not leak — `asyncio.CancelledError` is caught and ignored; (d) sub-test: after all 3 callers resolved, `input_queue.empty() == True` and `serializer._caller_to_request == {}`.
**AC9.14** *(v1.4 — V12 backward compat)* Construct any existing `InteractiveBase` subclass (rich, terminal, rich_terminal, queue, web) **without** calling `enable_serialization`. Verify that `asend_response`/`aget_input` behavior is **byte-identical** to today's single-caller path: same WS messages emitted, same `_current_input_mode` mutations, same return values. The opt-in serialization mechanism imposes zero overhead and zero behavior change on the single-caller default.

#### §9.12 SOPInferencer lifecycle — `run_one_turn` + completion (v1.3 — Q4/Q6)

**Why this subsection exists:** v1.2 §9.4 referenced `SOPInferencer.run_one_turn(user_input)` but `ConversationalInferencer` only has `run_agentic_loop(content, ...)` (verified at `conversational_inferencer.py:137`). v1.3 closes the gap by defining `run_one_turn` as a thin wrapper + completion-check.

##### §9.12.1 `run_one_turn` semantics

```python
class SOPInferencer(ConversationalInferencer):
    """One SOP instance per active workflow. Owns its own tracker, messages, prior_context."""

    instance_id: str = attrib()                    # WorkflowInstance.id
    sop_definition: SOP = attrib()                  # parsed SOP from registry
    _tracker: StateGraphTracker = attrib()         # this SOP's phase state
    # v1.4: shares the parent's interactive directly (no proxy wrapper);
    # parent attaches an InteractionSerializer to it via enable_serialization()
    # and sets _CURRENT_INTERACTION_CALLER ContextVar before dispatching.
    _shared_interactive: InteractiveBase = attrib()  # the parent's interactive (NOT wrapped)

    async def run_one_turn(self, user_input: str) -> SOPTurnResult:
        """One agentic-loop invocation scoped to this SOP.

        Terminates when either:
          (a) `_handle_conversation_tools` enqueues a tool to parent's queue and awaits future (loop yields)
          (b) `run_agentic_loop` completes naturally (no further tool calls)
        Returns SOPTurnResult{messages_added, phase_transitions, is_completed}.
        """
        prior_msg_count = len(self.messages)
        await self.run_agentic_loop(content=user_input)
        new_messages = self.messages[prior_msg_count:]
        is_completed = self._tracker.is_completed()
        return SOPTurnResult(
            instance_id=self.instance_id,
            messages_added=new_messages,
            phase_transitions=self._tracker.recent_transitions(since=prior_msg_count),
            is_completed=is_completed,
        )
```

##### §9.12.2 Completion lifecycle

Parent's dispatch loop (pseudocode):

```python
result = await sop_inferencer.run_one_turn(user_input)
self._workflow_manager.persist_turn(result)
if result.is_completed:
    await self._workflow_manager.complete_workflow(result.instance_id)
    # WorkflowManager.complete_workflow:
    #   - sets WorkflowInstance.status = "completed"
    #   - writes <session>/sops/<sop_id>__TS__uuid/state.json
    #   - emits {"type": "WorkflowCompleted", "instance_id": ..., "ts": ...} to session.jsonl
    #   - removes this sop_inferencer from active_inferencers dict
    #   - no per-SOP interactive cleanup needed (shared interactive owned by parent;
    #     v1.4: serializer cancels any pending future for this caller_id automatically)
    self._notify_parent_of_sop_completion(result.instance_id)
```

##### §9.12.3 Stop conditions for `run_agentic_loop`

`SOPInferencer` overrides the loop's stop predicate (or sets it via a kwarg if available) so the loop exits **as soon as** a conversation tool is enqueued (don't burn iterations awaiting a future inside the inferencer — return control to the parent so it can serialize across other SOPs). Implementation: check `len(self._shared_interactive.pending_futures) > 0` between iterations.

#### §9.13 SOP-scoped prompt template (v1.3 — Q5)

**New file:** `prompt_templates/sop/main/initial.jinja2`

Sections (template variables):

```jinja2
# SOP: {{ sop_definition.name }}

{{ sop_definition.description }}

## Current state
- Phase: {{ tracker.current_phase.label }} ({{ tracker.current_phase.id }})
- Status: {{ tracker.current_phase.status }}
- Completed phases: {{ tracker.completed_phase_ids | join(", ") }}

## Phase guidance
{{ tracker.current_phase.guidance }}

## Available tools (this phase only)
{% for tool in phase_specific_tools %}
- `{{ tool.name }}`: {{ tool.description }}
{% endfor %}

## Conversation history
{% for msg in sop_scoped_messages %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}

{# Explicitly NOT included: available_sops block (parent's concern), ongoing_sops routing block (parent's concern) #}
```

The parent's `prompt_templates/conversation/main/initial.jinja2` continues to render `{{ available_sops }}` + `{{ ongoing_sops }}` for routing; the SOP-scoped template renders neither. Clean separation.

#### §9.11 Shared-interactive coordination — `InteractionSerializer` (v1.4 — V3-V6 redesign)

**Architectural principle (v1.4):** the concurrent-callers-sharing-one-user-attention problem is a **transport-layer concern**, not an SOP concern. v1.2/v1.3 implemented it above the interactive layer as `RoutedInteractive` proxy + `PendingConversationToolQueue` — that worked but was SOP-specific and required wrapping every shared interactive. v1.4 moves the serialization mechanism **into `InteractiveBase`** as an opt-in attribute, so any caller (SOP inferencers, BTA workers, background tasks) benefits without per-domain wrappers.

**Problem (unchanged from v1.3):** `RichInteractiveBase._current_input_mode` (`rich_interactive_base.py:57-78`) and `WebSocketInteractive` (`server/services/websocket_interactive.py`) both hold per-call instance state. N concurrent `SOPInferencer` instances calling `aget_input()` on the *same* `interactive` will clobber that state and race on the single `input_queue`.

**Solution (v1.4):** three coordinated changes — one base-class enhancement, one cross-package inheritance fix, one new utility class.

##### §9.11.1 Change 1 — `InteractiveBase` gains optional serialization (the load-bearing change)

Add to `agent_foundation/ui/interactive_base.py`:

```python
import asyncio
from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_foundation.ui.interaction_serializer import InteractionSerializer

# Module-level context var carries the current caller's identity through await chains.
# Used by InteractionSerializer to attribute requests without polluting the call-site API.
_CURRENT_INTERACTION_CALLER: ContextVar[str] = ContextVar("interaction_caller", default="")


class InteractiveBase(ABC):
    # ... existing fields ...
    _serializer: Optional["InteractionSerializer"] = None  # NEW (v1.4)

    def enable_serialization(self, serializer: "InteractionSerializer") -> None:
        """Activate per-caller serialization for this interactive.

        Idempotent; safe to call multiple times with the same serializer.
        Calling with a different serializer when one is already active raises ValueError.
        """
        if self._serializer is not None and self._serializer is not serializer:
            raise ValueError("interactive already has a different serializer attached")
        self._serializer = serializer

    def disable_serialization(self) -> None:
        """Detach serializer. Pending requests in the serializer remain valid until
        the serializer itself is shut down. After this call, asend_response/aget_input
        revert to direct (single-caller) behavior."""
        self._serializer = None

    async def asend_response(
        self,
        response: Union[Any, List, Tuple],
        flag: InteractionFlags = InteractionFlags.TurnCompleted,
        **kwargs,
    ) -> None:
        """Async wrapper. With serializer: enqueues + returns; the serializer's
        worker presents to the user when this caller's turn comes up.
        Without serializer: direct behavior (asyncio.to_thread(self.send_response))."""
        if self._serializer is not None:
            await self._serializer.enqueue_send(
                interactive=self,
                response=response,
                flag=flag,
                kwargs=kwargs,
            )
            return
        await asyncio.to_thread(self.send_response, response, flag, **kwargs)

    async def aget_input(self) -> Any:
        """Async wrapper. With serializer: waits for the response delivered to
        THIS caller's request (identity from _CURRENT_INTERACTION_CALLER ContextVar).
        Without serializer: direct behavior (asyncio.to_thread(self.get_input))."""
        if self._serializer is not None:
            return await self._serializer.await_response_for_caller()
        return await asyncio.to_thread(self.get_input)
```

Backward compatibility: when `_serializer=None` (default), `asend_response`/`aget_input` are **byte-identical** to today's behavior. All 5 existing subclasses (rich, terminal, rich_terminal, queue, web) require **zero changes** — they inherit the new field transparently. The new `enable_serialization` / `disable_serialization` methods are opt-in.

##### §9.11.2 Change 2 — `WebSocketInteractive` inherits `InteractiveBase`

`OpenStartup/src/openteam/server/services/websocket_interactive.py` is currently duck-typed (module docstring: "*Does NOT inherit from InteractiveBase — run_agentic_loop uses hasattr duck-typing*"). v1.4 corrects this:

```python
from agent_foundation.ui.interactive_base import InteractiveBase, InteractionFlags

class WebSocketInteractive(InteractiveBase):
    """WS-backed implementation of InteractiveBase. Native async; sync abstracts
    are not supported (raise NotImplementedError) since WebSocket transport is
    inherently async."""

    def __init__(self, send_callback, input_queue, ...):
        super().__init__()  # picks up _serializer=None default
        self._send = send_callback
        self._input_queue = input_queue
        # ... existing fields ...

    # ----- Sync abstracts: not supported in WS transport -----
    def _get_input(self) -> str:
        raise NotImplementedError("WebSocketInteractive is async-only; use aget_input()")

    def _send_response(self, response: Any, flag) -> None:
        raise NotImplementedError("WebSocketInteractive is async-only; use asend_response()")

    def reset_input(self, flag) -> None:
        # No-op; WS protocol doesn't have an explicit reset.
        pass

    # ----- Native async overrides (replace InteractiveBase's asyncio.to_thread wrappers) -----
    async def asend_response(self, response, flag=InteractionFlags.TurnCompleted, **kwargs):
        if self._serializer is not None:
            await self._serializer.enqueue_send(
                interactive=self, response=response, flag=flag, kwargs=kwargs,
            )
            return
        # ... existing direct behavior (build pending_input msg, await self._send) ...

    async def aget_input(self) -> Any:
        if self._serializer is not None:
            return await self._serializer.await_response_for_caller()
        return await self._input_queue.get()

    # v1.5 — V16: native-async _direct_send override so the serializer's worker
    # can present widgets without recursing back through asend_response's
    # serializer check (which would re-enqueue and deadlock).
    async def _direct_send(self, response, flag, **kwargs) -> None:
        """Bypass-serializer direct send used by InteractionSerializer worker.
        Native-async path (no asyncio.to_thread) — replaces base-class default
        which would route through sync send_response → NotImplementedError."""
        # Same body as the non-serializer branch of asend_response above:
        # build pending_input message, await self._send(...).
        pending_msg = self._build_pending_input_message(response, flag, **kwargs)
        await self._send(pending_msg)
```

This is purely additive — every existing call site continues to work because the public API surface (`asend_response`/`aget_input`/`stream_token_batches`/`send_turn_boundary` etc.) is unchanged. Risk R20 (cross-package blast radius) covers test-suite verification.

##### §9.11.2.1 — Input-queue → serializer bridge (v1.5 — V17, HIGH severity gap fix)

**The load-bearing wiring detail v1.4 omitted:** today, the React UI posts `pending_input_response` over WebSocket; the route handler at `OpenStartup/src/openteam/server/routes/manager_websocket_routes.py:570-585` calls `input_queue.put(parsed_content)`. With the serializer active, `aget_input()` no longer reads from `input_queue` — it awaits `serializer.await_response_for_caller()`. Without an explicit bridge, the user's response sits in `input_queue` forever and the serializer's per-caller futures never resolve.

**v1.5 design — bridge inside `WebSocketInteractive.aget_input` (NOT in the route handler):**

The route handler stays unchanged (single responsibility: put into `input_queue`). The bridge lives one layer above, in `WebSocketInteractive` itself, because that's where the dual-source decision belongs:

```python
async def aget_input(self) -> Any:
    """Single source of truth for 'wait for user input' on this transport.
    - When serializer is active: drain input_queue (where WS route handler puts
      raw user input) and forward to serializer.deliver_response(); then await
      this caller's per-caller future from serializer.await_response_for_caller().
    - When serializer is inactive: direct queue read (today's behavior).
    """
    if self._serializer is not None:
        # Spawn a one-shot bridge task that drains input_queue ONCE and
        # forwards to deliver_response. This runs in parallel with our await
        # of await_response_for_caller — the first caller to await acts as
        # the bridge for that response cycle. Multiple concurrent callers
        # are correctly serialized because the serializer's _active lock
        # guarantees only one caller's future is resolved per deliver_response.
        bridge_task = asyncio.create_task(self._bridge_one_response())
        try:
            return await self._serializer.await_response_for_caller()
        finally:
            # If we resolved before the bridge task fired (e.g., another
            # caller's bridge already delivered), cancel ours.
            if not bridge_task.done():
                bridge_task.cancel()
    return await self._input_queue.get()

async def _bridge_one_response(self) -> None:
    """One-shot: take the next input from input_queue, hand it to the serializer.
    Runs once per aget_input call; cancellation-safe."""
    try:
        raw = await self._input_queue.get()
    except asyncio.CancelledError:
        return
    if self._serializer is not None:
        await self._serializer.deliver_response(raw)
```

**Why this is the right layer:** the WS route handler shouldn't know about `InteractionSerializer` (mixing transport with serialization is exactly the leak v1.4 set out to fix). `ConversationService.on_user_input_received` doesn't own `input_queue` (the route handler does). `WebSocketInteractive` owns the `input_queue` (passed at construction) AND owns the `_serializer` (via `enable_serialization`). It's the single class that sees both, so it's the natural home for the bridge.

**Concurrency correctness:** N concurrent `aget_input` callers each spawn a `_bridge_one_response` task. The first to grab from `input_queue` wins; the rest sit idle until `cancel()`. The serializer's `_active` lock guarantees exactly one of the N caller futures resolves per `deliver_response` call. Subsequent user responses unblock the next pending caller in the same way.

**Alternative considered + rejected:** putting the bridge in the WS route handler (`if active_serializer: serializer.deliver_response(parsed); else: input_queue.put(parsed)`). Rejected because: (a) the route handler is OpenStartup-side and shouldn't depend on AgentFoundation `InteractionSerializer`; (b) it requires the handler to discover the active serializer via session lookup (extra coupling); (c) `WebSocketInteractive`-internal bridge keeps the layering clean and isolates the "transport routing" concern in the transport class.

##### §9.11.3 Change 3 — New utility class `InteractionSerializer`

New file: `agent_foundation/ui/interaction_serializer.py` (~120 LoC).

```python
from __future__ import annotations
import asyncio
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional
from agent_foundation.ui.interactive_base import _CURRENT_INTERACTION_CALLER, InteractiveBase

@dataclass
class _PendingRequest:
    caller_id: str
    interactive: InteractiveBase
    response: Any
    flag: Any
    kwargs: dict
    response_future: asyncio.Future


class InteractionSerializer:
    """Serializes concurrent interaction requests through a single interactive.

    Use when N concurrent async callers (e.g., N SOPInferencer instances; N BTA
    worker tasks; background autonomous agents) share one user-facing interactive
    and you need exactly one widget visible to the user at a time.

    Caller identity is carried implicitly via _CURRENT_INTERACTION_CALLER ContextVar;
    callers set the var before invoking the inferencer; the serializer reads it on
    each enqueue/await.

    Activation: call `interactive.enable_serialization(InteractionSerializer())`.
    Deactivation: call `interactive.disable_serialization()` + `serializer.shutdown()`.
    """

    def __init__(self) -> None:
        self._pending: deque[_PendingRequest] = deque()
        self._active: Optional[_PendingRequest] = None
        self._caller_to_request: dict[str, _PendingRequest] = {}
        self._lock = asyncio.Lock()

    async def enqueue_send(
        self, interactive: InteractiveBase, response: Any, flag: Any, kwargs: dict,
    ) -> None:
        """Enqueue a widget-send request. Returns immediately; presentation happens
        when this caller's turn comes up. The user's response is later delivered
        through await_response_for_caller()."""
        caller_id = _CURRENT_INTERACTION_CALLER.get()
        if not caller_id:
            raise RuntimeError(
                "InteractionSerializer requires a caller identity. "
                "Set _CURRENT_INTERACTION_CALLER ContextVar before invoking."
            )
        if caller_id in self._caller_to_request:
            raise RuntimeError(
                f"caller {caller_id} already has a pending interaction request; "
                "a caller may only have one outstanding request at a time."
            )
        loop = asyncio.get_running_loop()
        req = _PendingRequest(
            caller_id=caller_id, interactive=interactive, response=response,
            flag=flag, kwargs=kwargs, response_future=loop.create_future(),
        )
        self._caller_to_request[caller_id] = req
        async with self._lock:
            self._pending.append(req)
        await self._try_present_next()

    async def await_response_for_caller(self) -> Any:
        """Block until the user has responded to THIS caller's outstanding request.
        Caller identity from ContextVar."""
        caller_id = _CURRENT_INTERACTION_CALLER.get()
        if caller_id not in self._caller_to_request:
            raise RuntimeError(
                f"caller {caller_id} has no outstanding request; "
                "enqueue_send must be called before await_response_for_caller."
            )
        req = self._caller_to_request[caller_id]
        try:
            return await req.response_future
        finally:
            self._caller_to_request.pop(caller_id, None)

    async def deliver_response(self, response: Any) -> None:
        """Called by the transport layer when the user responds to the active widget.
        Resolves the active caller's future, then promotes the next pending request."""
        async with self._lock:
            if self._active is None:
                # User responded without any active widget — shouldn't happen normally;
                # log and drop.
                return
            self._active.response_future.set_result(response)
            self._active = None
        await self._try_present_next()

    async def _try_present_next(self) -> None:
        async with self._lock:
            if self._active is not None or not self._pending:
                return
            self._active = self._pending.popleft()
        # Outside lock to avoid holding it across the (potentially network-bound) send.
        # Call the interactive's underlying direct-send path (bypassing serializer check)
        # by invoking the sync-ish core directly OR by using a private direct method.
        # Implementation note: we use `_direct_send` (a new InteractiveBase method that
        # wraps the existing direct-send logic) to avoid re-entering asend_response.
        await self._active.interactive._direct_send(
            self._active.response, self._active.flag, **self._active.kwargs,
        )

    def has_pending_for(self, caller_id: str) -> bool:
        """For §9.4 rule-based routing fast-path queries."""
        return caller_id in self._caller_to_request

    def has_any_active(self) -> bool:
        """True iff there is a widget currently being shown to the user (i.e.,
        an active request awaiting a response). Used by ConversationService
        (§9.11.4) to decide whether incoming raw input should be routed via
        serializer.deliver_response or treated as a fresh routing input."""
        return self._active is not None

    def shutdown(self) -> None:
        """Cancel any outstanding futures (cleanup on session teardown)."""
        for req in self._caller_to_request.values():
            if not req.response_future.done():
                req.response_future.cancel()
        self._pending.clear()
        self._caller_to_request.clear()
        self._active = None
```

`InteractiveBase` needs one additional helper to support the bypass path:

```python
async def _direct_send(self, response, flag, **kwargs) -> None:
    """Send-without-serializer-check; used internally by InteractionSerializer
    to avoid recursing through asend_response."""
    await asyncio.to_thread(self.send_response, response, flag, **kwargs)
```

`WebSocketInteractive._direct_send` overrides this with its native-async direct WS send (no `asyncio.to_thread` needed).

##### §9.11.4 How SOPs use the serializer (parent inferencer setup)

```python
# In ConversationService session-init (when first enter_sop happens in multi mode
# with len(active) about to become >= 2):
from agent_foundation.ui.interaction_serializer import InteractionSerializer
from agent_foundation.ui.interactive_base import _CURRENT_INTERACTION_CALLER

class ConversationService:
    def _maybe_enable_serialization(self) -> None:
        """Activate serializer lazily on first multi-active state; idempotent."""
        if self._serializer is None and self.workflow_manager.active_count() >= 2:
            self._serializer = InteractionSerializer()
            self.interactive.enable_serialization(self._serializer)

    async def dispatch_user_input_to_sop(self, sop_id: str, user_input: str) -> None:
        sop_inferencer = self.workflow_manager.get_sop_inferencer(sop_id)
        token = _CURRENT_INTERACTION_CALLER.set(sop_id)
        try:
            await sop_inferencer.run_one_turn(user_input)
        finally:
            _CURRENT_INTERACTION_CALLER.reset(token)

    async def on_user_input_received(self, raw_input: str) -> None:
        """WS handler delivers user input; if there's an active serialized request,
        route through serializer; otherwise treat as a regular routing input."""
        if self._serializer is not None and self._serializer.has_any_active():
            await self._serializer.deliver_response(raw_input)
            return
        # ... existing parent-routing logic (§9.4) ...
```

Routing fast-path (§9.4 rule (i)) consults `self._serializer.has_pending_for(sop_id)` instead of v1.3's `pending_tool_in_queue` snapshot — same semantics, cleaner implementation.

##### §9.11.5 Concurrency model (unchanged from v1.3, simpler implementation)

The parent inferencer's event loop runs:
- Its own `run_one_turn` (chitchat + routing).
- An `asyncio.Task` per **routed** SOP (only SOPs the parent's hybrid router (§9.4) has dispatched user input to in the current turn). Idle SOP inferencers are not tasks — they sit waiting for the parent to deliver input.

Inside each task, `_CURRENT_INTERACTION_CALLER` is set to the SOP's `instance_id` before invoking `run_one_turn`. Any `asend_response`/`aget_input` calls inside that task (including those nested in `run_agentic_loop`) automatically pick up the caller identity via ContextVar. No call-site changes needed in `ConversationalInferencer` or the inferencer base class beyond the serializer-aware wrappers.

##### §9.11.6 Resume semantics (unchanged from v1.3)

Restart-safe by construction (matches §10):
- Each `SOPInferencer`'s `messages` log + `_tracker` state persists to `<session>/sops/<sop_id>__TS__uuid/turns/` (turn records include `caller_id` field from ContextVar at write time — v1.4 V10).
- On parent reload, parent iterates `<session>/sops/*/` and reconstructs each `SOPInferencer` from its folder.
- The `InteractionSerializer` is **not** persisted: on reload, any pending widget is re-emitted by the relevant SOP inferencer naturally when it resumes its phase. The serializer is constructed fresh by `_maybe_enable_serialization`.

---

### §10. SOP runtime layout (Concern #5)

#### §10.1 On-disk layout

```
<runtime_root>/servers/server_<TS>_<uuid8>/sessions/<session_id>_<TS>/
├── session_state.json                                # gains: active_sops[], focus_mode fields
├── session.jsonl                                     # gains: SOPInvoked / SOPCompleted / SOPModeChanged / RoutingDecision / EnvOverride records
├── turn_001/, turn_002/, ...                         # parent conversation turns
├── tasks/                                            # existing — tasks invoked from parent conversation
└── sops/                                             # NEW (plural — matches `resources/sops/`)
    └── role_creation__20260526_154500__a1b2c3d4/    # one folder per WorkflowInstance
        ├── sop_state.json                            # WorkflowInstance.to_persistent_dict() + extras
        ├── sop_definition_snapshot.json              # frozen copy of sop.config.json at entry time
        ├── session.jsonl                             # SOP-run turn log (same schema as parent)
        ├── turn_001/
        │   ├── rendered_prompt.txt
        │   ├── template_feed.json
        │   ├── template_config.json
        │   ├── api_payload.json
        │   ├── inference_response.txt
        │   ├── user_input.txt                        # synthetic or human
        │   └── metadata.json                         # {source: "synthetic"|"human", phase_id, instance_id, caller_id, ...}  # v1.4 V10: caller_id is the value of _CURRENT_INTERACTION_CALLER ContextVar at write time (== sop_instance_id for SOP-scoped turns)
        ├── turn_002/, ...
        └── tasks/                                    # tasks invoked from THIS SOP run
            ├── create_role__20260526_154700__b5c6d7e8/
            └── role_setup__20260526_160230__c9d8e7f6/
```

#### §10.2 Identifiers

- **SOP run folder:** `<sop_name>__<YYYYMMDD_HHMMSS>__<uuid8>` (double-underscore separator visually distinguishes from task folders' single-underscore `<tool>_<TS>_<uuid8>` per `allocator.py:57-69`)
- **In-context SOP run ID:** `sop-<uuid8>` (mirrors `task-<uuid8>` at tool_dispatcher.py:186)
- **`WorkflowInstance.instance_id`** keeps current 8-hex format from manager.py:59; embedded as the `<uuid8>` segment of the folder name for back-reference.

#### §10.3 `SOPSession` — new file `AgentFoundation/.../server/sop_session.py` (~150 LoC)

```python
class SOPSession:
    """A conversational session scoped to a single SOP run.
    Lives at <parent_session>/sops/<sop_run_folder>/.
    """
    def __init__(self, parent_session_dir: Path, sop_name: str,
                 instance_id: str, sop_config: dict, yolo_mode: bool): ...
    def allocate(self) -> Path: ...                    # creates <parent>/sops/<folder>/
    def get_tasks_dir(self) -> Path: ...               # <sop_run>/tasks/ for nested task allocation
    def save_turn_data(self, turn_idx: int, *,
                       source: Literal["human","synthetic","system"],
                       prompt_data: dict, user_input: str, response: str,
                       phase_id: str, **kwargs) -> None: ...
    def get_jsonl_logger(self) -> JsonLogger: ...      # writes to <sop_run>/session.jsonl
    def to_persistent_dict(self) -> dict: ...          # for sop_state.json
```

#### §10.4 Critical refactor — `save_turn_artifacts` shared free function

Factor `save_turn_data` out of `SessionStore` (`OpenStartup/.../services/session_store.py:368-415`) into a shared free function:

```python
# new: AgentFoundation/.../server/turn_artifacts.py
def save_turn_artifacts(turn_dir: Path, *, prompt_data: dict, user_input: str,
                        response: str, source: Literal["human","synthetic","system"],
                        **extra) -> None:
    """Single source of truth for turn artifact format.
    Both SessionStore.save_turn_data and SOPSession.save_turn_data delegate to this.
    """
    ...
```

This avoids drift between parent-session and SOP-session turn artifact formats.

#### §10.5 Parent session linkage

`session_state.json` adds:
```jsonc
{
  // existing fields...
  "focus_mode": "multi",                               // §9
  "active_sops": [
    {
      "sop_name": "role_creation",
      "instance_id": "a1b2c3d4",
      "sop_run_folder": "sops/role_creation__20260526_154500__a1b2c3d4",
      "status": "active",
      "focused": false,                                // §9 (advisory in multi)
      "yolo_mode": true,
      "entered_at_turn": 7,
      "entered_at_iso": "2026-05-26T15:45:00Z"
    }
  ]
}
```

#### §10.6 New `session.jsonl` record types

```json
{"timestamp":"...", "type":"SOPInvoked", "sop_name":"role_creation", "instance_id":"a1b2c3d4", "sop_run_folder":"sops/...", "trigger_turn":7, "yolo_mode":true}
{"timestamp":"...", "type":"SOPCompleted", "instance_id":"a1b2c3d4", "completed_phases":["0","1","1b","2","2b","3"]}
{"timestamp":"...", "type":"SOPModeChanged", "from":"multi", "to":"single", "triggered_by":"slash_command"}
{"timestamp":"...", "type":"RoutingDecision", ...}  // §9.9
{"timestamp":"...", "type":"EnvOverride", "field":"focus_mode", "from":"multi", "to":"single", "source":"OPENTEAM_SOP_FORCE_SINGLE_FOCUS"}
```

#### §10.7 `source` field — end-to-end pipe

| Location | Field | Default |
|---|---|---|
| `OpenStartup/.../routes/manager_websocket_routes.py:348-355` user_msg dict | `source` | `"human"` (or `"auto_advance"` if existing JS flag is set) |
| `OpenStartup/.../services/conversation_service.py:559` JSONL UserInput record | `source` | forwarded |
| `OpenStartup/.../services/session_store.py:368-415` save_turn_data → metadata.json | `source` | forwarded |
| `SOPSession.save_turn_data` (new) | `source` | `"synthetic"` when yolo synthesized, else `"human"` |

#### §10.8 SOP entry mechanism

Both LLM-implicit and explicit (per Tony's direction):

**Implicit (LLM-driven, primary):** new `enter_sop` action tool (`AgentFoundation/.../resources/tools/enter_sop/{tool.json, executor.py}`). Args: `{sop_name: string, yolo: bool, params: dict}`. Orchestrator prompt instructs: "When user request matches an entry in `## Available SOPs` by keyword or example_request, emit `enter_sop` with the matching name. Multiple `enter_sop` calls in one turn are allowed."

**Explicit (slash command, escape hatch):** refactor `AgentFoundation/.../resources/tools/sop/executor.py:24-128` from "run end-to-end synchronously" to "enter and stay active". Remove `await instance._graph_task` (line 105-106) for enter-and-stay-active path; **keep `--autonomous` flag** to opt into the old WorkGraph end-to-end runner.

Symmetric: `exit_sop`, `set_sop_mode`, and slash flags `/sop --exit <id>`, `/sop --mode <name>`, `/sop --focus <id>`, `/sop --list`.

#### §10.9 Reuse (no duplication)

- `SessionStore.save_turn_data` (factored to `save_turn_artifacts` per §10.4)
- `JsonLogger`
- `_on_new_turn` callback pattern (`conversation_service.py:558-606`)
- `allocate_tool_workspace` (`AgentFoundation/.../workspace/allocator.py:72-115`) called with `base_dir=<sop_run>/tasks` for nested task allocation
- `WorkflowManager.enter_workflow / exit_workflow / resume_workflow` (workflow/manager.py:54-136)

#### §10.10 ACs

**AC10.1** Entering `role_creation` creates `<session>/sops/role_creation__<TS>__<uuid8>/` with `sop_state.json`, `sop_definition_snapshot.json`, empty `session.jsonl`, empty `tasks/`.
**AC10.2** First SOP turn produces `turn_001/` with all standard artifacts + `metadata.json` containing `source`, `phase_id`, `instance_id`.
**AC10.3** When the SOP invokes `/create-role`, the task workspace lands at `<session>/sops/role_creation__.../tasks/create_role__<TS>__<task_id>/`, NOT at `<session>/tasks/`.
**AC10.4** Parent `session_state.json` gains `active_sops[]` and `focus_mode` fields. Parent `session.jsonl` contains exactly one `SOPInvoked` record per `enter_sop` invocation; no SOP-internal turns leak into the parent log.
**AC10.5** `SOPSession.save_turn_data` and `SessionStore.save_turn_data` produce byte-identical `turn_NNN/` artifacts (verified by diff against the shared `save_turn_artifacts` helper).

---

### §11. PR sequence

5 PRs, ordered by dependency and risk. Each PR is independently shippable and revert-safe.

| PR | Phases | Risk | Why | Depends on |
|---|---|---|---|---|
| **PR-0** | Phase 0 prerequisites | Low | Empirically verify parser preserves unknown fields (§9.4 Mechanism A); add the 14 RED tests; no source changes | — |
| **PR-1** | §3 + §4 + §5 (description + meta-tags + SOP-as-resource) | Low | Pure additive: registry + prompt blocks + meta-tags + refined description. No behavior change for existing flows. | PR-0 |
| **PR-2** | §9 (mode system, MINUS per-instance tool_phase_map) | Medium | Mode system, routing, focus rendering. Independent of yolo. Per-instance map deferred to PR-4 behind feature flag. | PR-1 |
| **PR-3** | §6 + §7 (yolo synthesis + ToolDefinition.concurrency) | Medium | Yolo synthesis touches the agentic loop's `_handle_conversation_tools` gate; deletes `render_for_mode`. Concurrency field is orthogonal backward-compatible. | PR-1 (sop attribution surface) |
| **PR-4** | §10 + §8 (SOPSession + folder layout + per-instance tool_phase_map + queue) | Medium-high | `SOPSession` + folder layout + `source` labeling end-to-end across OpenStartup. Per-instance tool_phase_map behind feature flag. Queue only instantiated for autonomous path. | PR-2, PR-3 |
| **PR-5** | Polish | Low | role_creation.jinja2:34 typo fix; E2E test; JS `is_auto_advance` deprecation; UI surface for `focus_mode` | PR-4 |

**PR-0 verification gate (must pass before PR-1 starts):**
- AC9.10 (parser preservation of `sop_instance_id` field) — VERIFIED or parser patched
- 14 RED tests written (§11.2) — all xfail/strict before any source change

#### §11.1 PR-0: RED test inventory (write all before any source change)

| # | Test | File | Verifies |
|---|---|---|---|
| 1 | Parser preserves `sop_instance_id` field on tool calls | `test/.../test_tool_call_parser_extra_fields.py` | §9.4 Mechanism A |
| 2 | `SOPRegistry().load_all` returns 3 SOPs from both repos | `test_sop_registry.py` | AC5.1 |
| 3 | `linear_only: true` SOP rejects `__goto__` | `test_sop_registry_linear_only.py` | AC5.6 |
| 4 | `__keywords__` + `__example_requests__` parser accepts both line/bullet forms | `test_sop_meta_tags.py` | AC4.1, AC4.2 |
| 5 | Unknown meta-tag raises `SOPParseError` | `test_sop_meta_tags.py` | AC4.3 |
| 6 | `_synthesize_yolo_collected` returns correct shape for each tool type | `test_yolo_synthesis_protocol.py` | AC6.2, §6.3 |
| 7 | `prompt_llm` mode caches on `(prompt_hash, last_3_turn_hashes)` | `test_yolo_prompt_llm.py` | AC6.6 |
| 8 | `none` mode falls back to human interactive even in yolo | `test_yolo_synthesis.py` | AC6.7 |
| 9 | `ToolDefinition.concurrency` round-trips both new + legacy `asynchronous` | `test_tool_definition_concurrency.py` | AC7.1 |
| 10 | `InteractionSerializer.enqueue_send` + `await_response_for_caller` ordering semantics + `has_pending_for` query (replaces the deprecated `PendingConversationToolQueue.reorder_to_head` test) | `test_interaction_serializer_ordering.py` | AC8.1, AC8.2 (re-targeted to InteractionSerializer) |
| 11 | Queue NOT instantiated in tracker-driven path | `test_queue_lifecycle.py` | AC8.5 |
| 12 | `multi` mode + 1 active SOP omits routing block | `test_prompt_routing_block.py` | AC9.2 |
| 13 | Routing scenarios A-F via mock LLM | `test_routing_scenarios.py` | AC9.4, AC9.5 |
| 14 | `SOPSession.save_turn_data` byte-identical to `SessionStore.save_turn_data` | `test_save_turn_artifacts_parity.py` | AC10.5 |
| 15 | `ChoiceItem.recommended` round-trips parse/serialize; `recommended` yolo mode picks flagged choice | `test_choice_item_recommended_field.py` | AC6.9-6.12 (v1.1) |
| 16 | 3 concurrent `SOPInferencer`s firing `confirmation` tools serialize through `InteractionSerializer` attached via `interactive.enable_serialization()`; widgets shown 1-at-a-time; per-caller `_CURRENT_INTERACTION_CALLER` ContextVar correctly attributes responses; no `_current_input_mode` clobber; no cross-SOP message bleed | `test_interaction_serializer_multi_caller.py` | AC9.11 (v1.4 retargeted) |
| 17 | Backward compat: `InteractiveBase` subclass without `enable_serialization()` behaves byte-identically to today (single-caller direct path) | `test_interaction_serializer_optin_zero_overhead.py` | AC9.14 (v1.4) |
| 18 | `WebSocketInteractive` inherits `InteractiveBase`; all 4 protocol methods (`stream_token_batches`, `send_turn_boundary`, `asend_response`, `aget_input`) behave correctly under both duck-typed and inherited call paths | `test_websocket_interactive_inheritance.py` (OpenStartup-side) | R20 (v1.4) |

---

### §12. File inventory

#### §12.1 New files

| Path | Purpose | Approx LoC |
|---|---|---|
| `AgentFoundation/src/agent_foundation/resources/sops/__init__.py` | Package marker | 5 |
| `AgentFoundation/src/agent_foundation/resources/sops/registry.py` | `SOPRegistry` mirroring skills/registry.py | ~80 |
| `AgentFoundation/src/agent_foundation/resources/sops/code_optimization/{SOP.md, sop.config.json}` | Migrated SOP | (moved) |
| `AgentFoundation/src/agent_foundation/resources/sops/model_optimization/{SOP.md, sop.config.json}` | Migrated SOP | (moved) |
| `OpenStartup/src/openteam/server/resources/sops/role_creation/{SOP.md, sop.config.json}` | Migrated SOP + refined description | (moved + edited) |
| `OpenStartup/src/openteam/server/resources/sops/role_creation/references/role_categories.md` | Optional deep-dive notes | (new) |
| `AgentFoundation/src/agent_foundation/server/sop_session.py` | `SOPSession` class | ~150 |
| `AgentFoundation/src/agent_foundation/server/turn_artifacts.py` | `save_turn_artifacts` shared free function | ~50 |
| `AgentFoundation/src/agent_foundation/resources/tools/enter_sop/{tool.json, executor.py}` | LLM-driven SOP entry | ~40 |
| `AgentFoundation/src/agent_foundation/resources/tools/exit_sop/{tool.json, executor.py}` | LLM-driven SOP exit | ~30 |
| `AgentFoundation/src/agent_foundation/resources/tools/set_sop_mode/{tool.json, executor.py}` | LLM-driven focus_mode toggle | ~20 |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/templates/yolo_synthesize.jinja2` | `prompt_llm` mode template | ~20 |
| `AgentFoundation/src/agent_foundation/ui/interaction_serializer.py` *(v1.4 — V7)* | `InteractionSerializer` + `_PendingRequest` dataclass; the single transport-layer mechanism for serializing concurrent interactive callers (replaces the v1.2/v1.3 `RoutedInteractive` + `PendingConversationToolQueue` pair) | ~120 |
| `OpenStartup/test/services/test_websocket_interactive_inheritance.py` *(v1.4 — V9)* | Asserts `isinstance(ws_interactive, InteractiveBase)` AND that all 4 protocol methods behave correctly under both call paths | ~80 |
| `AgentFoundation/test/.../test_role_creation_sop_e2e.py` | Phase 0→3 in yolo (E2E) | ~150 |
| `AgentFoundation/test/.../test_multi_active_sops.py` | 2 SOPs in multi_focus | ~120 |
| `AgentFoundation/test/.../test_sop_mode_switching.py` | single↔multi transitions | ~80 |
| RED tests per §11.1 | 14 unit/integration tests | ~600 total |

#### §12.2 Modified files

| Path | Phases | Change |
|---|---|---|
| `RichPythonUtils/.../sop_manager.py` | §4 + §6 | Add `_KEYWORDS_RE` + `_EXAMPLE_REQUESTS_RE` + `_extract_top_level_tags()`; delete `render_for_mode` body |
| `AgentFoundation/.../resources/tools/{clarification,single_choice,multiple_choice,confirmation}/tool.json` | §6 | Add `yolo_default` field |
| `AgentFoundation/.../resources/tools/models.py` | §6 + §7 | Add `yolo_default: Optional[dict]`; add `concurrency: Literal[...]`; back-compat shim for `asynchronous` |
| `AgentFoundation/.../common/workflow/manager.py` | §5 + §6 + §9 | `render_prompt_sections` returns list-of-dicts with per-mode render_depth; drop yolo branch; expose `focus_mode` + `pending_tool_queue` slots; add `_resolve_focus_winner_on_mode_switch` |
| `AgentFoundation/.../common/workflow/registry.py` | §5 | Thin adapter delegating to `SOPRegistry` |
| `AgentFoundation/.../common/workflow/instance.py` | §9 | Add `pending_conversation_tool`, `last_user_input_advanced_at`, `is_computing`, `focused`, `sop_run_folder` |
| `AgentFoundation/.../server/workflow_context.py` | §9 | Per-instance `tool_phase_maps_by_instance`; `start_phase`/`complete_phase` accept optional `instance_id` (feature-flagged) |
| `AgentFoundation/.../inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | §6 + §9 + §10 | Gate split at line 289; `_synthesize_yolo_collected`, `_resolve_yolo_spec`, `_yolo_prompt_llm`, `_resolve_sop_attribution`; outer-inferencer `_run_outer_turn` with re-render loop; forward `source` to `on_new_turn`; read `focus_mode` from prior_context |
| `AgentFoundation/.../ui/interactive_base.py` *(v1.4 — V8)* | §9.11 | Add module-level `_CURRENT_INTERACTION_CALLER: ContextVar[str]`; add `_serializer: Optional[InteractionSerializer]` field; add `enable_serialization()` / `disable_serialization()` methods; update `asend_response`/`aget_input` async wrappers to delegate to serializer when active (zero behavior change when `_serializer=None`); add `_direct_send` helper used by serializer to bypass re-entry |
| `OpenStartup/src/openteam/server/services/websocket_interactive.py` *(v1.4 — V9)* | §9.11 | Make `WebSocketInteractive` inherit `InteractiveBase` (cross-package); call `super().__init__()`; sync abstracts raise `NotImplementedError("use async methods")`; native-async overrides of `asend_response`/`aget_input` check `_serializer` first; override `_direct_send` to call WS native-async path directly |
| `OpenStartup/src/openteam/server/services/conversation_service.py` *(v1.4 — V10)* | §9.11 | Add `_serializer: Optional[InteractionSerializer]` to `ConversationService`; `_maybe_enable_serialization()` lazy-attaches on first `len(active) >= 2`; `dispatch_user_input_to_sop` sets `_CURRENT_INTERACTION_CALLER` ContextVar before invoking `sop_inferencer.run_one_turn`; `on_user_input_received` delegates to `serializer.deliver_response()` when serializer has active request; turn record schema gains `caller_id` from ContextVar at write time |
| `AgentFoundation/.../inferencers/agentic_inferencers/conversational/tool_call_parser.py` | §9.4 | Preserve unknown top-level fields on `ParsedToolCall.extra` (only if PR-0 verification shows it's not already preserved) |
| `AgentFoundation/.../prompt_templates/conversation/main/initial.jinja2` | §5 + §9 | Rename `Available Workflows`→`Available SOPs`, `Ongoing Workflows`→`Active SOPs`; per-mode rendering loop with conditional routing block |
| `AgentFoundation/.../resources/tools/sop/{tool.json, executor.py}` | §10 | Refactor to enter-and-stay-active with `--autonomous` opt-in; use `SOPSession`; add `--mode`, `--focus`, `--exit`, `--list` flags |
| `OpenStartup/.../backends/factories.py:157-164` | §5 | Pass `workflow_manager=` to `ConversationalInferencer` |
| `OpenStartup/.../services/conversation_service.py` | §10 | `_compute_session_context` reads `active_sops[]`; `_on_new_turn` forwards `source`; `_persist_workflow_updates` updates `active_sops[]` |
| `OpenStartup/.../services/session_store.py` | §10 | Factor `save_turn_data` into shared `save_turn_artifacts`; persist `focus_mode` + `active_sops[]`; new `get_session_sops_dir` helper |
| `OpenStartup/.../services/tool_dispatcher.py` | §9 | Read `sop_instance_id` from tool call; route `context_updates` to matching `WorkflowInstance` |
| `OpenStartup/.../routes/manager_websocket_routes.py:354-355` | §9 + §10 | `source` field on user_msg; `POST /api/sessions/{id}/sop-mode` endpoint |
| `OpenStartup/.../ui/src/hooks/useManagerChat.js` | §10 (PR-5) | Surface `focus_mode` in UI; deprecate `is_auto_advance` shim |
| `OpenStartup/.../resources/sops/role_creation/SOP.md:line 34` | PR-5 | Typo fix: `[__requires_confirmation__]` (underscore) → `[__requires confirmation__]` (space) |

#### §12.3 Deletions (after each PR ships)

| Path | After PR | Why |
|---|---|---|
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_sop/` | PR-1 | Replaced by `resources/sops/` |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow_description/default.jinja2` | PR-1 | Replaced by `WorkflowManager.render_prompt_sections()` |
| `AgentFoundation/.../prompt_templates/conversation/main/_variables/workflow/.sop.config.yaml` | PR-1 | Replaced by per-SOP `sop.config.json` |
| `SOPManager.render_for_mode` body | PR-3 | Yolo no longer manipulates SOP text |
| `WorkflowManager.render_prompt_sections:174-177` (yolo branch) | PR-3 | Same |
| JS `is_auto_advance` shim | PR-5 | Replaced by server-side `source` field |

---

### §13. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Wiring `workflow_manager` into OpenStartup factory breaks sessions that don't expect `available_sops` block | LOW | Block only renders when SOPs exist (`{% if available_sops %}`); existing prompt unchanged when none. |
| R2 | Synthetic yolo responses produce nonsense for ambiguous multiple_choice | MED | Per-SOP `yolo_overrides` lets each SOP pick its strategy; `select_all` is conservative; `none` mode forces human reply. |
| R3 | `SOPSession` and `SessionStore` drift over time | HIGH | Factor `save_turn_data` into shared `save_turn_artifacts`; both classes call it. Single source of truth for artifact format. |
| R4 | Multi-focus prompt size balloons with many active SOPs | MED | P0 cap (`max_p0_full_render` default 3); `COMPACT` demotion + warning; deterministic budget algorithm in §9.3. |
| R5 | Multi-focus tool-call attribution: LLM forgets `sop_instance_id` | MED | Single-active fallback unambiguous; multi-active fallback to most-recently-progressed + warning + metric. Heuristic prefilter (§9.4 fallback) further reduces missed attributions. |
| R6 | `is_auto_advance` JS round-trip and server-side yolo synthesis race or duplicate | MED | Server detects `source` field on incoming msg; if `auto_advance` set, skip yolo synthesis for that turn. JS shim deprecated after parity verified (PR-5). |
| R7 | Refactoring `WorkflowRegistry` to delegate to `SOPRegistry` breaks `/sop --autonomous` | MED | Adapter exposes same `get / list_all` surface; `sop/executor.py:44-53` paths covered in PR-1 tests. |
| R8 | Per-SOP `yolo_overrides` shadowing global `yolo_default` makes debugging tricky | LOW | Debug logger in `_synthesize_yolo_collected`: `[yolo] tool=X mode=Y source=tool.json\|sop.config\|builtin value=Z instance=...`. |
| R9 | Mode switch in the middle of an SOP turn (e.g., user types `/sop --mode multi` while widget pending) | LOW | Mode change takes effect on the NEXT turn's prompt; in-flight widgets complete with current-mode semantics. Log `SOPModeChanged` with timestamp. |
| R10 | Deleting `default.jinja2` breaks code that imports it | LOW | Grep before deletion (only consumer is the dead-code branch in manager.py:174-177); add explicit removal commit message. |
| R11 | Linear-only SOPs with `__goto__`/`__branch__`/`__for_each__` | LOW | Registry validates `linear_only` at load; rejects with clear error naming offending phase (AC5.6). |
| R12 | Two SOPs invoke same action tool with conflicting params in same multi_focus turn | MED | Dispatcher validates: same tool name + differing args + same target → error tool result, no execution. |
| R13 | Per-instance `tool_phase_map` refactor breaks existing single-focus sessions | HIGH | Feature-flag (`OPENTEAM_PER_INSTANCE_TOOL_PHASE_MAP=true`); legacy single-map path preserved when off; parity verified before flag default flips. |
| R14 | Parser doesn't preserve unknown `sop_instance_id` field (§9.4 Mechanism A) | HIGH | PR-0 prerequisite test verifies; if false, 1-line parser patch lands BEFORE Mechanism A is shipped; meanwhile Mechanism B (`workflow_route` tool call) is the supported path. |
| R15 | `prompt_llm` yolo mode LLM call inflates per-turn cost | MED | Opt-in only (no builtin tool default uses it); cached on `(prompt_hash, last_3_turn_hashes)`; logged + metricized for cost observability. |
| R16 | role_creation.jinja2:34 typo persists in migrated SOP.md after move | LOW | PR-5 explicit typo fix; AC test asserts `_REQUIRES_CONFIRMATION_RE.search(SOP.md)` finds the parsed marker. |
| R17 | LLM flags MULTIPLE choices as `recommended: true` in `single_choice` (semantically wrong — single_choice expects at most one) | LOW | Tool description explicitly states "at most one for single_choice"; resolver picks first-in-choice-order on collision + logs warning `[recommended-mode] tool=single_choice flagged_count=N>1; picked first`. Future enhancement: schema validator can reject at parse time, but not blocking. |
| R18 *(v1.2 — P7; v1.4 V13 mitigation reaimed)* | `RichInteractiveBase._current_input_mode` is instance state; N concurrent `SOPInferencer` instances calling `aget_input()` on the same `interactive` clobber each other's pending input mode | HIGH | `InteractionSerializer` (§9.11.3) attached via `interactive.enable_serialization()` — `asend_response`/`aget_input` route through serializer when `_serializer is not None`. Serializer's single-active-request invariant + `asyncio.Lock` guarantee one widget visible at a time. Per-caller identity via `_CURRENT_INTERACTION_CALLER` ContextVar (no call-site changes in `ConversationalInferencer`). RED test #16 (AC9.11): 3 concurrent SOPs → 3 widgets shown sequentially. Replaces the v1.2/v1.3 `RoutedInteractive + PendingConversationToolQueue` design with a cleaner transport-layer mechanism that works for any concurrent-callers scenario (not SOP-specific). |
| R19 *(v1.3 — Q8)* | Multi-focus + LLM-fallback routing path (§9.4 route iii) incurs 2 LLM calls per user message — operational cost can balloon if multiple active SOPs are in ambiguous states | MED | Rule-based (i) and structural (ii) fast paths cost 0 parent LLM calls. Cost observability: parent inferencer logs `routing_path=fast_rule|fast_struct|llm` per turn to `session.jsonl`; weekly dashboard tracks `% llm fallback` per session. Mitigation if % too high: tune the fast-path coverage (e.g., extend rule (i) to "if exactly one SOP is in `awaiting_response` state" not just "pending tool"). |
| R21 *(v1.5 — V19 bridge concurrency)* | The `_bridge_one_response` task pattern relies on one bridge task per `aget_input` call — N concurrent callers spawn N bridge tasks, but only 1 user response arrives per round-trip. Risk: bridge task that loses the race holds a reference to the cancelled `asyncio.create_task` until GC; or worse, a slow `_input_queue.get()` could leak if the parent's event loop is destroyed mid-await | LOW | (a) Explicit `try/finally` cancels bridge_task on every aget_input return path (§9.11.2.1 lines 13-15); (b) `_bridge_one_response` catches `asyncio.CancelledError` and returns cleanly; (c) AC9.15 sub-test (d) asserts no leaked tasks remain in `asyncio.all_tasks()` after the 3-caller scenario completes; (d) if leak observed in production, fallback design is a single long-lived bridge worker task spawned at `enable_serialization()` time (simpler but adds a permanent task per session — currently NOT chosen because lazy spawning has zero idle cost). |
| R20 *(v1.4 — V14)* | `WebSocketInteractive` now inherits `InteractiveBase` (was duck-typed). Cross-package change: AgentFoundation base class change affects OpenStartup. Risk: subtle behavior diff if any current call site assumed the duck-typed (non-inheriting) behavior — e.g., `isinstance(interactive, InteractiveBase)` checks that were previously False would become True | MED | Pre-merge: (a) grep `isinstance.*InteractiveBase` across both repos — verify no behavior depends on the negative case; (b) full OpenStartup test suite must pass; (c) add explicit test in `OpenStartup/tests/services/test_websocket_interactive_inheritance.py` asserting that all 4 protocol methods (`stream_token_batches`, `send_turn_boundary`, `asend_response`, `aget_input`) still behave correctly when called via the duck-typed path AND via the inherited path; (d) sync abstract stubs raise `NotImplementedError("use async methods")` — never silently default to `pass`. |

---

### §14. Verification

#### §14.1 Per-PR smoke tests (run sequentially)

- **PR-0:** All 14 RED tests fail with `xfail(strict=True)` annotations; parser-extra-fields test passes or 1-line patch lands.
- **PR-1:** `python -c "from agent_foundation.resources.sops.registry import SOPRegistry; from pathlib import Path; print(list(SOPRegistry().load_all(extra_dirs=[Path('OpenStartup/src/openteam/server/resources/sops')]).keys()))"` returns `['code_optimization', 'model_optimization', 'role_creation']`. Start OpenStartup mock backend (`./src/openteam/run.sh`), open a fresh session, send "hello" — rendered prompt at `_runtime/.../turn_001/rendered_prompt.txt` contains a `## Available SOPs` block listing all three.
- **PR-2:** Start session (default `multi`). Enter `role_creation`. Enter `code_optimization`. Rendered prompt's `## Active SOPs` shows both at appropriate priority. `/sop --mode single` → most-recently-progressed becomes active; others suspended with notification.
- **PR-3:** Start with `--real-sessions --llm-backend claude_cli`. Enter `role_creation` SOP with `/sop role_creation --yolo`. Confirm: (a) no widget pops in UI, (b) `<session>/sops/role_creation__.../turn_001/metadata.json` shows `"source": "synthetic"`, (c) `user_input.txt` contains the synthesized response per `role_creation`'s `yolo_overrides` `first_choice`, (d) SOP advances past Phase 0 without human input.
- **PR-4:** After PR-3: verify `_runtime/.../sessions/<sid>/sops/role_creation__<TS>__<id>/` exists with `sop_state.json`, `sop_definition_snapshot.json`, `session.jsonl`, multiple `turn_NNN/` folders. Verify `tasks/create_role__<TS>__<task_id>/outputs/` populated under the SOP folder (NOT under `<session>/tasks/`). Verify parent `session_state.json` has `active_sops: [{... "sop_run_folder": "sops/role_creation__...", ...}]`. Verify parent `session.jsonl` has one `SOPInvoked` record.
- **PR-5:** Pre-change baseline: capture `turn_001/rendered_prompt.txt` from a clean OpenStartup session with "hello". Post-change comparison: same session + message. Diff shows only the new `## Available SOPs` block (when SOPs exist); existing prompt content unchanged. Non-yolo conversation tool flow: send a message that triggers a `single_choice` widget; verify the widget pops in UI exactly as today (no synthesis kicks in unless `yolo_mode=True`).

#### §14.2 End-to-end tests

- `test_role_creation_sop_e2e.py`: drives `role_creation` Phase 0 → Phase 3 in yolo against a mock LLM. Asserts file tree from PR-4 verification + `source: "synthetic"` in every turn metadata.
- `test_multi_active_sops.py`: enters 2 SOPs in multi_focus; sends messages targeting each individually; asserts independent tracker progression + correct attribution.
- `test_sop_mode_switching.py`: round-trips single ↔ multi with N=1, 2, 3 active SOPs; asserts focus-winner selection logic + notification messages.

#### §14.3 Manual regression checklist

1. Non-yolo single-SOP flow: identical to today (widget pops as expected; no synthesis path triggered).
2. Mid-session mode switch: in-flight widget completes with current-mode semantics; next turn shows new layout.
3. Server restart preserves `active_sops[]` and `focus_mode`: resume session, verify state restored.
4. Concurrent sessions: 2 sessions in same server with different `focus_mode` settings render independently.
5. Kill-switch: `OPENTEAM_SOP_FORCE_SINGLE_FOCUS=true` set → restart server → all sessions force-single regardless of session config; `EnvOverride` logged.

---

### §15. Out of scope (explicitly NOT in this plan)

- **WorkGraph/StateGraph substrate decisions** (covered by v7.2). This plan adopts the tracker-driven choice for enter-and-stay-active and retains WorkGraph for `/sop --autonomous`; no further substrate changes.
- **Skill registry format changes:** YAML frontmatter under `<name>/SKILL.md` stays as-is. Only SOPs use the `.md + .json` pattern.
- **Conversation tool TYPE additions:** the four existing types + `tool_argument_form` keep their core schemas; only `yolo_default` is added.
- **SOP grammar additions beyond `__keywords__` + `__example_requests__`:** v7.2's `__depends on__`, `__for_each__`, `__goto__`, `__if__`, `__initial__`, `__branch__`, `__requires confirmation__` are all retained.
- **Database persistence:** filesystem-only stays (per OpenStartup CLAUDE.md). No new DB tables.
- **OpenStartup tool topologies, run.sh, backend registry:** unchanged.
- **Multi-LLM-call routing classifier:** routing is done by the orchestrator LLM in its existing decision-procedure step; no separate classifier model added.
- **Multi-server SOP sharing:** SOPs live per-server-process; no cross-server registry sync.
- **SOP versioning beyond `version` field:** no migration tooling for old `sop_state.json` schemas.

---

### §16. Plan ledger and archive plan

| File | Lines | Status |
|---|---|---|
| **`sop_framework_UNIFIED_v1_plan.md`** | (this file) | **← CURRENT (single source of truth)** |
| `sop_runtime_enablement_plan.md` | 904 | Superseded — to be archived after review |
| `multi_sop_focus_and_tool_concurrency_plan.md` | 665 | Superseded — to be archived after review |
| `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md` | 1352 | Companion (overall architecture — NOT superseded; this plan layers on top) |
| `conversational_workflows_and_sop_framework_INTEGRATED_v2_plan.md` | ~1300 | Historical (already pre-v7.2) — already archive candidate |
| `conversational_workflows_and_sop_framework_INTEGRATED_v7_plan.md` | ~1290 | Historical (already pre-v7.2) — already archive candidate |
| `conversational_workflows_and_sop_framework_plan.md` | 586 | Historical (v1) — already archive candidate |

**Archive procedure** (run after review of this unified plan):
```bash
cd CoreProjects/AgentFoundation/_docs/_plan/workflows_and_sop/
mkdir -p _archive/
mv sop_runtime_enablement_plan.md _archive/
mv multi_sop_focus_and_tool_concurrency_plan.md _archive/
mv conversational_workflows_and_sop_framework_INTEGRATED_v2_plan.md _archive/
mv conversational_workflows_and_sop_framework_INTEGRATED_v7_plan.md _archive/
mv conversational_workflows_and_sop_framework_plan.md _archive/
# v7.2 stays at top level as the companion architecture plan
```

After archive, only 2 files remain at top level:
- `sop_framework_UNIFIED_v1_plan.md` (this plan — runtime/registry/queue/mode)
- `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md` (architecture — WorkGraph substrate, StateGraph foundation)

---

### §17. Honest comparison — if you had to pick ONE plan today

If forced to pick a single plan to implement and **not** allowed to consolidate:

| Pick | Verdict | Reasoning |
|---|---|---|
| `sop_runtime_enablement_plan.md` v3.1 alone | NOT recommended | Missing mode system; missing queue; missing concurrency labeling. Cannot implement `role_creation` end-to-end if user wants multi-SOP. |
| `multi_sop_focus_and_tool_concurrency_plan.md` v1.1 alone | NOT recommended | Assumes Plan #1's foundation (SOPs as resources, yolo synthesis, runtime layout). Standalone has no SOPs to run. |
| External "v8" alone | NOT recommended | Missing queue; missing `ToolDefinition.concurrency` enum; envelope-format and inner/outer inferencer gaps. |
| **This unified plan** | ✅ **STRONGLY RECOMMENDED** | Single source of truth combining best parts of all three; 3 architectural disagreements explicitly resolved with rationale; empirically-verified throughout; PR-sequenced for incremental rollout. |

The unified plan strictly dominates each predecessor on coverage, while correcting the empirical errors in each (yolo response shape from Plan #1; routing envelope from Plan #2; missing queue from v8).


