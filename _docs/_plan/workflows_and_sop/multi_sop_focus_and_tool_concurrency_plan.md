# Multi-SOP Focus & Tool Concurrency Plan

**Author:** Rovo Dev (drafted in conversation with Tony Chen)
**Date:** 2026-05-26 18:46 (v1.1 audit patches applied)
**Status:** Draft v1.1 — audit-corrected; ready for re-review

### §-1. Audit history

| Round | Date | Reviewer | Verdict | Patches applied |
|---|---|---|---|---|
| v1.0 | 2026-05-26 16:14 | (initial draft) | — | — |
| v1.1 | 2026-05-26 18:46 | external agent (10-issue audit) | 9 valid, 1 fabricated | 3 HIGH (envelope format, inner/outer split, `prior_context` attribute) + 4 MED (P0 cap, re-render-after-delivery, fragile routing-first, missing writer for `last_user_input_advanced_at`) + 2 LOW (feature-flag precedence; the singular-`sop/` claim was empirically false → rejected) |

**Sibling to:** `sop_runtime_enablement_plan.md` (this plan extends concerns #4 and #5 of that plan with multi-active-SOP support and formalizes tool concurrency semantics)
**Companion to:** `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md` (the v7.2 architecture stands; this plan adds two new orthogonal capabilities on top)

---

## §0. Why this is a separate plan

The `sop_runtime_enablement_plan.md` plan (v3.1) makes one SOP run end-to-end. This plan adds two orthogonal capabilities that change the **shape** of SOP-runtime interaction:

1. **Multi-focus mode** — N SOPs active simultaneously, with LLM-driven routing of user input across them.
2. **Tool concurrency labeling** — a first-class `concurrency: blocking | async_background | async_awaitable` field on `ToolDefinition` (today's `asynchronous: bool` collapses 2 of 3 cases).

Both capabilities affect more than just SOPs (tool concurrency labeling helps any compound tool dispatch; multi-focus exercises the conversation-tool queue that's needed even outside SOPs when multiple background tools complete simultaneously). Bundling them into the runtime-enablement plan would have made that plan unfocused. This plan can be implemented either **before** or **after** runtime enablement: it's strictly additive.

**Non-goals:**
- Re-litigating the runtime-enablement plan's decisions on frontmatter, yolo synthetic dispatch, registry, or session storage.
- Cross-SOP tool fan-out (e.g., one user command triggering parallel phases across two SOPs simultaneously) — that's a future direction, not v1.
- Compound widgets across SOPs (one UI prompt collecting answers for two SOPs at once) — explicitly deferred; queue + single-head is the v1 path.

---

## §1. Empirical baseline (verified 2026-05-26 16:12)

| Component | Path | State |
|---|---|---|
| `ToolDefinition.asynchronous: bool` | `AgentFoundation/.../resources/tools/models.py:105` | ✅ Exists; comment: "Fire-and-forget: tool runs in background, turn completes immediately" |
| Conversation tool concurrency flag | (none — implicit `blocking`) | ❌ No flag; all conversation tools assumed blocking |
| Tool queue concept | (none) | ❌ No `PendingConversationToolQueue` or similar; tools dispatch turn-by-turn sequentially |
| `asyncio.gather` in conversation inferencer | `conversational_inferencer.py` | ❌ Zero usages — tools today run sequentially per turn |
| `WorkflowManager` (per session) | v7.2 §6 | ✅ Exists with `active_instances: dict[str, WorkflowInstance]` |
| `WorkflowInstance.status` | v7.2 §6 | ✅ Has `active/suspended/completed`; no `pending_conversation_tool` field |
| `_handle_conversation_tools` compound dispatcher | `conversational_inferencer.py:1099` | ✅ Returns `Optional[dict[str, str]]` of `{var_name: value}`; sequential per tool |

**Implications:**
- Tool concurrency labeling is genuinely missing — adding it is a real schema enhancement, not a rename.
- The conversation tool path today handles compound-in-one-turn correctly but **does NOT** handle compound-across-turns (no queue). Multi-focus surfaces the gap.
- v7.2 already has the substrate (`WorkflowManager`, `WorkflowInstance`) — this plan adds `focus_mode` + `pending_conversation_tool`, not new substrate.

---

## §2. Design — Multi-focus mode

### §2.1 Attribute on `WorkflowManager`

```python
@attrs(slots=False, kw_only=True)
class WorkflowManager:
    # ... existing v7.2 fields ...
    focus_mode: Literal["single", "multi"] = "multi"   # NEW — default multi per Tony 2026-05-26
    pending_tool_queue: "PendingConversationToolQueue" = attrib(factory=lambda: PendingConversationToolQueue())  # NEW
```

**Default:** `multi` — but the multi-mode prompt cost is **conditional**: when `len(active_instances) <= 1`, the prompt renders identically to single-mode (no routing block, no priority headers). So multi-as-default has zero overhead in the common single-SOP case.

**Configuration sources** (v1.1 patch — Claim 9: explicit precedence):

Resolution order (first hit wins):
1. Session-init payload: `{"focus_mode": "single" | "multi"}` (per-session override; persisted in `session_metadata.json`)
2. Feature flag `OPENTEAM_USE_MULTI_FOCUS_SOP` if `false` → forces `"single"` regardless of below
3. Server-side default in `sop_config.json` (per-deployment default; defaults to `"multi"`)
4. Hardcoded fallback: `"multi"`

**Feature flag semantics (v1.1):** `OPENTEAM_USE_MULTI_FOCUS_SOP=false` is the kill-switch. When set, `WorkflowManager.__init__` coerces `focus_mode` to `"single"` **regardless** of session-init payload or server default. The session log records the coercion as an `EnvOverride` event in `session.jsonl`. Once Phase 8 lands and the flag flips to default `true`, the kill-switch remains available for emergency rollback. **AC1 strengthened** in §7.

### §2.2 `single` mode semantics

- At most 1 SOP can be `status=active` at a time.
- Entering a new SOP via `enter_workflow(new_id)` **auto-suspends** the currently-active one:

  ```python
  async def enter_workflow(self, def_id: str, **kwargs) -> WorkflowInstance:
      if self.focus_mode == "single":
          for inst in list(self.active_instances.values()):
              if inst.status == "active":
                  await self.suspend_workflow(inst.instance_id)
      # ... create new instance ...
  ```

- Prompt rendering: classic v3.1 §5.5a — one full block for the active SOP; suspended SOPs as one-line summaries.
- Routing: implicit — user input always belongs to the (single) active SOP.

### §2.3 `multi` mode semantics

- N SOPs can be `status=active` simultaneously. Each runs as an independent `asyncio.Task` (the WorkGraph for each instance is its own task).
- No auto-suspension on `enter_workflow`.
- Prompt renders all active SOPs with **priority-based budget allocation** (see §2.5).
- Routing: explicit — when `len(active_instances) >= 2`, the prompt includes a routing instruction; the LLM emits `{"type":"workflow_route","ids":[<id>|"none"|<id_a>,<id_b>]}` as the first tool call.

### §2.4 Mid-session toggle

**Allowed.** Via a new conversation tool `set_focus_mode`:

```json
{"type":"sop","action":"set_focus_mode","mode":"single"|"multi"}
```

**Behavior on toggle:**

| From → To | Action | Suspended SOPs |
|---|---|---|
| single → multi | All current `active` and `suspended` stay as-is. Prompt re-renders with multi layout next turn. | Unchanged |
| multi → single | Pick the focus winner: `P0` (has pending tool) → else `most recent activity` → else `first by creation_ts`. Auto-suspend others. | Newly suspended (the non-winners) |

**Mode-switch notification** — the next turn's prompt preamble includes:

```
[Mode switch] Focus mode is now {mode}. {role_creation} is active{; <others> are suspended (resume with `resume_workflow('id')`)}.
```

This prevents the LLM from being surprised by a prompt-layout change.

**`session_metadata.json`** persists `focus_mode` so a resumed session restores the correct mode.

### §2.5 Priority-based rendering (`multi` mode only, only when `len(active) >= 2`)

| Priority | Condition | Rendering | Budget |
|---|---|---|---|
| **P0** | `pending_conversation_tool is not None` (a tool fired and is awaiting user response) | FULL context + highlighted "AWAITING RESPONSE: <tool_type>: '<prompt_excerpt>'" | Up to `max_p0_full_render` (default 3); excess P0 demote to compact with warning (v1.1 — Claim 5) |
| **P1** | `last_user_input_advanced_this_sop is True` (the most recent user-input turn advanced this SOP — event-driven, not wall-clock) | FULL context | Share remaining budget proportionally |
| **P2** | Active but neither P0 nor P1 (running, e.g., long-async tool in progress, or just-entered but no input yet) | COMPACT — name + phase + one-line status + `is_computing: bool` flag | One-line per SOP |
| **P3** | `status="suspended"` | ONE-LINE — `name (instance_id) — suspended; resume with resume_workflow('id')` | One-line per SOP |

**Deterministic budget algorithm:**

```python
def allocate_budget(sops: list[WorkflowInstance], total_budget: int = 2000) -> dict[str, int]:
    p0 = [s for s in sops if s.priority == "P0"]
    p1 = [s for s in sops if s.priority == "P1"]
    p2 = [s for s in sops if s.priority == "P2"]
    p3 = [s for s in sops if s.priority == "P3"]

    # P0 always full
    p0_alloc = {s.id: estimate_full_context_tokens(s) for s in p0}
    used = sum(p0_alloc.values())
    remaining = max(0, total_budget - used)

    # P1 shares remaining proportionally
    if p1 and remaining > 0:
        per_p1 = remaining // len(p1)
        p1_alloc = {s.id: per_p1 for s in p1}
    else:
        p1_alloc = {s.id: COMPACT_TOKEN_BUDGET for s in p1}  # demote to compact

    # P2 and P3 always compact / one-line (no budget pressure)
    return {**p0_alloc, **p1_alloc, **{s.id: COMPACT_TOKEN_BUDGET for s in p2}, **{s.id: ONE_LINE_TOKEN_BUDGET for s in p3}}
```

`total_budget` default `2000` configurable via `sop_config.json`.

### §2.6 Prompt template — multi-mode routing block (v1.1 — Claim 1 envelope fix)

**Real tool-call envelope** (verified at `tool_call_parser.py`): `<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>`. v1's `{"type":"workflow_route", "ids":[...]}` did NOT match. v1.1 uses the real shape:

```
<tool_call>{"name": "workflow_route", "arguments": {"ids": ["<instance_id>" | "none" | "<id_a>", "<id_b>"]}}</tool_call>
```

`workflow_route` is registered as a tool name (handled in-process by the inferencer, not dispatched to an external action tool). The parser sees `name="workflow_route"` and routes to a special pre-dispatch handler that consumes the route *before* other tools execute. See §5 for envelope-vs-`{"type":"sop"}` unification.

`active_sop_routing.jinja2` (NEW; rendered only when `focus_mode == "multi" AND len(active_instances) >= 2`):

```jinja
# Step 0 — Active Workflow Routing

You have {{ active_sop_count }} active workflows. Before responding, identify which one(s)
the user's message relates to. Heuristics:
- If a workflow has "AWAITING RESPONSE", the user's message is most likely answering that.
- If the message mentions a workflow by name or topic, route to that workflow.
- If the message is unrelated to any workflow, treat as chitchat (route `"none"`).
- If genuinely ambiguous, ask the user which workflow they mean (use plain prose; no tool call).

Emit your routing decision as a tool call (preferably FIRST; see Claim 7 fallback):
<tool_call>{"name": "workflow_route", "arguments": {"ids": ["<instance_id>" | "none" | "<id_a>", "<id_b>"]}}</tool_call>

# Active Workflows (by priority)

{% for sop in active_sops_by_priority %}
## {{ sop.name }} ({{ sop.instance_id }}){% if sop.has_pending_tool %} — AWAITING RESPONSE{% endif %}

{% if sop.has_pending_tool %}
**Pending tool:** {{ sop.pending_tool_type }} — "{{ sop.pending_tool_prompt }}"
{% endif %}
{% if sop.render_mode == "full" %}
{{ sop.full_context }}
{% elif sop.render_mode == "compact" %}
Status: {{ sop.one_line_status }}{% if sop.is_computing %} (computing){% endif %}
{% else %}{# one_line #}
{{ sop.name }} ({{ sop.instance_id }}) — {{ sop.status }}{% if sop.status == "suspended" %}; resume with `resume_workflow('{{ sop.instance_id }}')`{% endif %}
{% endif %}

{% endfor %}
```

### §2.7 Routing scenarios (formalized as ACs)

| Scenario | Active SOPs | User input | Expected routing |
|---|---|---|---|
| A — Clear (single P0) | role_creation (P0: pending confirmation "Approve role doc?"), code_opt (P2 running) | "yes, looks good" | `{"ids":["role_creation_<id>"]}` |
| B — Ambiguous (multi P0) | role_creation (P0: pending confirmation), code_opt (P0: pending clarification "What target path?") | "yes" | LLM responds with plain prose asking user to disambiguate; no routing tool call |
| C — New SOP entry | role_creation (P1 active in Phase 1) | "also optimize the pipeline at src/pipeline/" | `enter_workflow("code_optimization")` → 2 active SOPs |
| D — Meta-query | (any) | "what's the status of all my workflows?" | `{"ids":["none"]}`; LLM summarizes from the rendered Active SOPs section |
| E — Invalid routing ID | role_creation (P1) | (LLM hallucinates ID) | Runtime validates `ids` against `active_instances`; on miss, treats as `"none"` + logs warning |
| F — Single-active-SOP optimization | role_creation only | "yes" | No routing block in prompt (because `len(active) == 1`); user input goes directly to role_creation |

### §2.8 Routing audit trail

Each turn appends a `RoutingDecision` event to `session.jsonl`:

```jsonc
{"type": "RoutingDecision", "turn": 7, "ts": "...",
 "active_sop_count": 3,
 "pending_tools_count": 1,
 "llm_routed_to": ["role_creation_abc12345"],
 "routing_valid": true,        // false if invalid ID was returned
 "routing_fallback": null      // "none" if invalid → fallback to chitchat
}
```

---

## §3. Design — Tool concurrency labeling

### §3.1 Three-value enum on `ToolDefinition`

```python
@dataclass
class ToolDefinition:
    # ... existing fields ...
    concurrency: Literal["blocking", "async_background", "async_awaitable"] = "blocking"
    # Backward compat:
    # - asynchronous: bool   (deprecated; derived)

    @property
    def asynchronous(self) -> bool:
        """DEPRECATED — use concurrency."""
        return self.concurrency != "blocking"
```

**Semantics:**

| Value | Meaning | Turn semantics | Examples |
|---|---|---|---|
| `"blocking"` | Must complete before turn advances | Agentic loop awaits result | All conversation tools today; fast sync actions |
| `"async_background"` | Fire-and-forget; turn advances immediately; result delivered via callback or polled later | Loop continues; result joins via separate mechanism | `research_propose` (today: `asynchronous=True`); long-running subagents |
| `"async_awaitable"` | Concurrently runnable within a turn; loop fans out via `asyncio.gather`; turn advances when all complete | Loop awaits gather | NEW — not exercised today; future fan-out (e.g., parallel API calls in one turn) |

**Backward-compat migration:**
- Existing `tool.json` files with `"asynchronous": true` → derived as `"concurrency": "async_background"`
- Existing `tool.json` files without `"asynchronous"` → derived as `"concurrency": "blocking"`
- New `tool.json` should use `"concurrency"` explicitly
- `ToolDefinition.from_dict()` accepts both forms; emits `DeprecationWarning` on `"asynchronous"` in 2 releases

### §3.2 Conversation-tool concurrency

All conversation tools (`clarification`, `confirmation`, `multiple_choice`, `single_choice`, `tool_argument_form`) keep `concurrency: "blocking"` — they fundamentally require user response before the turn can complete. **They have no `async_*` variants.**

Action tools opt into `async_background` (already supported via `asynchronous: bool`) or `async_awaitable` (NEW — not used in v1; reserved for future).

### §3.3 Tool concurrency × Multi-focus interaction

Critical insight (the reason this plan unifies the two topics):

| Concurrency | Multi-focus implication |
|---|---|
| `blocking` (conversation tools) | **MUST queue** across SOPs — shared user attention (only one prompt at a time) |
| `async_background` (research_propose et al) | **No queue needed** — each SOP's WorkGraph asyncio.Task runs the tool independently; results route back via the SOP's own task |
| `async_awaitable` (future) | **No cross-SOP queue needed** — each SOP gathers its own concurrent calls |

**Therefore:** The `PendingConversationToolQueue` (§4) is **specifically for `blocking` tools fired by different SOPs**. Other concurrency types don't need cross-SOP coordination.

---

## §4. Design — `PendingConversationToolQueue`

### §4.1 The problem

Today: one SOP, one `aget_input()` at a time. Trivial.

Multi-focus: two SOPs' phases may fire conversation tools concurrently (different WorkGraph asyncio tasks). Both block on `aget_input()`. The user can only answer one prompt at a time.

**Without a queue:** `aget_input()` calls race; whoever's first wins; whoever's second silently hangs or gets routed the wrong response.

**With a queue:** all pending tools are tracked centrally; only one is "head" (actively presented to user); others wait. LLM's routing decision determines which queue entry the user response satisfies.

### §4.2 Single-head + multi-display semantics

Per Tony's decision 2026-05-26 16:14:
- **Single-head:** Only one tool is actively bound to `aget_input()` at any time — the one at the queue head
- **Multi-display:** The prompt's "AWAITING RESPONSE" lines list **all** P0 tools (head + queued), so the LLM and user can see what's coming
- When the user responds: LLM routing decision says which SOP the response is for; if it's the head, deliver and pop; if it's a non-head queued tool, **reorder the queue** (move that tool to head) and re-present

This keeps the user's UI simple (one prompt at a time) while letting the LLM be smart about which question gets answered.

### §4.3 API

```python
@dataclass
class PendingTool:
    instance_id: str              # which WorkflowInstance fired this
    tool: ConversationTool
    future: asyncio.Future        # the WorkGraph node awaits this
    enqueued_at: datetime
    prompt_excerpt: str           # for "AWAITING RESPONSE" display

class PendingConversationToolQueue:
    """Per-session queue of blocking conversation tools across multiple active SOPs."""

    _queue: deque[PendingTool]

    async def enqueue(self, instance_id: str, tool: ConversationTool) -> asyncio.Future:
        """Called from inside a WorkGraph node before aget_input()."""
        future = asyncio.get_event_loop().create_future()
        self._queue.append(PendingTool(instance_id, tool, future, datetime.utcnow(), tool.prompt[:80]))
        return future

    def head(self) -> Optional[PendingTool]:
        """The tool currently presented to the user (or None if empty)."""
        return self._queue[0] if self._queue else None

    def all(self) -> list[PendingTool]:
        """All queued tools (for multi-display in the prompt)."""
        return list(self._queue)

    def reorder_to_head(self, instance_id: str) -> None:
        """LLM routed user response to a non-head tool → move it to head and re-present."""
        for i, pt in enumerate(self._queue):
            if pt.instance_id == instance_id:
                if i != 0:
                    self._queue.rotate(-i)
                return
        raise KeyError(f"No queued tool for instance {instance_id}")

    def deliver(self, instance_id: str, response: str) -> None:
        """Called from the inferencer when user responds + LLM routing identifies SOP."""
        if not self._queue or self._queue[0].instance_id != instance_id:
            self.reorder_to_head(instance_id)
        head = self._queue.popleft()
        # v1.1 Claim 8: writer for P1 priority — record that this instance was advanced by user input
        instance = self._workflow_manager.active_instances.get(instance_id)
        if instance:
            instance.last_user_input_advanced_at = datetime.utcnow()
            instance.pending_conversation_tool = None  # clear P0 marker (also in v1 §4.5)
        head.future.set_result(response)
```

### §4.4 Integration with `_handle_conversation_tools` — v1.1: inner vs outer inferencer + correct attribute

**Critical v1.1 correction (Claims 2 + 3): two inferencers, not one.** Per v7.2 §6 / sibling-plan §6.2, an SOP runs as a **sub-inferencer** (`SOPSubInferencer` wraps `ConversationalInferencer` inside `SOPWorkGraphNode._execute_phase`). The user-facing inferencer is the **outer** inferencer. Each has a distinct role in the queue lifecycle:

| Inferencer | Role | Method |
|---|---|---|
| **Inner** (per SOP, runs in `SOPWorkGraphNode`) | Fires conversation tool → enqueues + awaits future | `_handle_conversation_tools` (modified) |
| **Outer** (user-facing, one per session) | Renders prompt with queue state; routes user response to queue.deliver(); loops to re-render after delivery | new `_handle_pending_tool_delivery` method |

**`self._current_sop_context` (v1.1 — Claim 3): use `self.prior_context["sop_instance_id"]`** — `prior_context: dict[str, Any]` exists at `conversational_inferencer.py:107`. The SOPWorkGraphNode sets it via `set_prior_context({..., "sop_instance_id": instance.id})` (line 525) before invoking the inner inferencer.

**Inner inferencer** (per SOP, inside `SOPWorkGraphNode._execute_phase`):

```python
async def _handle_conversation_tools(self, tools, ...) -> Optional[dict[str, str]]:
    if self.yolo_mode:
        return self._synthesize_yolo_collected(tools)

    # v1.1 Claim 3 fix: use prior_context, not nonexistent _current_sop_context
    instance_id = self.prior_context.get("sop_instance_id")

    if self.workflow_manager and instance_id:
        # Multi-focus path: enqueue + await future
        collected = {}
        futures = []
        for tool in tools:
            f = await self.workflow_manager.pending_tool_queue.enqueue(instance_id, tool)
            futures.append((tool.output_variable, f))
        # Await all — head is actively presented; others sit in queue
        for var_name, f in futures:
            collected[var_name] = await f
        return collected

    # Non-SOP / single-focus path: original behavior
    return await self._original_interactive_handler(tools)
```

**Outer inferencer** (user-facing; v1.1 NEW):

```python
async def _handle_pending_tool_delivery(self, user_input: str) -> bool:
    """Called by the outer inferencer when user responds AND there's a queue head.

    Returns True if delivered (and outer loop should re-render); False otherwise.
    """
    queue = self.workflow_manager.pending_tool_queue
    if not queue.head():
        return False

    # Routing already happened (workflow_route consumed pre-dispatch); use its result
    routed_to = self.last_routing_decision.target_instance_id  # set by workflow_route handler
    if routed_to and routed_to != queue.head().instance_id:
        queue.reorder_to_head(routed_to)

    head = queue.head()
    queue.deliver(head.instance_id, user_input)
    # v1.1 Claim 6: re-render loop — keep delivering until no new P0 surfaces
    return True

# v1.1 Claim 6: outer agentic loop re-renders after delivery
async def _run_outer_turn(self, user_input: str) -> None:
    while await self._handle_pending_tool_delivery(user_input):
        # SOP advanced; may have fired a new tool. Yield control briefly to let
        # the inner inferencer's WorkGraph task enqueue if it will, then re-render.
        await asyncio.sleep(0)  # event-loop yield
        if not self.workflow_manager.pending_tool_queue.head():
            break
        # Re-render prompt with new queue state; LLM picks next response.
        # (No new user input — the LLM continues the conversation with itself.)
        new_response = await self._inference_one_turn()
        user_input = new_response  # treat LLM's continuation as the next "input" to route
```

**Synthetic response format (v1.1 — Claim 4): cross-reference to sibling plan.** `_synthesize_yolo_collected` shapes are spec'd in `sop_runtime_enablement_plan.md` §4.4 (v3.1) — verified against `_process_widget_response` semantics. Do NOT use the invented `{"selected":[...]}` / `{"confirmed": True}` shapes from earlier drafts. Use `choice_index`, `choice`, `custom_text`, `param_overrides`, `variable_override`, or `values` keys as documented in the sibling plan's v3.1 §4.4 table.

**Pre-dispatch parser hook for `workflow_route` (v1.1 — Claim 1 & 7 polish):**

```python
def split_routing_from_tools(parsed_calls: list[ParsedToolCall]) -> tuple[ParsedToolCall|None, list[ParsedToolCall]]:
    """Find the workflow_route call (anywhere in list, not just first); peel it off."""
    routing = next((c for c in parsed_calls if c.name == "workflow_route"), None)
    remaining = [c for c in parsed_calls if c.name != "workflow_route"]
    return routing, remaining
```

The pre-dispatch hook consumes `workflow_route` regardless of position (Claim 7: not strictly first). If absent AND user input is unambiguous (e.g., only 1 SOP is P0), infer routing from the pending tool's instance_id. Only require explicit `workflow_route` when inference is ambiguous (multi-P0).

### §4.5 Mark `WorkflowInstance.pending_conversation_tool`

When the queue enqueues a tool, it also sets:
```python
instance = workflow_manager.active_instances[instance_id]
instance.pending_conversation_tool = {"tool_type": tool.tool_type, "prompt_excerpt": tool.prompt[:80]}
```

This makes the SOP visible in the prompt as P0 priority and lets the rendering logic show "AWAITING RESPONSE" without needing to inspect the queue.

When `deliver()` is called: clear `pending_conversation_tool = None`.

---

## §5. Tool envelope unification

v3 §5.7 specified `{"type":"sop","name":"<sop>"}` for SOP entry. v3.2 unifies this into a single dispatchable envelope:

```json
{"type":"sop","action":"enter","name":"role_creation"}
{"type":"sop","action":"resume","instance_id":"abc12345"}
{"type":"sop","action":"suspend","instance_id":"abc12345"}
{"type":"sop","action":"set_focus_mode","mode":"single"}
```

`tool_call_parser.py` recognizes the `{"type":"sop","action":...}` shape and dispatches to the matching `WorkflowManager` method. Additionally, `{"type":"workflow_route","ids":[...]}` (§2.6) is recognized in `multi` mode and processed by the inferencer before any other tool calls in the turn.

**`set_focus_mode` is also exposed via UI** (separate from the LLM-callable tool) — a UI button in the conversation panel. UI invocation calls the same `WorkflowManager.set_focus_mode(mode)` method; the LLM is informed via the next-turn mode-switch notification (§2.4).

---

## §6. New `WorkflowInstance` fields

```python
@attrs(slots=False, kw_only=True)
class WorkflowInstance:
    # ... existing v7.2 fields ...
    pending_conversation_tool: Optional[dict] = None    # {tool_type, prompt_excerpt}; set by queue, cleared on deliver
    last_user_input_advanced_at: Optional[datetime] = None  # for P1 priority; written by queue.deliver() and workflow_route handler (v1.1 — Claim 8)
    is_computing: bool = False                          # true while an async_background tool is in flight
```

**Priority derivation** (`WorkflowInstance.priority` property):

```python
@property
def priority(self) -> Literal["P0", "P1", "P2", "P3"]:
    if self.status == "suspended":
        return "P3"
    if self.pending_conversation_tool is not None:
        return "P0"
    if self.last_user_input_advanced_at is not None and self._is_most_recently_advanced():
        return "P1"
    return "P2"
```

`_is_most_recently_advanced()` consults the `WorkflowManager`'s session-wide most-recent-advance timestamp; this is **event-driven, not wall-clock** (addresses Issue 2 from prior critique).

---

## §7. Acceptance criteria

### Multi-focus

- **AC1.** Default `focus_mode == "multi"` on a fresh session unless overridden.
- **AC2.** With `focus_mode == "multi"` and exactly 1 active SOP, the rendered prompt contains NO routing block (no token cost in the common case).
- **AC3.** With `focus_mode == "multi"` and 2+ active SOPs, the rendered prompt contains the §2.6 routing block.
- **AC4.** Scenario A (clear P0 routing): `{"ids":["role_creation_<id>"]}` is the LLM's first tool call; the response delivers to role_creation's queued tool.
- **AC5.** Scenario B (ambiguous multi-P0): LLM emits plain-prose disambiguation question; no tool call; both SOP futures remain unresolved.
- **AC6.** Scenario C (new SOP entry while another active): `enter_workflow("code_optimization")` succeeds without auto-suspending role_creation; both end up active.
- **AC7.** Scenario D (meta-query): `{"ids":["none"]}` is emitted; LLM summarizes from the Active SOPs section.
- **AC8.** Scenario E (invalid ID): runtime treats invalid `ids` as `"none"` + appends `RoutingDecision.routing_valid: false` to `session.jsonl`.
- **AC9.** Scenario F (single-active in multi mode): prompt looks identical to single-mode (no routing block, no priority headers).
- **AC10.** `set_focus_mode("single")` mid-session: most-recent-P0/P1 stays active; others become suspended; next-turn prompt has mode-switch notification.
- **AC11.** `set_focus_mode("multi")` mid-session: active SOPs unchanged; next-turn prompt has multi-layout (if 2+ active).
- **AC12.** `session_metadata.json.focus_mode` persists; resumed sessions restore the correct mode.

### Token budget

- **AC13.** `total_budget = 2000` (default) is configurable in `sop_config.json`.
- **AC14.** P0 SOPs always receive their estimated full-context budget; P1 SOPs share remaining proportionally; P2/P3 always compact/one-line.
- **AC15.** With 5 P0 SOPs (`max_p0_full_render` default = 3): first 3 by `enqueued_at` get FULL rendering; SOPs #4 and #5 demote to COMPACT (one-line) with a warning emitted ("P0 cap reached: 2 SOP(s) demoted; consider suspending less-urgent SOPs"). (v1.1 — Claim 5)

### Tool concurrency

- **AC16.** `ToolDefinition.concurrency` round-trips correctly from `tool.json` (both new `concurrency` and legacy `asynchronous` accepted; legacy emits `DeprecationWarning`).
- **AC17.** All existing conversation tools (`clarification`, `confirmation`, etc.) load with `concurrency == "blocking"` by default.
- **AC18.** Existing `research_propose` (today `asynchronous: True`) loads with `concurrency == "async_background"` via backward-compat derivation.
- **AC19.** A new `async_awaitable` tool can be loaded; integration test verifies `asyncio.gather` semantics (deferred — no v1 tool uses this).

### Queue

- **AC20.** Two SOPs fire conversation tools in the same agentic-loop iteration: both are enqueued; only one is bound to `aget_input()`; the other's WorkGraph node awaits its future without racing.
- **AC21.** LLM routes user response to non-head queued tool: queue reorders; new head is presented; original head moves to position 1.
- **AC22.** On `deliver(instance_id, response)`: matching `WorkflowInstance.pending_conversation_tool` is cleared; SOP demotes from P0.
- **AC23.** Queue is per-session (not global) — verified by spinning up 2 sessions in the same server, each with concurrent multi-SOP queues; no cross-session contamination.

---

## §8. Risk register

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | LLM routes response to wrong SOP (e.g., "yes" with 2 P0 SOPs both expecting confirmation) | MED | MED | Scenario B AC; LLM trained to disambiguate; user can override via `/route <instance_id>` CLI |
| R2 | Token-budget blowout with many active SOPs | MED | LOW | Compact rendering for P2/P3; warning emitted; user can `suspend_workflow` to reduce load |
| R3 | LLM hallucinates invalid `instance_id` in `workflow_route` | LOW | LOW | AC8 validation; fallback to `"none"` + log |
| R4 | Mid-session `set_focus_mode("single")` loses focus on wrong SOP | LOW | MED | Priority tiebreaker is deterministic (P0 → P1 → most recent activity → creation_ts); mode-switch notification names the winner explicitly so user can correct |
| R5 | Queue head changes silently (e.g., user delays response while LLM reorders) | LOW | MED | Queue reorder is explicit (only on `deliver()` with non-head ID); never spontaneous |
| R6 | `pending_conversation_tool` not cleared on tool failure → SOP stays P0 forever | LOW | HIGH | All `aget_input()` paths wrap in try/finally; on exception, clear `pending_conversation_tool` + log |
| R7 | Future event-loop semantics — `await f` on the non-head's future blocks the WorkGraph node indefinitely if user never answers | MED | MED | Configurable timeout (default 24h); on timeout, raise `PendingToolTimeout` and let the SOP handle it (e.g., suspend itself) |
| R8 | `async_background` tools fired by an SOP — when SOP is suspended/exited, what happens to in-flight tasks? | MED | MED | On suspend: tasks continue, results stored in `WorkflowInstance.async_results: dict`; on next resume, results replay from store. On exit: tasks cancelled; pending results discarded with warning. |
| R9 | UI doesn't render all P0 tools in "AWAITING" — user only sees the head, doesn't know others are queued | MED | LOW | Multi-display in the prompt informs the LLM; LLM can mention queue depth in its response to the user |
| R10 | Backward-compat derivation of `asynchronous → concurrency` doesn't cover edge cases (e.g., tool.json with both `asynchronous` AND `concurrency` set) | LOW | LOW | Validation: if both present, prefer `concurrency` and emit warning; documented in §3.1 |
| R11 | The routing-step prompt instruction is ignored by some LLMs (especially weaker models) | MED | MED | Routing instruction is in the first-position prompt section (high attention); enforced by parser — if LLM emits other tool calls before `workflow_route`, parser rejects; LLM corrected with structured error |
| R12 | `is_computing: bool` becomes stale (an `async_background` tool completes but flag isn't cleared) | LOW | LOW | Completion callback always clears `is_computing` in try/finally |

---

## §9. Open questions

| # | Question | Why it matters | Recommendation |
|---|---|---|---|
| Q1 | Should `single` mode reject `enter_workflow` if user has an active SOP, OR auto-suspend? | UX consistency | Auto-suspend (per §2.2); reject would be hostile to "I want to switch focus" |
| Q2 | Should suspended SOPs render in `single` mode prompt at all? | Prompt size vs discoverability | YES, one-line P3 list; user needs to see resumable SOPs |
| Q3 | What happens if user `resume_workflow(id)` for a SOP that completed long ago (in `completed_instances`)? | API surface clarity | Error: "Cannot resume completed workflow; use `enter_workflow` to start a new instance" |
| Q4 | Should `set_focus_mode` be available via slash command (`/focus single`) in addition to LLM tool + UI button? | Power-user CLI | YES, low cost (one parser case) |
| Q5 | When a P1 SOP's `last_user_input_advanced_at` becomes stale (e.g., user has done 3 turns of chitchat since), does P1 demote to P2? | Priority semantics | YES — P1 is "most recently advanced"; chitchat doesn't advance any SOP → next-most-recently-advanced becomes P1 |
| Q6 | If a user starts an SOP and immediately switches to multi mode, does the original SOP get P1 status by default? | Prevent confusing demotion to P2 | YES — entry counts as "advancement"; P1 until a different SOP becomes more recent |
| Q7 | Should the `workflow_route` envelope support `"ids": ["all"]`? | Some commands legitimately apply to all SOPs (e.g., "suspend all") | NO — use `enter_workflow` etc. directly; "all" is a footgun |
| Q8 | When two SOPs are P0 with pending tools, AND user says "yes", AND LLM disambiguates by asking user — does the queue head change while disambiguating? | UI complexity | NO — queue head only changes on `deliver()` or explicit `reorder_to_head()`; disambiguation is a chitchat turn |
| Q9 | Per-deployment `sop_config.json` — where does it live? | Configuration source consistency | Co-located with the SOP registry: `resources/sops/sop_config.json` (AgentFoundation default) + `resources/sops/sop_config.json` (OpenStartup override) |
| Q10 | Compound widget across SOPs (deferred per §0) — when to revisit? | Future direction | When the queue depth in production exceeds N=3 with regularity AND user feedback says "I want to answer multiple at once" |

---

## §10. Phased rollout

| Phase | Scope | Risk | LOC est | Days |
|---|---|---|---|---|
| 1 | `concurrency` field on `ToolDefinition` + backward-compat derivation + tests | LOW | ~80 + ~120 tests | 1 |
| 2 | `WorkflowInstance.pending_conversation_tool` + `priority` + `is_computing` + `last_user_input_advanced_at` fields + tests | LOW | ~60 + ~80 tests | 1 |
| 3 | `PendingConversationToolQueue` class + per-session lifecycle + AC20-AC23 tests | MED | ~150 + ~200 tests | 2 |
| 4 | `WorkflowManager.focus_mode` attribute + persistence + AC1/AC10-12 tests | LOW | ~70 + ~100 tests | 1 |
| 5 | `single`-mode auto-suspend + `multi`-mode entry + AC6 + Scenario E validation | MED | ~80 + ~100 tests | 1 |
| 6 | Multi-mode prompt template (`active_sop_routing.jinja2`) + priority renderer + token budget + AC2-AC15 tests | MED | ~200 + ~250 tests | 2 |
| 7 | `workflow_route` envelope parser + dispatch + Scenarios A/B/D/E tests | MED | ~120 + ~150 tests | 1.5 |
| 8 | `set_focus_mode` tool + slash command + UI button + AC10-11 e2e | LOW | ~100 + ~80 tests | 1 |
| 9 | E2E verification with 3-SOP test session (role_creation + code_optimization + model_optimization) | HIGH | tests only ~300 | 1 |

**Total:** ~860 LOC source + ~1380 tests; ~11 days.

**Feature flag:** `OPENTEAM_USE_MULTI_FOCUS_SOP` — default `false` initially (single-focus only) until Phase 8 lands; then default `true` (matches §2.1 multi-default).

**Phased landing:** Phases 1-3 can land independently and are useful even without multi-focus (concurrency labels improve docs; queue helps even in single-focus future use). Phases 4-9 are bundled by feature flag.

---

## §11. File inventory

### New files

| File | Purpose |
|---|---|
| `AgentFoundation/src/agent_foundation/common/workflow/pending_conversation_tool_queue.py` | Per-session queue (§4) |
| `AgentFoundation/src/agent_foundation/resources/prompt_templates/conversation/main/_variables/active_sop_routing.jinja2` | Multi-mode routing prompt block (§2.6) |
| `AgentFoundation/src/agent_foundation/resources/sops/sop_config.json` | Default `focus_mode`, `total_budget`, etc. |
| `OpenStartup/src/openteam/server/resources/sops/sop_config.json` | Per-deployment overrides |

### Modified files

| File | Change |
|---|---|
| `AgentFoundation/src/agent_foundation/resources/tools/models.py` | Add `concurrency: Literal[...]` field + `asynchronous` backward-compat property + `from_dict` migration (Phase 1) |
| `AgentFoundation/src/agent_foundation/common/workflow/manager.py` | Add `focus_mode`, `pending_tool_queue`, `set_focus_mode()` (Phase 4) |
| `AgentFoundation/src/agent_foundation/common/workflow/instance.py` | Add `pending_conversation_tool`, `is_computing`, `last_user_input_advanced_at`, `priority` property (Phase 2) |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | `_handle_conversation_tools` integration with queue (Phase 3); multi-mode rendering (Phase 6) |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/tool_call_parser.py` | Recognize `{"type":"sop","action":...}` and `{"type":"workflow_route","ids":...}` (Phase 7) |
| `AgentFoundation/src/agent_foundation/resources/prompt_templates/conversation/main/initial.jinja2` | Include `active_sop_routing.jinja2` conditionally (Phase 6) |
| `OpenStartup/src/openteam/server/services/conversation_service.py` | Populate `focus_mode`, `active_sops_by_priority`, `pending_tools_queue_head/all` in session_context (Phase 6) |
| `OpenStartup/src/openteam/server/services/session_store.py` | Persist `focus_mode` in `session_metadata.json` (Phase 4) |
| `AgentFoundation/src/agent_foundation/ui/conversation_panel.py` (or similar) | UI button for `set_focus_mode` (Phase 8) |
| All conversation tool `tool.json` files | (Optional) Add explicit `"concurrency": "blocking"` for clarity (Phase 1; defaults work without it) |
| All async action tool `tool.json` files | Migrate `"asynchronous": true` → `"concurrency": "async_background"` (Phase 1; backward-compat handles unmigrated files) |

### Deleted files

None — this plan is strictly additive.

---

## §12. Honest comparison to alternatives

| Alternative | Why we chose this design |
|---|---|
| **Single mode only (no multi)** | Doesn't match user's stated need; users will hit ambiguity the moment they try to run 2 SOPs |
| **Multi mode only (no single)** | Removes a useful constraint for single-SOP-focused workflows (some users want strict serialization); cost of supporting both is trivial |
| **Always-routed multi mode (routing block even with 1 active SOP)** | Wasteful — most sessions have 0 or 1 active SOP; routing block costs ~300 tokens per turn for no benefit |
| **Routing via separate LLM call** | Extra latency; the existing in-prompt routing is one message, one inference, zero extra round-trips |
| **Compound widget across SOPs** | Cross-SOP synchronization is hard (different WorkGraph asyncio tasks); UI complexity high; queue + single-head solves 95% of cases |
| **Auto-suspend even in multi mode (e.g., suspend after 10 min idle)** | Surprising state change; users should explicitly suspend; idle timeout can be added later if needed |
| **Concurrency = `bool` (just `is_blocking`)** | Loses the `async_awaitable` future direction; rework would be expensive |
| **Concurrency = 5+ values (e.g., `streaming`, `cancellable`, etc.)** | Over-engineering; 3 values cover all observed and near-future use cases |

---

## §13. Open issues to track separately

1. **Queue inspection UI** — operators may want to see "what's queued" in a debug panel. Not v1.
2. **Queue analytics** — average queue depth, max queue depth per session, etc. Useful for capacity planning but not blocking.
3. **`async_awaitable` reference implementation** — no v1 tool uses it; add when a real use case appears (e.g., parallel deep-research subtasks within a single agentic loop iteration).
4. **Cross-session queue (multi-tenant agents)** — explicitly out of scope; each session has its own queue.
5. **Resume-after-server-restart** — `pending_conversation_tool` survives via persistence; queue is in-memory only. On restart, queue is empty; SOPs with `pending_conversation_tool != None` are in a "stuck" state requiring user action. Need explicit recovery: on startup, scan suspended/active SOPs, re-create queue entries OR notify user "session N had pending tools; please re-engage".

---

## §14. Summary

**One sentence:** Multi-focus mode is the default (free when ≤1 SOP active; cheap when ≥2 via priority budget); a per-session `PendingConversationToolQueue` resolves blocking-tool contention via single-head + multi-display; `ToolDefinition.concurrency` formalizes 3 concurrency levels (blocking / async_background / async_awaitable) with backward-compat for today's `asynchronous: bool`.

**Bottom line:** This plan absorbs the empirical critique from the prior round (`exit_workflow` vs `suspend_workflow` distinction, event-driven P1 not wall-clock, deterministic budget algorithm, explicit invalid-ID handling, shared-InteractiveBase race resolution via queue) and adds the empirical concurrency-labeling finding (today's `asynchronous: bool` is too coarse). It's strictly additive to `sop_runtime_enablement_plan.md` and can be implemented in 9 phases over ~11 days; phases 1-3 can land independently as foundation work.
