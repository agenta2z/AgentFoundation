# Conversation Tool Groups, Assistant Rounds, and Turn Artifacts

## Status

Integrated Codex plan, updated after comparing:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/conversation_tool_enhancement/parallel_tool_groups_and_per_round_message_boxes_plan.md`
- `/Users/tchen7/.claude/plans/update-your-task-tool-adaptive-goose.md`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/conversation_tool_enhancement/codex/parallel_groups_assistant_rounds_turn_artifacts_plan.md`

Best plan if only one of the three current files can be used: this integrated Codex plan, `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/conversation_tool_enhancement/codex/parallel_groups_assistant_rounds_turn_artifacts_plan.md`, after the integration edits in this file. The latest `.claude` plan correctly keeps the existing direct-map compound payload and contributes useful execution detail, but it still conflates sequential conversation-tool groups with assistant rounds in a few places. The longer peer plan has useful verification detail, but references stale/non-active websocket files and an unsafe compound payload shape in several places. This integrated plan keeps the valid turn/round model, the `.claude` plan's direct-map widget insight, the peer plan's test rigor, and source-correct active file targets.

## Summary

Human testing exposed three related issues:

1. Multiple conversation tools in one assistant response are bundled by the backend, but the shared UI renders the compound payload as a sequential stepper (`Step 1 of N`) instead of a tabbed multi-input widget.
2. One user message can produce multiple assistant LLM rounds inside the agentic loop; the UI has one global streaming slot, so later rounds can replace earlier visible assistant output.
3. Runtime `turn_NNN` directories currently conflate a user request, an assistant LLM round, and prompt-view artifacts. In the tested session, the user input landed in `turn_001`, while both RovoDev streams landed under `turn_002`.

The fix is to make the model explicit:

- **User turn** = one human message and the complete agentic response to it. It owns one top-level `turn_NNN/` directory.
- **Assistant round** = one LLM call inside that user turn. It owns one canonical `assistant_round_id`, one UI assistant bubble, and one nested `round_MMM/` artifact/cache directory. `message_id` and `stream_id` are derived from or equal to `assistant_round_id` for transport compatibility.
- **Conversation tool group** = one interactive collection point inside an assistant round. Independent tools in the same group render as tabs and submit together.

## Source-Verified Root Causes

- `ConversationToolWidget` directly intercepts `metadata.compound` and renders local `CompoundWidget`, whose state is `step` and whose UI says `Step {step + 1} of {tools.length}`. The existing `MultiInputWidget` renders tabs, but it scalarizes child responses and can drop composite `{ choice_index, inputs }`; it is not safe as a drop-in replacement.
- The current compound widget already preserves each child widget's raw response under an output-variable keyed direct map and submits `JSON.stringify(responses)`. That direct map is the low-churn contract to preserve. A `{ values: ... }` envelope remains a supported compatibility shape, but forcing the UI to switch envelopes is unnecessary churn.
- `useManagerChat.js` has a singleton `streamingContentRef` and `streamingMessage`; `message_start` clears that singleton. It cannot represent two assistant rounds unless the server provides stable per-round identity and the UI keys streams by it.
- The active websocket/server path is OpenStartup's `manager_websocket_routes.py`, `websocket_interactive.py`, and `conversation_service.py`. Plans referencing `agent_service_bridge.py` or `agent_websocket_routes.py` are stale for this UI.
- `conversation_service.py` currently sets first cache folder to `turn_{initial_turn + 1}`, then `_on_new_turn` can rotate to `turn_{new_turn + 1}`. That splits the user's turn root from its response streams.
- `session_routes.py` and the UI prompt viewer currently fetch prompt data by only `turn_number`. With nested assistant rounds, that becomes ambiguous unless the REST path accepts round identity or assistant-round identity.
- Submitted widget responses are currently inserted locally by the UI, while the websocket server only forwards the submitted payload into the active input queue. A reload can therefore lose the widget-response card unless the server persists it.
- The conversation prompt currently describes multiple tools as an execution-ordered list. With `parallel_group`, prompt guidance must distinguish independent same-group conversation questions from later action tools that may reference collected outputs.
- `session_store.create_session()` updates the sessions index, but `append_message()` and `update_session()` currently persist without refreshing the index, so sidebar counts can stay stale after messages are appended.

## Implementation Plan

### 1. Add Explicit Assistant Round Lifecycle

Add optional round lifecycle hooks to `ConversationalInferencer.run_agentic_loop` without importing OpenStartup concepts into AgentFoundation:

- `on_round_start(iteration, current_input, turn_number) -> RoundContext | None`
- `on_round_complete(round_context, raw_response, clean_response, prompt_data, parsed_response_metadata) -> None`

The callback return from `on_round_start` should be a small typed object or typed mapping with a documented `cache_folder` field. AgentFoundation should treat all OpenStartup-specific ids as opaque, but it may read `cache_folder` before prompt rendering and backend invocation. If the active interactive has `set_round_context(...)`, pass the context to it so websocket events can carry the same ids.

For completion, add a dedicated LLM-response-complete hook immediately after the raw response has been cleaned, prompt data is available, and the response has been parsed, but before `_handle_conversation_tools` sends `pending_input` and awaits the user's widget response. The existing `_fire_turn_complete` callback is too late for widget-pending rounds because it fires after conversation input handling returns; it can remain as a compatibility callback, but it must not be the source of round persistence. The important invariant is that round completion fires once for every LLM call, including action-tool continuation rounds that do not pass through widget handoff and do not change `turn_number`.

OpenStartup's `ConversationService.run_conversation_turn` owns the concrete context:

- `user_turn_number`
- `parent_user_message_id`
- `round_index`
- `assistant_round_id`
- `message_id` derived from `assistant_round_id`
- `stream_id` derived from `assistant_round_id`
- `turn_dir`
- `round_dir`
- `cache_folder`

This avoids guessing round boundaries from `pending_input` and handles every LLM call, including action-tool continuations that do not pass through widget handoff.

Do not keep using the existing `on_new_turn` callback as the top-level turn allocation seam. It currently fires before the first LLM render and again around widget handoffs, which is exactly how one user request can be split across multiple sibling `turn_NNN/` directories. After this change, OpenStartup should allocate the user turn once before `run_agentic_loop`; `on_round_start` allocates assistant round artifacts; and any legacy `on_new_turn` callback should either be removed from this path or reduced to compatibility-only metadata that does not create directories, rotate `cache_folder`, or increment the persisted user turn.

### 2. Fix User Turn and Artifact Semantics

In OpenStartup:

- Allocate one top-level `turn_NNN/` per inbound user message in `run_conversation_turn`.
- Save `user_input.txt` and root `turn.json` at the turn root.
- Put each LLM call's prompt/response/cache artifacts under `turn_NNN/round_MMM/`.
- Remove the `turn_{new_turn + 1}` cache-folder rotation. Internal loop boundaries may record widget/user input, but must not create sibling top-level turns.
- Keep `/api/sessions/{id}/turns/{turn_number}` compatible by returning root turn metadata plus the latest/default round prompt data, and add enough metadata for the UI to open the exact round associated with an assistant bubble.
- Extend exact prompt lookup with `round_index` and/or `assistant_round_id`. Concretely, allow `/api/sessions/{id}/turns/{turn_number}?round_index=N` or `?assistant_round_id=...`, update `DataService.get_turn_data` / `SessionStore.get_turn_data` to resolve `turn_NNN/round_MMM/turn.json`, and update `fetchTurnData` / `handleViewPrompt` to pass the message's round metadata. Plain `/turns/{turn_number}` remains a compatibility fallback returning root turn data plus the latest/default round.
- Update both `DataService.save_turn_data/get_turn_data` and `SessionStore.save_turn_data/get_turn_data` for round-aware persistence. The wrapper and store signatures must move together, otherwise REST can only reach flat turn artifacts.
- Persist `turn_number`, `round_index`, `assistant_round_id`, and `message_id` on each assistant message. On session reload, `useManagerChat` should preserve these persisted fields instead of recomputing a unique turn number per assistant bubble when they are already present.
- For new sessions, do not count the welcome assistant message as `turn_001`. User-turn numbering starts at the first real manager/user message; old sessions without persisted turn metadata may continue using compatibility fallback counting.
- Persist submitted widget responses server-side in the `pending_input_response` handler, keyed to the active `assistant_round_id` / `message_id`, so reload matches the live UI timeline.
- Update `session_store.append_message()` and `update_session()` to refresh the sessions index after persistence.

Acceptance for the motivating case:

- One human request creates exactly one new top-level `turn_NNN/`.
- That directory contains `round_001/` for the SOP-entry LLM round and `round_002/` for the Phase 0a question round.
- `session_state.json` persists both assistant round messages, not only the final or pending widget state.

### 3. Emit and Render Per-Round Assistant Message Boxes

In `WebSocketInteractive`:

- Add round context state set by `set_round_context(...)`.
- Reset `_clean_output` when a new round starts, and bind/clear round context per round so a correction from one stream cannot leak into the next assistant bubble.
- Emit `message_start` from the round-start path when round context is installed, not once at the route level for the whole user request. `stream_token_batches` should then use the already-current round context for tokens. This covers first-round, streaming, and non-streaming paths uniformly.
- Treat `assistant_round_id` as the canonical identity. For compatibility, set `message_id` and `stream_id` to that id unless there is a concrete need for separate derived suffixes.
- Include `message_id`, `assistant_round_id`, `stream_id`, `turn_number`, and `round_index` on:
  - `message_start`
  - `token`
  - `stream_correction`
  - `pending_input`
  - `message_end`
- Keep `turn_boundary` compatibility-only; it is not the authoritative round lifecycle event.
- Emit/record `message_end` once for every assistant round after the clean final text is known. For rounds that need user input, `message_end` precedes `pending_input`, and `pending_input` references the same `message_id`. The UI dedupes by `message_id` so a round ending in a widget does not create two assistant bubbles.

In `manager_websocket_routes.py`:

- Keep route-level `message_start/message_end` only for mock/non-agentic fallback.
- For agentic `run_conversation_turn`, do not append a duplicate final assistant message after the loop if per-round persistence already appended the visible assistant rounds.
- In `pending_input_response`, append a persisted `widget_response` message with the submitted payload and the linked `assistant_round_id` / `message_id` before releasing the response into the input queue. The server should assign or echo a stable `widget_response_id` / `pending_input_id`; the optimistic UI card and persisted history card use that same id for dedupe.
- Ensure prompt data snapshots are taken after the relevant assistant message/round data is known, not from a stale pre-append session snapshot.
- When saving prompt/turn metadata, build `api_payload.messages` from a fresh post-append session read so the logged payload includes the assistant round just committed.

In `useManagerChat.js`:

- Track active streams by `stream_id`/`message_id`, not one singleton buffer.
- `message_start` opens a new assistant stream; if another stream is active, commit it first.
- `token` and `stream_correction` update only the matching stream.
- `message_end` commits the matching stream as an assistant bubble and dedupes by `message_id`.
- `pending_input` commits the matching assistant stream if it has not already been committed, then shows the widget linked to that assistant message.
- The local `widget_response` optimistic card must use the same id/round metadata shape the server persists, and `session_init` must preserve `role: "widget_response"` instead of mapping every non-manager role to `agent`. Persisted widget responses must render without duplicating the optimistic one after reconnect.
- `fetchTurnData` and `handleViewPrompt` must pass `round_index` or `assistant_round_id` when available, falling back to turn-only lookup for old sessions.
- Reloaded history should match live display.

### 4. Add Conversation Tool `parallel_group`

In AgentFoundation:

- Add `parallel_group: int | None` to `ConversationTool`.
- Parse optional top-level `parallel_group` from each `ToolsToInvoke` conversation object.
- Preserve it in `to_dict()` / `from_dict()`.
- Keep `parallel_group` separate from existing `group_id` / `on_group_resolve` metadata. Those rich-widget group fields have different semantics and must not be reused for parallel collection.
- Add grouping helper semantics:
  - same explicit `parallel_group` means collect in one group;
  - different explicit groups execute in first-appearance order;
  - if no tools declare `parallel_group`, all consecutive conversation tools in the assistant response form one implicit group;
  - tools without a group inside a mixed explicit response coalesce with adjacent ungrouped tools as their own ordered implicit group, not into an unrelated explicit group.
- Apply the same grouping in interactive and yolo paths.
- Sequential groups are not assistant rounds. They are multiple interactive collection points inside one assistant round because they come from one LLM response. If separate pending widgets are displayed sequentially, give each group a stable `tool_group_index` and `widget_id` linked to the parent `assistant_round_id`; do not increment `round_index` unless another LLM call occurs. UI state, persisted widget responses, and any prompt/debug metadata for the group should use `{assistant_round_id, tool_group_index, widget_id}` rather than fabricating an LLM round.
- Merge collected outputs across sequential groups before resolving action-tool `__var__` placeholders.
- Validate multi-tool groups before rendering: each grouped tool must have a unique primary output key, or the group must be split/error-reported with a clear diagnostic. This prevents duplicate output vars, missing outputs, or same-type no-output tools from overwriting each other in the direct-map payload.
- Confirmation tools and other side-effecting conversation tools must either stay out of implicit multi-tool groups or share the same decode/finalize path as single-tool handling, including parameter overrides, variable overrides, and SOP input-gate behavior.

Prompt guidance:

- Document `parallel_group` in `conversation/main/initial.jinja2`.
- Use the same `parallel_group` only for independent conversation inputs that can be answered in any order.
- Use different groups only when the UI should ask one group after another.
- Do not model dependencies between conversation tools inside one group. If an input depends on a previous answer, use a later group or a later assistant round.
- Action tools may reference collected conversation outputs after all needed conversation groups complete.
- Use a later assistant round, not just a later group, when the next question semantically depends on the previous answer.

### 5. Replace Compound Stepper With Tabbed Compound UI

In shared React UI:

- Update `ConversationToolWidget`'s compound path to render tabs for multi-tool groups.
- Keep the existing raw-response contract: `responses[output_var] = rawChildResponse`.
- Submit the combined payload as the existing output-variable keyed direct map: `JSON.stringify({ [output_var]: rawChildResponse })`. The backend also tolerates `{ values: ... }` and `user_input.values`, but the direct map is current behavior and should remain the canonical low-churn UI contract. If an output variable is literally `values` or `user_input`, fall back to the `{ values: ... }` envelope to avoid unwrap collisions.
- Keep child state while switching tabs. Because leaf widgets currently keep unsubmitted draft state locally, implement tabs with mounted panels or lifted draft state; do not unmount inactive tabs and lose drafts.
- Enable submit only when required tabs/fields are complete.
- Show clear completion markers per tab.
- Do not use existing `MultiInputWidget` unless it is first refactored to preserve raw child payloads, dispatch child widgets using `metadata.widget_type`, count completion only for real tool output keys rather than control keys such as `variable_override`, and preserve child state across tab switches. Current scalar extraction is unsafe for composite single-choice nested inputs.

## Tests and Verification

### AgentFoundation Python

- `ConversationTool` parse/round-trip tests for absent, same, and different `parallel_group`.
- Grouping helper tests for implicit all-tools group, explicit ordered groups, and mixed explicit/implicit tools.
- `_handle_conversation_tools` tests proving tabbed compound payloads preserve `output_var` mapping and composite nested bindings.
- Compound decoder tests for the direct output-var map shape plus existing compatibility shapes, including the `values`/`user_input` collision fallback.
- Yolo grouping test proving multi-tool synthetic responses bind by output variables and nested input names, and that required nested inputs for synthetic single-choice responses are not silently dropped.
- Confirmation/side-effect grouping test proving grouped handling preserves parameter overrides, variable overrides, and SOP gate behavior, or rejects unsupported group composition clearly.
- Round lifecycle callback unit test proving `on_round_start` is called before each LLM call and `on_round_complete` after each clean response.
- Round lifecycle regression proving action-tool continuation produces a new round context even when `turn_number` does not change.
- Widget-pending round regression proving `on_round_complete` fires before `_handle_conversation_tools` waits for user input, so the assistant preamble can be persisted and ended before `pending_input`.
- Regression test proving a simple no-widget user message creates exactly one top-level `turn_NNN/`, not one directory before render plus another for the response.

### AgentFoundation React Shared UI

- Compound widget renders tabs instead of `Step 1 of N`.
- Switching tabs preserves typed values.
- Submit-all emits output-variable keyed raw child payloads.
- Composite single-choice child emits and preserves `{ choice_index, inputs }`.
- Legacy one-tool conversation widgets still render unchanged.

### OpenStartup Server/UI

- One user message with two LLM rounds creates one `turn_NNN/` with `round_001/` and `round_002/`.
- The same scenario appends two assistant messages to `session_state.json`, each with `assistant_round_id`, `message_id`, `turn_number`, and `round_index`.
- `sessions_index.json` message counts update after both `append_message` and `update_session`.
- WebSocket event simulation proves two `message_start/message_end` pairs create two assistant bubbles and do not double-commit when `pending_input` arrives.
- `stream_correction` updates only the matching `stream_id`.
- Prompt viewer resolves the correct turn/round artifact for each assistant bubble.
- REST prompt lookup supports `round_index` and/or `assistant_round_id`, and falls back to the old turn-only shape for old sessions.
- Submitted widget responses are persisted, reloaded, and rendered exactly once.
- Welcome-message exclusion test proving the initial assistant greeting does not consume `turn_001`.
- DataService wrapper tests proving round-aware `save_turn_data/get_turn_data` reach the same artifacts as SessionStore.
- Sequential group test proving multiple tool groups inside one LLM response get distinct widget ids but do not increment `round_index`.

### Manual Smoke

Restart server and UI, hard refresh, then send:

`help improve the model for responsible ai`

Expected:

- SOP-entry response appears as one Orchestrator bubble.
- Phase 0a setup response appears as a second Orchestrator bubble.
- Target path and modeling-artifact questions appear as tabs in one widget.
- Filling tabs in either order submits both values correctly.
- Disk has one new top-level user turn with nested round directories.
- Reloading the page preserves the same visible message timeline.

## Risks and Guardrails

- The turn-directory layout change affects prompt viewer, session reload, and any code scanning `turn_NNN` directories. Audit readers before changing writers.
- Do not let `pending_input` and `message_end` both append the same assistant bubble. Deduplication must be keyed by `message_id`.
- Do not collapse multiple assistant rounds into one final assistant message; that hides real agent actions and caused the original UI confusion.
- Do not treat `parallel_group` as a dependency system. It is presentation/collection grouping only; semantic dependencies belong in later assistant rounds.
- Keep route-level mock streaming compatible so simple mock sessions do not need the full round lifecycle.

## Out of Scope

- True concurrent independent submission of each tab.
- Refactoring the older handler-registry vs inline conversation-tool rendering split.
- Renaming every public `turn` API. Compatibility paths remain, with clarified semantics and round metadata.
