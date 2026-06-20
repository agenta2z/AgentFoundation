# Conversation Tool Enhancement Plan: Typed Path Inputs and Composite Choices

Status: canonical integrated Codex plan

Last updated: 2026-06-18

Compared inputs:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/conversation_tool_enhancement/codex/conditional_path_inputs_plan.md`
- `/Users/tchen7/.claude/plans/update-your-task-tool-adaptive-goose.md`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/conversation_tool_enhancement/typed_input_and_composite_choice_plan.md`

## 0. Verdict

Use this file as the canonical implementation plan.

The strongest base is the Codex architecture: typed input specs, per-choice embedded input, host-provided path completion, live-runtime updates, and separate variables for "selected mode" and "entered value".

The newer Claude plan contributes important source-verified corrections:

- AgentFoundation already has a `/path-complete` endpoint; do not invent `/api/paths`.
- OpenStartup does not mount a workspace path-complete route; add one.
- `clarification/tool.json` has a path-hostile yolo default: `"Follow your best judgment."`.
- Compound conversation-tool collection currently uses `str(raw_value)` and can turn lists into Python repr strings.
- The current runtime is string-oriented at SOP/Jinja/action boundaries, so raw `list[str]` should not be published blindly as a prompt variable.

The typed-input plan contributes a useful test/checklist style, but it is not safe as-is because it still contains stale and contradictory guidance around `InputMode.PATH`, registry routing, `/api/paths`, and handler-only implementation.

If forced to pick one of the original plans with zero edits, choose the Codex plan for architecture. If choosing an implementation checklist only, borrow the typed-input plan's test discipline. If choosing a bug-audit memo, borrow the Claude plan.

## 1. Current Source Facts

### F1. The live runtime does not use the handler registry as the source of truth

The active OpenStartup path reaches:

- `ConversationalInferencer._handle_conversation_tool(...)`
- `ConversationalInferencer._handle_conversation_tools(...)`
- `_build_input_mode(tool)`

The handler registry exists, but the live path still calls `_build_input_mode(tool)` directly and decodes responses inline.

Implementation consequence:

- Do not update only `handlers/clarification.py` or `handlers/single_choice.py`.
- Add shared helpers, then call them from the live inline path and from handlers.

### F2. Current parser canonicalization is incomplete

Current parser only aliases:

- `multiple_choices` -> `multiple_choice`
- `single_choices` -> `single_choice`

It does not normalize:

- `single-choice`
- `single choice`
- `multiple-choice`
- `multiple choice`
- `expected-input-type`
- `allow-multiple-input`
- nested `choices[].input`
- `output: "workflow_target_path"` into `["workflow_target_path"]`

Implementation consequence:

- Canonicalize before constructing `ConversationTool`.
- Do not rely primarily on `ConversationToolType._missing_`; today `tool_type` remains a plain string and later misses equality checks.

### F3. `output: "x"` is a real bug

Both parser and `ConversationTool.from_dict` can preserve `output` as a raw string. Later code uses `output_vars[0]`, which becomes the first character of the string.

Implementation consequence:

- Normalize `output` and `output_vars` to `list[str]` at parse/model boundaries.

### F4. `ConversationTool.to_dict()` omissions are real but not the live blocker

`ConversationTool.to_dict()` does not currently serialize fields such as `expected_input_type`, `prefix`, or `output_vars`.

That should be fixed for round-trip correctness and tests, but it is not the main reason path metadata fails in live OpenStartup. The live handler path reads the live object.

Implementation consequence:

- Fix `to_dict()` as part of schema hardening.
- Do not frame it as the sole or primary live runtime bug.

### F5. Composite choices cannot survive the current model/UI path

Today:

- `ChoiceItem` only carries `label`, `value`, `description`.
- `ChoiceItem.from_dict()` drops a nested `input` block.
- `ChoiceOption` has no `input` field.
- `_build_input_mode()` drops descriptions for normal `single_choice` and `multiple_choice`.

Implementation consequence:

- Add `InputFieldSpec`.
- Add `ChoiceItem.input`.
- Add `ChoiceOption.input`.
- Preserve descriptions and input specs through `_build_input_mode()`.

### F6. `InputMode.PATH` is unnecessary churn for this phase

Current shared-ui dispatch uses:

```javascript
metadata.widget_type || mode
```

Current built-ins register `free_text -> TextInputWidget`, `single_choice -> SingleChoiceWidget`, etc.

Implementation consequence:

- Do not add `InputMode.PATH` as the canonical path.
- Use `InputMode.FREE_TEXT` with typed fields plus `metadata.widget_type = "path_input"` for direct path prompts.
- Keep metadata compatibility during migration.

### F7. AgentFoundation has path completion, but containment must be hardened

AgentFoundation already has:

```text
GET /path-complete
```

in:

`/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/webui/backend/routes/workspace_routes.py`

OpenStartup does not mount an equivalent route.

The existing containment check uses string `startswith`, which is prefix-spoofable.

Implementation consequence:

- Factor a reusable path-completion helper.
- Replace string `startswith` containment with `Path.resolve().relative_to(...)`.
- Mount OpenStartup route at `/api/workspace/path-complete`.
- Validate submitted paths in backend response finalization too, not only in autocomplete.

### F8. Multi-value handling needs an explicit publish format

`set_session_variables()` accepts `Any`, but current SOP/Jinja/action-tool plumbing is string-oriented:

- Current compound collection uses `str(raw_value)`.
- Proposal selection deliberately stores comma-joined IDs for CLI usage.
- `{{ var }}` rendering of a Python list is not a stable user-facing contract.

Implementation consequence:

- Keep typed values during UI payload handling and backend decode/finalization.
- Publish output variables through an explicit serializer.
- For Phase 0a multi-path output, default to a reversible JSON array string unless the input spec explicitly requests a different format such as comma.
- Preserve existing comma behavior for `proposal_selection` IDs.
- Never use raw `str(list)`.

This resolves the false binary between "store raw lists everywhere" and "make comma strings canonical." The canonical backend value is typed until the publication boundary; the current published SOP variable is a string with a declared serialization.

### F9. Yolo defaults must be expected-input-type aware

`clarification/tool.json` currently uses:

```json
{
  "mode": "fixed",
  "value": "Follow your best judgment."
}
```

For path clarification, that binds prose as a filesystem path.

Implementation consequence:

- Add expected-input-type-aware yolo default resolution.
- For path input, use a concrete valid path such as resolved `session_root_path` only when acceptable.
- Otherwise gate the input instead of fabricating prose.
- Yolo synthesis must bind through `output_vars` and the same finalizer as normal responses.

### F10. Tool prefixes are not rendered after parsing

The SOP can tell the LLM to emit:

```json
{
  "prefix": "{{ session_root_path }}"
}
```

SOP guidance and template feed values are rendered before the LLM call, but a conversation tool argument emitted by the LLM is parsed after that render. The live `_build_input_mode()` path currently forwards `tool.prefix` directly to the UI. The compound build also forwards `tool.prefix` directly in `tool_configs[]`.

Implementation consequence:

- Do not assume `{{ session_root_path }}` has already become a real path in `tool.prefix`.
- Runtime normalization must render templated string fields before building input modes and before finalization.
- Apply the same rendering to nested `choices[].input.prefix`.
- Rendering belongs in the conversational runtime/helper layer, not in the parser, because the parser does not own session context or `prompt_renderer.render_string`.

### F11. Composite choices require distinct multi-binding, not output-var aliasing

Current binding sites mostly treat `output_vars` as aliases for one value. That is fine for legacy widgets, but wrong for composite choices.

Composite single-choice has two sources:

- selected choice value -> tool-level `output_vars`, for example `workflow_modeling_artifacts_mode = "manual_paths"`
- selected choice nested input -> `ChoiceItem.input.name`, for example `workflow_modeling_artifacts_path = <entered paths>`

Implementation consequence:

- Response decode must return a distinct `bindings: dict[str, value]`.
- Binding sites must stop assuming every `output_var` gets the same value.
- The nested input variable name is not necessarily listed in `tool.output_vars`; it lives on the selected choice's `InputFieldSpec.name`.

### F12. Compound responses nest child payloads under each tool output variable

The current shared-ui compound widget stores each child response under `currentTool.output_var` and submits the combined object. For a composite choice, the server should expect:

```json
{
  "workflow_modeling_artifacts_mode": {
    "choice_index": 1,
    "inputs": {
      "workflow_modeling_artifacts_path": ["data/features", "experiments/run_42"]
    }
  }
}
```

not a top-level `inputs` object.

Implementation consequence:

- The compound decoder must first fetch the child payload by the tool's primary output key.
- It must then pass that child payload to the same single-tool decode helper.
- It must merge the helper's returned `bindings` into the compound result and into session variables.
- It must not stringify the whole child payload as the mode variable.

### F13. Multi-tool yolo currently ignores `output_vars`

The current yolo collector checks `getattr(tool, "output_variable", None)`, but `ConversationTool` has `output_vars`. That means multi-tool yolo currently falls back to keys such as `tool.tool_type` instead of the declared output variable.

Implementation consequence:

- Multi-tool yolo must use `tool.output_vars[0]` when present.
- Better: synthesize a normal child response, feed it through the same decode/finalize/publish helper, and merge returned bindings.
- This avoids reintroducing singular/plural output-variable bugs.

### F14. Phase 0a modeling-artifacts path is currently forward-scaffolding

In the current `model_optimization/SOP.md`, `workflow_modeling_artifacts_path` is collected in Phase 0a but not consumed downstream. `workflow_modeling_artifacts_mode` is the cleaner new mode variable but also needs downstream usage if the workflow expects behavior changes.

Implementation consequence:

- Either wire Phase 2 / `understand-data` guidance to use the collected artifacts path, or explicitly document it as forward-scaffolding.
- Prefer wiring it now if the intent is user-visible value in this feature.
- Tests should assert not only that the variables are collected, but also that the intended downstream phase can consume them if that behavior is in scope.

## 2. Target Behavior

### Scenario A: path clarification

LLM/SOP may emit:

```json
{
  "type": "conversation",
  "name": "clarification",
  "arguments": {
    "prompt": "Choose the workflow target path.",
    "expected-input-type": "path",
    "prefix": "{{ session_root_path }}"
  },
  "output": "workflow_target_path"
}
```

Backend canonical form after parser canonicalization and runtime prefix rendering:

```json
{
  "tool_type": "clarification",
  "expected_input_type": "path",
  "prefix": "/session/root",
  "output_vars": ["workflow_target_path"]
}
```

UI behavior:

- Renders a path autocomplete widget.
- Uses OpenStartup `/api/workspace/path-complete`.
- Falls back to manual text if suggestions are unavailable.
- Stores a single path as a string after backend validation/finalization.

### Scenario B: single choice with conditional multi-path input

LLM/SOP may emit:

```json
{
  "type": "conversation",
  "name": "single-choice",
  "arguments": {
    "prompt": "How should modeling artifacts be selected?",
    "choices": [
      {
        "label": "Auto-discover",
        "value": "auto_discover",
        "description": "Let the workflow inspect the target repository."
      },
      {
        "label": "I will provide paths",
        "value": "manual_paths",
        "description": "Choose one or more artifact directories/files.",
        "input": {
          "name": "workflow_modeling_artifacts_path",
          "expected-input-type": "path",
          "allow-multiple-input": true,
          "prefix": "{{ session_root_path }}",
          "required": true,
          "serialization": "json"
        }
      }
    ],
    "allow_custom": false
  },
  "output": "workflow_modeling_artifacts_mode"
}
```

UI response for manual choice:

```json
{
  "choice_index": 1,
  "choice_value": "manual_paths",
  "inputs": {
    "workflow_modeling_artifacts_path": [
      "data/features/",
      "experiments/run_42/"
    ]
  }
}
```

Published SOP variables after backend validation/finalization:

```json
{
  "workflow_modeling_artifacts_mode": "manual_paths",
  "workflow_modeling_artifacts_path": "[\"data/features/\", \"experiments/run_42/\"]"
}
```

If a particular downstream tool requires comma-separated paths, the SOP/input spec can set:

```json
{
  "serialization": "comma"
}
```

UI response for auto-discover:

```json
{
  "choice_index": 0,
  "choice_value": "auto_discover",
  "inputs": {}
}
```

Published SOP variables:

```json
{
  "workflow_modeling_artifacts_mode": "auto_discover"
}
```

Do not put `"auto_discover"` in `workflow_modeling_artifacts_path`. The mode variable means strategy; the path variable means path values.

## 3. Design Decisions

### D1. Add typed input specs, not ad-hoc metadata blobs

Add a shared model:

```python
@dataclass
class InputFieldSpec:
    name: str = ""
    expected_input_type: str = "free_text"
    prefix: str = ""
    allow_multiple_input: bool = False
    required: bool = False
    placeholder: str = ""
    label: str = ""
    description: str = ""
    serialization: str = "auto"  # auto | scalar | json | comma
    yolo_default: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Add matching tool-level fields for standalone typed inputs:

```python
@dataclass
class ConversationTool:
    ...
    expected_input_type: str = "free_text"
    prefix: str = ""
    allow_multiple_input: bool = False
    serialization: str = "auto"  # auto | scalar | json | comma
```

Tool-level `serialization` applies to direct clarification/path inputs. `InputFieldSpec.serialization` applies to nested choice inputs and overrides only that nested value.

Extend `ChoiceItem`:

```python
@dataclass
class ChoiceItem:
    label: str
    value: str
    description: str = ""
    input: InputFieldSpec | None = None
```

Rationale:

- A choice still has a stable `value`.
- The selected choice can bind a mode variable.
- The nested input can bind a separate value variable.
- This avoids the brittle "choice is either label or input" union.

### D2. Use one canonical widget primitive, with compatibility routing

Create reusable UI primitives:

- `PathAutocompleteInput`
- `MultiValueInput`

Create a small registered wrapper:

- `PathInputWidget`

Canonical direct path prompt route:

```json
{
  "mode": "free_text",
  "metadata": {
    "widget_type": "path_input",
    "expected_input_type": "path",
    "prefix": "/session/root"
  }
}
```

Compatibility route:

- If `TextInputWidget` receives `expected_input_type == "path"` without `widget_type`, it may delegate to the same `PathAutocompleteInput`.
- This is a fallback, not a second competing design.

Do not add `InputMode.PATH` in this phase.

### D3. Keep metadata compatibility while adding first-class typed fields

Extend `InputModeConfig` and `ChoiceOption` with typed fields:

- `expected_input_type`
- `prefix`
- `allow_multiple_input`
- `input`

For one migration window, serialize both:

- first-class fields
- legacy `metadata.expected_input_type` / `metadata.prefix`

This avoids breaking older consumers while new UI reads typed fields first.

Exit criterion:

- Once OpenStartup consumes the first-class fields and `PathInputWidget` is registered in the shared-ui build, legacy metadata mirrors remain compatibility-only.
- Do not add new features that rely only on legacy metadata.
- A follow-up can remove duplicated legacy metadata after one stable release/window.

### D4. Runtime helpers are the source of truth

Add shared helpers, then call them from live runtime and handlers:

- `render_tool_runtime_fields(tool, *, render_string, context) -> ConversationTool`
- `build_input_mode_from_conversation_tool(tool) -> InputModeConfig`
- `decode_conversation_tool_response(tool, response) -> ConversationToolDecodeResult`
- `decode_compound_conversation_tool_responses(tools, values) -> ConversationToolDecodeResult`
- `finalize_input_value(spec, raw, session_root) -> FinalizedInputValue`
- `publish_conversation_tool_outputs(result) -> dict[str, str]`
- `_conversation_tool_render_context() -> dict[str, Any]`

The decode result must carry distinct bindings:

```python
@dataclass
class ConversationToolDecodeResult:
    display_value: str
    bindings: dict[str, Any]
    raw_values: dict[str, Any] = field(default_factory=dict)
```

`bindings` is the source of truth for `set_session_variables()` and for collected values returned to `__var__` action-tool substitution.

Legacy aliasing rule:

- For non-composite tools that declare multiple `output_vars`, preserve today's aliasing behavior: every declared output variable receives the same finalized scalar value.
- Composite choices are the exception: tool-level `output_vars` receive the selected choice value, while nested input specs bind their own `InputFieldSpec.name` values.
- Proposal selection keeps its existing comma-joined aliasing contract.

Use them in:

- `_build_input_mode`
- `_handle_conversation_tool`
- `_handle_conversation_tools`
- `ClarificationHandler`
- `SingleChoiceHandler`

This prevents handler tests from passing while OpenStartup live chat still fails.

Runtime rendering rule:

- Render `tool.prefix` and `choices[].input.prefix` after parsing and before building UI config/finalizing values.
- Use `prompt_renderer.render_string` with the same effective context as the last prompt render (`_last_template_feed`, `prior_context`, and session variables).
- Call `render_tool_runtime_fields(...)` at the class-method call sites before `_build_input_mode(...)`: once in `_handle_conversation_tool` for the single-tool path, and once per tool in `_handle_conversation_tools` for the compound path.
- Keep `_build_input_mode(...)` render-free because it is module-level and has no `self`, `prompt_renderer`, or `_last_template_feed`.
- Implement `_conversation_tool_render_context()` on `ConversationalInferencer` so both call sites use the same context merge order. Include `_last_template_feed`, because `_render_prompt()` inserts `session_root_path` there.
- If no renderer is available, leave the string untouched and let backend validation fail clearly rather than silently treating a Jinja expression as a filesystem path.

### D5. Value lifecycle: typed decode, explicit publication

The value lifecycle is:

1. UI payload: typed JSON, e.g. `["path/a", "path/b"]`.
2. Decode result: typed Python values.
3. Finalization: validate paths, normalize relative paths, reject traversal.
4. Publication: serialize according to the input spec.

Default publication:

- single scalar -> scalar string
- multi path -> JSON array string
- proposal IDs -> comma string, preserving existing contract
- explicit `serialization: "comma"` -> comma string
- explicit `serialization: "json"` -> JSON array string
- `serialization: "auto"` -> scalar when `allow_multiple_input` is false, JSON array string when `allow_multiple_input` is true, except for explicitly preserved legacy contracts such as proposal IDs

Never use `str(list)`.

Do not claim the variable manager has full typed-rendering semantics until that is designed and tested end-to-end.

### D6. Path validation is backend-authoritative

Autocomplete is convenience, not security.

Every submitted path must be validated against:

- declared prefix
- session root or allowed root

Use:

```python
resolved.relative_to(root_resolved)
```

Use `Path.resolve().relative_to(root_resolved)` as the primary containment check and catch `ValueError`. Avoid string `startswith`. Use `os.path.commonpath` only as a fallback if a call site cannot use `Path` objects cleanly.

Reject sibling-prefix attacks and `../` traversal.

### D7. Host boundary for path completion

Shared-ui must not inspect the filesystem directly.

Shared-ui accepts:

```ts
type PathAutocompleteProvider = (request: {
  prefix: string;
  partial: string;
  dirsOnly?: boolean;
  limit?: number;
}) => Promise<Array<{ name: string; path: string; is_dir: boolean }>>;
```

OpenStartup wires that provider to:

```text
GET /api/workspace/path-complete?prefix=...&partial=...&dirs_only=...&limit=...
```

The OpenStartup route validates the prefix against the session root / allowed root policy.

### D8. Yolo uses the same finalization path

Yolo synthesis must not bypass output binding/finalization.

Rules:

- Use tool output vars as keys, not only `tool_type`.
- Resolve expected-input-type-aware defaults.
- For path input, return a real path only when a safe default is configured and available.
- Gate means: do not synthesize a value; surface the widget to the user even in yolo mode, with a clear reason that the value cannot be safely fabricated.
- Feed synthesized values through the same finalizer/serializer used by normal UI responses.

## 4. Implementation Plan

### Commit 0: Preflight source locks

Add failing tests before implementation.

Must cover:

- `single-choice` and `single choice` normalize to `single_choice`.
- `expected-input-type` normalizes to `expected_input_type`.
- `allow-multiple-input` normalizes to `allow_multiple_input`.
- `output: "x"` normalizes to `["x"]`.
- Nested `choices[].input` survives parser/model round trip.
- `_build_input_mode` preserves choice descriptions.
- `_build_input_mode` preserves `choices[].input`.
- Templated `prefix: "{{ session_root_path }}"` renders to the active session root before reaching the UI.
- Nested `choices[].input.prefix` renders the same way.
- `_conversation_tool_render_context()` exposes `session_root_path` at both render call sites after a normal prompt render.
- Composite single-tool decode returns distinct bindings for mode var and nested input var.
- Legacy non-composite tools with multiple `output_vars` still alias all declared vars to the same finalized value.
- Compound composite decode reads the child payload under the tool's `output_var`.
- Multi-tool yolo uses `output_vars`, not `output_variable` or `tool_type`.
- Single-tool yolo also returns/binds through declared `output_vars`, not a bare unbound string.
- Compound decode does not use raw `str(list)`.
- Path containment rejects sibling-prefix attacks.
- Current `model_optimization` either consumes `workflow_modeling_artifacts_path` downstream or explicitly documents it as forward-scaffolding.

### Commit 1: Parser canonicalization

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_response_parser.py`
- possibly shared helper in the same package

Changes:

- Add recursive key canonicalization.
- Normalize tool names before constructing `ConversationTool`.
- Normalize `output` / `output_vars` to `list[str]`.
- Apply canonicalization to:
  - legacy `<ConversationTools>`
  - new `ToolsToInvoke`
  - nested `choices`
  - nested `choices[].input`

Tests:

- Hyphenated keys.
- Space/hyphen tool names.
- Underscore form wins when both hyphen and underscore are present.
- String output and list output both normalize.

### Commit 2: Conversation tool schema

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py`

Changes:

- Add `InputFieldSpec`.
- Add `allow_multiple_input` to `ConversationTool`.
- Add `serialization` to `ConversationTool` for standalone typed inputs.
- Add `ChoiceItem.input`.
- Add robust `to_dict` / `from_dict`.
- Serialize non-default:
  - `expected_input_type`
  - `prefix`
  - `output_vars`
  - `allow_multiple_input`
  - `choices[].input`
  - `serialization`

Tests:

- Backward-compatible label-only choices.
- Input-bearing choices.
- ConversationTool round-trip with path fields.
- No field loss in `to_dict()`.

### Commit 3: Input mode and live runtime build path

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/input_modes.py`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py`
- handler files as shared-helper callers

Changes:

- Add typed fields to `InputModeConfig`.
- Add `input` to `ChoiceOption`.
- Add `render_tool_runtime_fields()` or equivalent normalization before input-mode construction.
- Render `tool.prefix` and nested `choices[].input.prefix` using the active prompt-render context.
- Wire that render helper in the two class-method call sites before `_build_input_mode(...)`; do not make `_build_input_mode(...)` reach for `self`.
- Update `_build_input_mode()` or replace it with `build_input_mode_from_conversation_tool()`.
- Preserve descriptions for simple choices.
- Preserve nested input specs.
- For path clarification:
  - keep `mode = free_text`
  - set typed fields
  - set `metadata.widget_type = "path_input"`
  - keep legacy metadata path fields for compatibility

Tests:

- Path clarification input mode.
- Path clarification with `prefix: "{{ session_root_path }}"` renders to the real session root.
- Composite single-choice nested input prefix renders to the real session root.
- Simple free-text unchanged.
- Simple single-choice descriptions preserved.
- Composite single-choice option input serialized.
- Proposal selection unchanged.
- Confirmation unchanged.

### Commit 4: Runtime response decode, finalization, and publication

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py`
- optional helper module such as `conversation_tool_runtime.py`
- handler files as shared-helper callers

Changes:

- Add decode result type with `bindings: dict[str, Any]`.
- Decode structured payloads:
  - `{content}`
  - `{choice_index}`
  - `{choice_index, choice_value}`
  - `{choice_index, choice_value, inputs}`
  - `{paths}`
  - `{custom_text}`
  - proposal selection lists
  - confirmation `param_overrides` and `variables`
- Bind top-level choice output vars to selected choice value.
- Bind nested input `name` to finalized nested input value.
- For composite choices, merge both bindings into one result; do not alias all output vars to one value.
- In compound mode, read each child payload from `values[tool.output_vars[0]]` / `values[tool.output_var]`, then pass that child payload through the same single-tool decoder.
- Preserve child payload objects until decode; do not stringify `{choice_index, inputs}` into the mode variable.
- Validate path values.
- Publish output variables through explicit serializer.
- Replace compound `str(raw_value)` path.
- Preserve proposal selection comma-joined contract.

Tests:

- Manual path choice sets mode and serialized path value.
- Manual path choice returns bindings for both `workflow_modeling_artifacts_mode` and `workflow_modeling_artifacts_path`.
- Auto-discover sets mode only.
- Legacy multi-output-var tools still alias all tool-level `output_vars` to the same finalized scalar value.
- No stale path from switching choices.
- Compound response with nested payload under the mode output var binds both variables correctly.
- Multi-path JSON serialization by default.
- Explicit comma serialization.
- Raw `str(list)` never appears.
- Compound and single-tool paths agree.
- Submitted path traversal rejected.

### Commit 5: Yolo defaults and yolo binding

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/clarification/tool.json`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py`

Changes:

- Add expected-input-type-aware yolo default resolution.
- Replace clarification's unconditional prose default for path inputs.
- Use session root path only when configured/available.
- Add a gate mode for inputs that cannot be safely fabricated.
- Fix the current singular/plural bug: use `tool.output_vars[0]` when present, not nonexistent `tool.output_variable`.
- Prefer generating a normal child response and passing it through the same decode/finalize/publish path.
- Apply that rule to both branches of `_synthesize_yolo_collected`: the single-tool branch must return/apply bindings keyed by declared output vars, not only a bare string summary.
- Ensure yolo calls `set_session_variables()` with the same published bindings that interactive decode would produce.
- Ensure multi-tool yolo merges returned bindings, rather than keying results by `tool_type`.

Tests:

- Free-text yolo default remains backward compatible.
- Path yolo returns valid path or gates.
- URL yolo gates unless explicitly configured.
- Single-tool yolo binds the declared output variable.
- Multi-tool yolo keys by output var.
- Multi-tool yolo for two conversation tools returns both declared output variables.
- Yolo path output is finalized/serialized the same as UI path output.

### Commit 6: Path completion helper and OpenStartup route

AgentFoundation files:

- Factor helper from:
  `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/webui/backend/routes/workspace_routes.py`

Suggested helper:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/workspace/path_completion.py`

OpenStartup files:

- Add:
  `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/routes/workspace_routes.py`
- Update:
  `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/main.py`

Endpoint:

```text
GET /api/workspace/path-complete
```

Changes:

- Reuse one helper in AgentFoundation and OpenStartup.
- Harden containment with `Path.resolve().relative_to(...)`.
- Constrain OpenStartup prefix to session root / allowed root.
- Keep bounded results.
- Hide dotfiles unless explicitly enabled later.

Tests:

- Basic suggestions.
- `dirs_only` honored.
- Files returned when `dirs_only=false`.
- `../` traversal rejected.
- Sibling-prefix attack rejected.
- OpenStartup route is mounted.

### Commit 7: Shared-ui path widgets

Files under:

`/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/react-shared/src`

Add:

- `inputs/PathAutocompleteInput.js`
- `inputs/MultiValueInput.js`
- `inputs/PathInputWidget.js`
- path suggestion hook/provider utility as needed

Update:

- `protocol/registerBuiltins.js`
- `protocol/ConversationToolWidget.js`
- `inputs/TextInputWidget.js`
- `inputs/SingleChoiceWidget.js`
- `inputs/GroupedWidget.js` if compound threading requires it
- `index.js`
- package build exports if needed

Behavior:

- Direct path prompt routes through `metadata.widget_type = "path_input"`.
- `TextInputWidget` delegates to `PathAutocompleteInput` only as compatibility fallback when path metadata arrives without `widget_type`.
- `SingleChoiceWidget` renders `option.input` only when that option is selected.
- Multi-value path input can add/remove values.
- Missing provider gracefully degrades to manual text.

Tests:

- Registered `path_input` resolves.
- Direct path widget calls provider.
- Provider failure fallback.
- Prefix adornment shown.
- Single-choice nested path input appears only for selected option.
- Switching away clears stale nested input from submitted payload.
- Multi-value add/remove.
- Existing simple TextInputWidget and SingleChoiceWidget behavior unchanged.

### Commit 8: OpenStartup UI provider wiring and shared-ui build

Files:

- `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/ui/src/components/views/ManagerChatView.js`
- `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/ui/src/utils/api.js` or suitable API helper
- shared-ui package files as needed

Changes:

- Thread `pathAutocompleteProvider` into `ConversationToolWidget`.
- Provider calls `/api/workspace/path-complete`.
- Avoid `/api/api/...`.
- Keep local OpenStartup widget wrappers as re-exports.
- Build shared-ui before OpenStartup verification because OpenStartup consumes the built package.

Verification:

- `npm run build` in shared-ui package.
- OpenStartup UI compile.
- Live pending input renders path widget.

### Commit 9: Tool docs, SOP docs, and E2E lock

Files:

- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/clarification/tool.json`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/single_choice/tool.json`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/multiple_choice/tool.json`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/templates/conversation.jinja2`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/sops/model_optimization/SOP.md`

Changes:

- Document canonical underscore args.
- Keep parser accepting hyphen/space aliases.
- Document `choices[].input`.
- Document `allow_multiple_input`.
- Document `serialization`.
- Update Phase 0a to use:
  - `workflow_modeling_artifacts_mode`
  - `workflow_modeling_artifacts_path`
- Decide and document Phase 2 consumption:
  - preferred: update Phase 2 / `understand-data` guidance to use `workflow_modeling_artifacts_path` when `workflow_modeling_artifacts_mode == "manual_paths"`
  - acceptable only if intentional: mark `workflow_modeling_artifacts_path` as forward-scaffolding and explain that no downstream phase consumes it yet
- Add E2E test for verbatim Phase 0a.

E2E assertions:

- Path clarification parses and renders.
- Single-choice composite parses and renders.
- Manual response binds mode and serialized path value.
- Auto response binds mode only.
- If Phase 2 consumption is wired, the E2E verifies the downstream phase can see/use the artifacts path.
- If Phase 2 consumption is deferred, the E2E verifies the SOP explicitly documents the variable as forward-scaffolding.
- Path completion route returns suggestions.
- Existing proposal selection still works.
- Existing confirmation still works.

## 5. Comparison of the Three Plans

### Codex plan

Best parts retained:

- Correct high-level protocol shape.
- Host-injected path provider.
- Separate mode and path variables.
- Live-runtime awareness.
- Server-side validation language.

Weaknesses fixed here:

- It previously left the list/string publication boundary too vague.
- It did not sufficiently emphasize the yolo default bug.
- It did not sufficiently emphasize multi-tool yolo/output-var binding.
- It did not explicitly call out that parsed `prefix` values containing `{{ ... }}` need runtime rendering before UI/finalization.
- It did not spell out two-source composite binding mechanics deeply enough.

### Claude plan

Best parts retained:

- Existing `/path-complete` awareness.
- OpenStartup route gap.
- Bad clarification yolo default.
- Compound `str(raw_value)` bug.
- Efficient critical-file list.

Weaknesses corrected here:

- Comma-joined paths are not made globally canonical.
- `TextInputWidget` branching is compatibility fallback, not the only design.
- Path completion must be hardened; do not reuse the existing guard unchanged.

### Typed-input/composite-choice plan

Best parts retained:

- Strong layered test discipline.
- Commit/checklist format.
- E2E Phase 0a mindset.

Weaknesses corrected here:

- No `InputMode.PATH` for this phase.
- No new `/api/paths` or WebSocket path-suggestion endpoint.
- No handler-only implementation.
- No registry/branching contradiction.
- No claim that enum `_missing_` is the main fix.
- No claim that `to_dict()` stripping is the main live blocker.

## 6. Acceptance Criteria

Implementation is done only when:

- Legacy free-text clarification works.
- Legacy simple single choice works.
- Legacy multiple choice works.
- Legacy proposal selection works.
- `single-choice`, `single choice`, and `single_choice` all work.
- `output: "x"` and `output: ["x"]` both work.
- `expected-input-type` and `allow-multiple-input` aliases work.
- Direct path clarification renders path autocomplete.
- Parsed `prefix: "{{ session_root_path }}"` does not reach the UI literally.
- Single-choice selected option can reveal a path input.
- Composite single-choice stores both selected mode and nested input value.
- Multi-value path input publishes through declared serialization.
- Raw `str(list)` never appears in stored variables.
- Compound conversation tools use the same finalizer as single tools.
- Compound composite responses are decoded from the child payload under each tool's output variable.
- Path completion is mounted in OpenStartup.
- Path completion and submitted paths reject traversal.
- Path yolo never fabricates prose as a path.
- Multi-tool yolo uses declared `output_vars`.
- `workflow_modeling_artifacts_path` is either consumed downstream or explicitly documented as forward-scaffolding.
- Shared-ui builds and OpenStartup consumes the built package.

## 7. Non-Goals

- Full JSON-schema form builder.
- Arbitrary nested conditional trees.
- Global filesystem browser outside session/allowed roots.
- Full migration to handler registry.
- Removing legacy metadata compatibility.
- Adding `InputMode.PATH`.
- Designing typed variable rendering across all Jinja/action-tool paths.

Those are future work after the path clarification and composite single-choice flow is stable.
