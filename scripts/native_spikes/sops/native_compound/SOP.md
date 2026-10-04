# Native Compound Setup

Collect a target path and an artifacts mode in one step (like `model_optimization` Phase 0a), then write a brief.

[__keywords__] compound setup, native compound, setup target
[__example_requests__]
- start the native_compound SOP
- set up the compound workflow for <path>

## Phase 0a -- Setup target path and artifacts mode
[__initial__]

[__requires user input__] Use a `clarification` conversation tool to set up this workflow's target path. For tool args:
- set `expected_input_type` to "path"
- **If the user already gave the target path in their request, set `default` to it** so the input pre-fills and the user just confirms.
- set `output` to "workflow_target_path".

[__requires user input__] Use a `single_choice` conversation tool to set up this workflow's modeling artifacts location. Set the tool-level `output` to "workflow_modeling_artifacts_mode". Present two choices:
- `{ "label": "Auto discover", "value": "auto_discover" }`
- `{ "label": "Specify paths", "value": "manual_paths" }`

**Tools**[__required__]:
- clarification
- single_choice

## Phase 1 -- Write the brief
[__depends on__ Phase 0a]

Write a short brief on `{{ workflow_target_path }}` by calling the write_brief tool.

**Tools**[__required__]:
- /write-brief <topic>
