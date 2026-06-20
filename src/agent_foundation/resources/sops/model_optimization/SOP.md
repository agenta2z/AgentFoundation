# Machine Learning Model Evolution

A phased workflow for systematically optimizing ML model architectures: from codebase investigation through research-driven proposals to validated implementation. The user selects an evolution strategy; the Orchestrator executes each phase with confirmation gates.

[__keywords__] optimize model, improve model, model evolution, model architecture
[__example_requests__]
- optimize the recommendation model
- improve training efficiency for the ranking model
- evolve the search model architecture

## Phase 0a -- Setup workflow target path
[__initial__]

[__requires user input__] Use a `clarification` conversation tool to set up this workflow's target path. For tool args:
- set `expected_input_type` to "path"
- set `prefix` to the session root path: `{{ session_root_path }}`. This enables path auto-completion in the UI. The workflow target path MUST be a subpath under this session root path.
- set `output` to "workflow_target_path". This workflow target path can be either:
  * A **directory** — the workflow investigates all code within it.
  * A **file** — treated as the entry point for code investigation. The workflow will start from this file and explore its dependencies and surrounding codebase.

[__requires user input__] Use a `single_choice` conversation tool to set up this workflow's modeling artifacts location (data, scripts, previous experiment results and learnings, etc.). Set the tool-level `output` to "workflow_modeling_artifacts_mode" (records which option the user chose). Present two choices:
- The first choice: `{ "label": "Auto discover", "value": "auto_discover" }` — the workflow infers the artifacts from the target repository.
- The second choice carries an embedded path input: `{ "label": "Specify paths", "value": "manual_paths", "input": { "name": "workflow_modeling_artifacts_path", "expected_input_type": "path", "allow_multiple_input": true, "prefix": "{{ session_root_path }}", "required": true } }`. Selecting it reveals a path picker accepting one or more directory/file paths, bound to `workflow_modeling_artifacts_path`.

Outcome: picking "Auto discover" sets `workflow_modeling_artifacts_mode` = "auto_discover" (and leaves `workflow_modeling_artifacts_path` unset); picking "Specify paths" sets `workflow_modeling_artifacts_mode` = "manual_paths" and binds the chosen path(s) to `workflow_modeling_artifacts_path`.

**Tools**[__must__]:
- clarification
- single_choice

## Phase 0b -- Setup model evolution strategy
[__depends on__ Phase 0a]

[__requires user input__] Use a `single_choice` conversation tool to let the user pick an evolution strategy. Set "evolution_strategy" as the `output` variable name. Choices:
- Paradigm-Shifting Innovation — aggressively explore state-of-the-art architectures, novel techniques, and breakthrough design changes
- Incremental Improvement — find targeted gains through structured analysis of existing bottlenecks and component-level optimizations
- Efficiency Optimization — reduce training/inference cost — memory footprint, latency, throughput, and computational efficiency
- Holistic Improvement — systematically evaluate all dimensions — architecture, efficiency, quality, and robustness — for balanced gains

**Tools**[__must__]:
- single_choice

## Phase 1 -- Codebase Investigation
[__depends on__ Phase 0b]

Perform comprehensive and in-depth understanding and analysis of the codebase at `{{ workflow_target_path }}`. Use "modeling" as the `template-version` for ML-focused investigation.

**Tools**[__must__]:
- understand-codebase

### Phase 1b -- Codebase Documentation Review
[__depends on__ Phase 1]

[__requires user input__] Present the codebase investigation outcome to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated code documentation. Summarize key architectural findings and invite the user to review the full documentation via the "View Documentation" button. Only proceed to the next phase after the user confirms they are satisfied.

**Tools**[__must__]:
- confirmation

## Phase 2 -- Data Investigation
[__depends on__ Phase 1b]

Investigate the data landscape for `{{ workflow_target_path }}`. Use "modeling" as the `template-version` for ML-focused data investigation.
{% if workflow_modeling_artifacts_mode == "manual_paths" %}The user specified the modeling artifacts (data, scripts, prior experiment results/learnings) at: `{{ workflow_modeling_artifacts_path }}` — prioritise these locations in the investigation.{% else %}Auto-discover the modeling artifacts (data, scripts, prior experiment results/learnings) within the target.{% endif %}

**Tools**[__must__]:
- understand-data

### Phase 2b -- Data Investigation Review
[__depends on__ Phase 2]

[__requires user input__] Present data investigation findings to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated data documentation. Summarize key data quality findings, pipeline bottlenecks, and dataset characteristics, and invite the user to review the full documentation via the "View Documentation" button. Only proceed after the user confirms.

**Tools**[__must__]:
- confirmation

## Phase 3 -- Research & Proposal
[__depends on__ Phase 2b]

[__requires user input__] Break down the research goal into sub-queries, execute parallel deep research streams, generate architecture proposals, and synthesize them into a unified design.

The research goal should be derived from the chosen strategy and the findings from Phase 1. The `--docs-path` and `--workflow-target-path` arguments are auto-populated from prior phase outputs if not explicitly provided.

**Tools**[__must__]:
- research-propose <goal> --docs <reference_documentation>

### Phase 3b -- Proposal Review & Selection
[__depends on__ Phase 3]

[__requires user input__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `proposal_selection` conversation tool (or a `confirmation` tool if `proposal_selection` is unavailable). Pass the tool argument `proposals_path = {{ workspace_path__research_propose }}/outputs/proposals.json` — this is the `proposals.json` that research-propose writes (the BTA INVARIANT location), and `workspace_path__research_propose` is published into the workflow context by the bridge dispatcher when Phase 3 runs. The user reviews the proposals (ID, title, impact, complexity, dependencies) and selects which to advance to Phase 4. The output variable MUST be named `selected_proposal_ids` for this conversation tool — Phase 4 consumes it by that name.

**Tools**[__must__]:
- proposal_selection

## Phase 4 -- Implementation, Experiment & Analysis
[__depends on__ Phase 3b; __branch__]

For each selected proposal from Phase 3b, plan and implement the proposed changes, run experiments to validate, and analyze results to identify bottlenecks and improvement opportunities. Invoke the `task` tool with:
- `--use-proposal {{ workspace_path__research_propose }}/outputs/proposals.json` (the proposals produced in Phase 3)
- `--proposal-ids {{ selected_proposal_ids }}` (the Phase 3b selection, comma-joined, e.g. `P1,P3`)

The `task` tool inlines the selected proposals into its plan, implements the changes, runs experiments to validate, and records results.

**Tools**[__must__]:
- task

## Phase 4b -- Summary & Evolve
[__depends on__ Phase 4; __goto__ Phase 3 __afterwards__ __if__ `continue`]

After analysis reveals bottlenecks or opportunities, summarize findings and decide whether to loop back for another research-proposal-experiment cycle. Each iteration builds on prior results and workspace artifacts. Present results and ask the user whether to continue evolving or conclude.
