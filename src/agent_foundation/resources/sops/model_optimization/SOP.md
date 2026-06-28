# Machine Learning Model Evolution

A phased workflow for systematically optimizing ML model architectures: from codebase investigation through research-driven proposals to validated implementation. The user selects an evolution strategy; the Orchestrator executes each phase with confirmation gates.

[__keywords__] optimize model, improve model, model evolution, model architecture, modeling experiment, run modeling experiments, explore modeling experiments, systematically improve or optimize a model
[__example_requests__]
- optimize the recommendation model
- improve training efficiency for the ranking model
- evolve the search model architecture
- explore / run modeling experiments on <model codebase>
- systematically improve the <model> architecture

## Phase 0a -- Setup workflow target path
[__initial__]

[__requires user input__] Use a `clarification` conversation tool to set up this workflow's target path. For tool args:
- set `expected_input_type` to "path"
- set `prefix` to the session root path: `{{ session_root_path }}`. This enables path auto-completion in the UI. The workflow target path MUST be a subpath under this session root path.
- **If the user already gave the target path/codebase in their request, set `default` to it** so the input pre-fills and the user just confirms (instead of re-typing what they already said).
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

## Phase 1 -- Codebase & Data Investigation
[__depends on__ Phase 0b]

Perform comprehensive and in-depth understanding and analysis of both the following codebase and modeling artifacts (data, scripts, prior experiment results/learnings). Use "modeling" as the `template-version` for ML-focused investigation.

Main codebase is at `{{ workflow_target_path }}`. Codebase is related to modeling work, but not necessarily only about modeling; e.g., model architecture, data/training/inference pipeline or infra, model serving, etc.

{% if workflow_modeling_artifacts_mode == "manual_paths" %}The user specified the modeling artifacts at: `{{ workflow_modeling_artifacts_path }}` — prioritize these locations in the investigation into modeling artifacts.{% else %}Auto-discover the modeling-related artifacts under above codebase.{% endif %}


**Tools**[__must__]:
- understand-codebase

### Phase 1b -- Codebase Documentation Review
[__depends on__ Phase 1]

[__requires user input__] Present the codebase & modeling artifacts investigation outcome to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated documentation. Summarize key architectural findings and invite the user to review the full documentation via the "View Documentation" button. Only proceed to the next phase after the user confirms they are satisfied.

**Tools**[__must__]:
- confirmation

## Phase 2 -- Research & Proposal
[__depends on__ Phase 1b]

[__requires user input__] Break down the research goal into sub-queries, execute parallel deep research streams, generate architecture proposals, and synthesize them into a unified design.

The research goal should be derived from the chosen strategy and the findings from Phase 1. The `--docs-path` and `--workflow-target-path` arguments are auto-populated from prior phase outputs if not explicitly provided.

**Tools**[__must__]:
- research-propose <goal> --docs <reference_documentation>

### Phase 2b -- Proposal Review & Selection
[__depends on__ Phase 2]

[__requires user input__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `proposal_selection` conversation tool (or a `confirmation` tool if `proposal_selection` is unavailable). Pass the tool argument `proposals_path = {{ workspace_path__research_propose }}/outputs/proposals.json` — this is the `proposals.json` that research-propose writes (the BTA INVARIANT location), and `workspace_path__research_propose` is published into the workflow context by the bridge dispatcher when Phase 2 runs. The user reviews the proposals (ID, title, impact, complexity, dependencies) and selects which to advance to Phase 3. The output variable MUST be named `selected_proposal_ids` for this conversation tool — Phase 3 consumes it by that name.

**Tools**[__must__]:
- proposal_selection

## Phase 3 -- Implementation, Experiment & Analysis
[__depends on__ Phase 2b; __branch__]

For each selected proposal from Phase 2b, plan and implement the proposed changes, run experiments to validate, and analyze results to identify bottlenecks and improvement opportunities. Invoke the `task` tool with:
- `--use-proposal {{ workspace_path__research_propose }}/outputs/proposals.json` (the proposals produced in Phase 2)
- `--proposal-ids {{ selected_proposal_ids }}` (the Phase 2b selection, comma-joined, e.g. `P1,P3`)

The `task` tool inlines the selected proposals into its plan, implements the changes, runs experiments to validate, and records results.

**Tools**[__must__]:
- task

## Phase 3b -- Summary & Evolve
[__depends on__ Phase 3; __goto__ Phase 2 __afterwards__ __if__ `continue`]

After analysis reveals bottlenecks or opportunities, summarize findings and decide whether to loop back for another research-proposal-experiment cycle. Each iteration builds on prior results and workspace artifacts. Present results and ask the user whether to continue evolving or conclude.
