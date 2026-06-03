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
- set `expected-input-type` to "path"
- set `prefix` to the session root path: `{{ session_root_path }}`. This enables path auto-completion in the UI. The workflow target path MUST be a subpath under this session root path.
- set `output` to "workflow_target_path".

The workflow target path can be either:
- A **directory** — the workflow investigates all code within it.
- A **file** — treated as the entry point for code investigation. The workflow will start from this file and explore its dependencies and surrounding codebase.

**Tools**[__must__]:
- clarification

## Phase 0b -- Setup evolution strategy
[__depends on__ Phase 0a]

[__requires user input__] Use a `single_choice` conversation tool to let the user pick an evolution strategy. Set "evolution_strategy" as the `output` variable name. Choices:
- Paradigm-Shifting Innovation — aggressively explore state-of-the-art architectures, novel techniques, and breakthrough design changes
- Incremental Improvement — find targeted gains through structured analysis of existing bottlenecks and component-level optimizations
- Efficiency Optimization — reduce training/inference cost — memory footprint, latency, throughput, and computational efficiency
- Holistic Improvement — systematically evaluate all dimensions — architecture, efficiency, quality, and robustness — for balanced gains

**Tools**[__must__]:
- single-choice

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

[__requires user input__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `review proposal tool` (DO NOT use the simpler `confirmation tool`). The tool will decide how to present the proposal to the user. The user selects which proposals to advance to Phase 3.

## Phase 4 -- Implementation, Experiment & Analysis
[__depends on__ Phase 3b; __branch__]

Plan and implement the proposed changes, run experiments to validate, and analyze results to identify bottlenecks and improvement opportunities.

**Tools**[__must__]:
- task

## Phase 4b -- Summary & Evolve
[__depends on__ Phase 4; __goto__ Phase 3 __afterwards__ __if__ `continue`]

After analysis reveals bottlenecks or opportunities, summarize findings and decide whether to loop back for another research-proposal-experiment cycle. Each iteration builds on prior results and workspace artifacts. Present results and ask the user whether to continue evolving or conclude.
