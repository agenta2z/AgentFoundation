## Phase 0 -- Setup: `workflow_target_path` and `strategy`
[__initial__]

User needs to specify the workflow target code path and an evolution strategy. The workflow target path should be a subpath under the session root path: `{{ session_root_path }}`.

The workflow target path can be either:
- A **directory** — the workflow investigates all code within it.
- A **file** — treated as the entry point for code investigation. The workflow will start from this file and explore its dependencies and surrounding codebase.

When asking for the workflow target path, use a `clarification` conversation tool with `expected_input_type: "path"` and `prefix` set to `{{ session_root_path }}`. This enables path autocomplete in the UI. The output variable MUST be named `workflow_target_path` for this conversation tool.

User also need to specify one of the following evolution strategies:
- Paradigm-Shifting Innovation — aggressively explore state-of-the-art architectures, novel techniques, and breakthrough design changes
- Incremental Improvement — find targeted gains through structured analysis of existing bottlenecks and component-level optimizations
- Efficiency Optimization — reduce training/inference cost — memory footprint, latency, throughput, and computational efficiency
- Holistic Improvement — systematically evaluate all dimensions — architecture, efficiency, quality, and robustness — for balanced gains

**Tools**[__must__]:
- /set-workflow-target-path <path>
- /set-strategy <strategy name>

## Phase 1 -- Codebase Investigation: `codebase_understanding`
[__depends on__ Phase 0]

[__requires confirmation__] **STOP: Before executing any tool for this phase, you MUST use a `confirmation` conversation tool to get explicit user approval.** Do NOT invoke /understand-codebase until the user confirms. Present a summary of what will be investigated (target path, strategy, estimated scope) and let the user approve or adjust.

Once confirmed, perform in-depth analysis of the target codebase at `{{ workflow_target_path }}` to understand its design, architecture, dependencies, data flow, model structure, and extension points. Produces structured documentation of findings with a navigable HTML documentation site built via Sphinx.

**Tools**[__requires_confirmation_first__]:
- /understand-codebase <path>

### Phase 1b -- Documentation Review
[__depends on__ Phase 1]

[__requires confirmation__] After the codebase investigation completes, present the results to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated documentation. Summarize key architectural findings and invite the user to review the full documentation via the "View Documentation" button. Only proceed to Phase 2 after the user confirms they are satisfied.

## Phase 2 -- Research & Proposal: `research_proposals`
[__depends on__ Phase 1b]

[__requires confirmation__] Break down the research goal into sub-queries, execute parallel deep research streams, generate architecture proposals, and synthesize them into a unified design.

The research goal should be derived from the chosen strategy and the findings from Phase 1. The `--docs-path` and `--workflow-target-path` arguments are auto-populated from prior phase outputs if not explicitly provided.

Command: `/research-propose <goal>`

### Phase 2b -- Proposal Review & Selection
[__depends on__ Phase 2]

[__requires confirmation__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `proposal_selection` conversation tool (NOT a simple `confirmation`). The system automatically parses the unified plan and populates the widget with proposals organized by implementation phase. The user selects which proposals to implement and can add custom research queries. Their selections guide Phase 3.

## Phase 3 -- Implementation, Experiment & Analysis: `experiment_result`
[__depends on__ Phase 2b; __for each__ `research_proposal` __in__ `research_proposals`]

Plan and implement the proposed changes, run experiments to validate, and analyze results to identify bottlenecks and improvement opportunities.

When hypotheses are selected via `proposal_selection`, the system automatically creates a task queue and executes each hypothesis as a `/task` invocation. Tasks run sequentially based on `max_parallel_tasks` (default 1). Do NOT manually invoke `/task` for queued hypotheses — the system handles execution automatically.

Command: `/task <request>`

## Phase 4 -- Summary & Evolve: decide if we should `continue` another research-proposal-experiment cycle.
[__depends on__ Phase 3; __go to__ Phase 2 __if__ `continue`]

After analysis reveals bottlenecks or opportunities, loop back to any earlier
phase. Each iteration builds on prior results and workspace artifacts. The
Evolve methodology continuously refines approaches based on experimental
findings.
