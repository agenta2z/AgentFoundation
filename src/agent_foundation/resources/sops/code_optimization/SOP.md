## Phase 0 -- Setup
[__initial__]

User needs to specify the workflow target code path (the codebase or sub-tree to optimize) and an optimization strategy. The workflow target path MUST be a subpath under the session root path: `{{ session_root_path }}`.

The workflow target path can be either:
- A **directory** — the workflow investigates all code within it (recommended for service-level optimization).
- A **file** — treated as the entry point for code investigation. The workflow will start from this file and explore its dependencies and surrounding codebase.

When asking for the workflow target path, use a `clarification` conversation tool with `expected_input_type: "path"` and `prefix` set to `{{ session_root_path }}`. This enables path autocomplete in the UI. The output variable MUST be named `workflow_target_path` for this conversation tool.

**Tools**[__must__]:
- /set-workflow-target-path <path>

## Phase 1 -- Codebase Investigation:
[__depends on__ Phase 0]

[__requires confirmation__] **IMPORTANT**: Before executing any tool for this phase, you MUST first use a `confirmation tool` to get explicit user approval.** Do NOT invoke /understand-codebase until the user confirms. Present a summary of what will be investigated (target path, strategy, estimated scope, expected output structure) and let the user approve or adjust. Once confirmed, continue with the following codebase understanding operation.

Once confirmed, perform in-depth static analysis of the target codebase at `{{ workflow_target_path }}` to understand its design, architecture, module boundaries, dependencies, data flow, hotspots, and historical change patterns. Produces reStructured documentation of findings with a navigable HTML documentation site. 

**Tools**[__must__]:
- /understand-codebase <codebase_path>

### Phase 1b -- Codebase Documentation Review
[__depends on__ Phase 1]

[__requires confirmation__] After the codebase investigation completes, present the results to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated documentation. Summarize key architectural findings (top hotspots, biggest files, ownership map, refactor candidates) and invite the user to review the full documentation via the "View Documentation" button. Only proceed to the next Phase after the user confirms they are satisfied.

## Phase 2 -- System & Signals Investigation
[__depends on__ Phase 1b]

[__requires confirmation__] Before executing any tool for this phase, you MUST first use a `confirmation tool` to get explicit user approval. Static code analysis alone is insufficient for code optimization — runtime signals are required to ground prioritization in real-world impact. This phase explicitly investigates **operational reality**, not just code structure. nce confirmed, continue with the following codebase system & signals understanding operation.

Investigate the production system and operational signals associated with the target code. Sources to consult in priority order (use whatever is reachable):

1. **Service & SLO catalog** (PRIMARY-SOURCE — highest confidence)
   - Tome / SLO control-plane API for capabilities, SLOs, breach events, low-traffic suppression status, post-incident SLO concessions
   - Service inventory (e.g. Compass) for ownership, dependencies, lifecycle status
2. **Infrastructure-as-Code** (HIGH confidence)
   - Terraform/Helm/Spinnaker manifests for current capacity, autoscaling, alarms, IAM, network topology
   - bitbucket-pipelines.yml for CI shape and known flake annotations
3. **Incident & post-incident records** (MEDIUM-HIGH confidence)
   - Jira HOT/PIR tickets in the relevant project (filter by capability or service tag)
   - Root-cause analyses, action items, follow-through status
4. **Runbooks & operational docs** (MEDIUM confidence — verify against IaC; runbooks drift)
   - Confluence operational playbooks, on-call docs, escalation paths
5. **Live telemetry** (HIGH confidence when reachable — often gated), for example
   - Splunk dashboards/searches, SignalFx detectors, Databricks notebooks
6. **Org context** (LOW-MEDIUM confidence — useful for direction-of-travel, not facts)
   - Strategic blogs, OKR documents, team announcements about deprecation / migration plans

**IMPORTANT**: Many of above require MFA / interactive auth; if blocked, use `clarification tool` asking user to address the authentication issues; if unable to resolve authentication, declare the limitation explicitly and fall back to IaC + ticket-based inference

**Tools**[__must__]:
- /investigate-system <codebase_path> --docs <codebase_docs_path>

**Tools**[__optional__]:
- clarification

### Phase 2b -- System & Signals Documentation Review

[__depends on__ Phase 2]

[__requires confirmation__] After the system investigation completes, present the results to the user for review using a `confirmation` conversation tool. 
- The `summary` parameter MUST include:
  * A **data-source-transparency table** showing what was verifiable via primary source vs. inferred
  * A **gaps section** listing what could NOT be verified (e.g. "Splunk auth blocked", "TWG didn't have this repo indexed") so the user can choose to upgrade tooling before proceeding
- The `view` parameter MUST point to the generated system-and-signals-understanding documentation. Only proceed to the next Phase after the user confirms.

## Phase 3 -- Research & Proposal:

[__depends on__ Phase 2b]

Perform a holistic deep dive into the outcome codebase and system investigation, and identify concrete refactor / fix / observability opportunities, and synthesize them into a unified, ranked proposal.

For each proposed item, the synthesis MUST capture:
- **Type classification** — e.g. OPP (opportunity / new initiative), REFA (refactor), BUG (defect fix), STRA (strategic / planning-only)
- **Priority justification** — current customer pain? trajectory? time-sensitivity? blast radius? reversibility?
- **Estimate in "human engineer-weeks"** — calendar time for one full-time human software engineer end-to-end (design, code, tests, review, deploy). Ranges (e.g. "4-6 human engineer-weeks") reflect risk / unknowns. NEVER use bare "engineer-weeks" — always prefix "human" to disambiguate from AI-accelerated estimates.
- **Risk + reversibility** — low/medium/high; one-way door vs. experimental
- **Dependencies** — both code-level and Jira-level (does this block / depend on another item?)

For each opportunity surfaced, the SOP MUST record:
- **Source confidence** (primary-source live API vs. IaC vs. inferred-from-IaC-and-tickets)
- **Reproducible command** (curl recipe, JQL query, Cypher query) so the finding can be re-verified later
- **Currently painful vs. historically painful** distinction (verified via live state, not just docs)

**Tools**[__must__]:
- /research-propose <goal> --docs <reference_documentation>

### Phase 3b -- Proposal Review & Selection
[__depends on__ Phase 3; __goto__ Phase 3 __afterwards__ __wait__ 1h]

[__requires confirmation__; __must__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `review proposal tool` (DO NOT use the simpler `confirmation tool`). The tool will decide how to present the proposal to the user. The user selects which proposals to advance to Phase 4.

## Phase 4 -- Proposal Implementation
[__depends on__ Phase 3b; __branch__]

Perform proper, elegant implementation with the proposed item. 
- Re-read the issue and re-validate against current codebase main or master branch.
- For testing
  * You MUST perform local validation per the codebase's testing SOP: per-module unit tests + lint/formatter auto-fixes
  * If the user request includes submitting pull requests, and online pipeline build is avaialble, then skip heavy integration tests locally; rely on online build pipeline for heavy testing.

If the implementation inlcudes pull requests (PRs), mointor non-merged PRs to address 
- pipeline build failure
- comments
- rebase PR to latest master if PR has not been updated for 7days+

**Tools**[__must__]:
- /task <request>
  
**Tools**[__optional__]:
- /monitor --type pull_request