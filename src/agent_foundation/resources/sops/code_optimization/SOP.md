## Phase 0 -- Setup
[__initial__]

User needs to specify the workflow target code path (the codebase or sub-tree to optimize) and an optimization strategy. The workflow target path MUST be a subpath under the session root path: `{{ session_root_path }}`.

The workflow target path can be either:
- A **directory** — the workflow investigates all code within it (recommended for service-level optimization).
- A **file** — treated as the entry point for code investigation. The workflow will start from this file and explore its dependencies and surrounding codebase.

When asking for the workflow target path, use a `clarification` conversation tool with `expected_input_type: "path"` and `prefix` set to `{{ session_root_path }}`. This enables path autocomplete in the UI. The output variable MUST be named `workflow_target_path` for this conversation tool.

## Phase 1 -- Codebase Investigation
[__depends on__ Phase 0]

[__requires user input__] **STOP: Before executing any tool for this phase, you MUST use a `confirmation` conversation tool to get explicit user approval.** Do NOT invoke /understand-codebase until the user confirms. Present a summary of what will be investigated (target path, estimated scope) and let the user approve or adjust.

Once confirmed, perform in-depth analysis of the target codebase at `{{ workflow_target_path }}` to understand its design, architecture, dependencies, data flow, hotspots, and extension points. Produces structured documentation of findings with a navigable HTML documentation site.

**Tools**[__must__]:
- /understand-codebase <codebase_path>

### Phase 1b -- Codebase Documentation Review
[__depends on__ Phase 1]

[__requires user input__] After the codebase investigation completes, present the results to the user for review. Use a `confirmation` conversation tool with the `view` parameter pointing to the generated documentation. Summarize key architectural findings and invite the user to review the full documentation via the "View Documentation" button. Only proceed to the next phase after the user confirms they are satisfied.

## Phase 2 -- System & Signals Investigation
[__depends on__ Phase 1b]

[__requires user input__] **STOP: Before executing any tool for this phase, you MUST use a `confirmation` conversation tool to get explicit user approval.** Present what will be investigated and let the user approve. Once confirmed, investigate the production system and operational signals — runtime signals are required to ground prioritization in real-world impact beyond static code analysis.

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

**IMPORTANT**: Many of these require MFA / interactive auth; if blocked, use a `clarification` conversation tool asking the user to address authentication. If unable to resolve, declare the limitation explicitly and fall back to IaC + ticket-based inference.

**Tools**[__must__]:
- /investigate-system <codebase_path> --docs <codebase_docs_path>

**Tools**[__optional__]:
- clarification

### Phase 2b -- System & Signals Documentation Review
[__depends on__ Phase 2]

[__requires user input__] After the system investigation completes, present the results to the user for review. Use a `confirmation` conversation tool where:
- The `summary` parameter MUST include:
  * A **data-source-transparency table** showing what was verifiable via primary source vs. inferred
  * A **gaps section** listing what could NOT be verified (e.g. "Splunk auth blocked", "TWG didn't have this repo indexed") so the user can choose to upgrade tooling before proceeding
- The `view` parameter MUST point to the generated system-and-signals documentation. Only proceed to the next phase after the user confirms.

## Phase 3 -- Research & Proposal
[__depends on__ Phase 2b]

Perform a holistic deep dive into the codebase and system investigation outcomes, identify concrete refactor / fix / observability opportunities, and synthesize them into a unified, ranked proposal.

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
[__depends on__ Phase 3]

[__requires user input__] After the research & proposal phase completes, present the unified proposals to the user for review and selection. Use a `proposal_selection` conversation tool (or `confirmation` if proposal_selection is unavailable). The user selects which proposals to advance to Phase 4.

## Phase 4 -- Create Jira Epic & Issues
[__depends on__ Phase 3b]

Create a Jira Epic on the team's board to track all selected proposals, then create one child issue per selected proposal. This makes the proposals visible to the team and enables human approval via issue assignment.

1. **Create Epic** on the team's Jira board:
   - Summary: `[Code Optimization] <target codebase or service short name>`
   - Description: optimization scope summary, link to the research output, and list of child proposals
2. **Create child issues** (Story or Task) under the Epic — one per selected proposal:
   - Summary: `[<TYPE>-YYYYMM-NN>] <proposal title>` (e.g. `[OPP-202506-01] Add Redis caching layer`)
   - Description: structured WHY / WHAT / IMPACT / PROPOSED APPROACH / EFFORT / RISK sections drawn from the proposal
   - Priority: mapped from the proposal's rank and impact assessment
   - Labels: type classification (OPP, REFA, BUG, STRA) + relevant tags
   - **Leave Assignee unset** — human assignment serves as approval to proceed
3. **Report** the created Epic key and child issue keys to the user via a `confirmation` conversation tool.

**Tools**[__must__]:
- Jira issue creation (via MCP-Atlassian or equivalent)
- confirmation

## Phase 5 -- Monitor Epic for Human Assignment (Approval)
[__depends on__ Phase 4; __goto__ Phase 5 __if__ `epic_has_unassigned_issues` __wait__ 1h]

Poll the Jira Epic periodically to detect human approval signals. The approval mechanism is **issue assignment** — when a human assigns an issue (to themselves or to the agent), that constitutes approval to implement.

For each child issue under the Epic:
- **Newly assigned** (was unassigned, now has an assignee) → approved. Transition to `In Progress` if not already, then trigger Phase 6 for that issue.
- **Still unassigned** → awaiting human review. Leave alone.
- **Transitioned to Done/Cancelled by a human** → skip; the human handled it outside this workflow.
- **Already in flight** (previously triggered) → skip; Phase 6/6b is already handling it.

Continue polling as long as unassigned issues remain under the Epic.

**Tools**[__must__]:
- Jira issue monitoring (via MCP-Atlassian or equivalent)

## Phase 6 -- Implement & Open Pull Request
[__depends on__ Phase 5; __branch__]

For each approved (assigned) Jira issue, perform proper, elegant implementation:

1. **Re-read** the Jira issue and re-validate the proposal against the current codebase main/master branch.
2. **Implement** the proposed change:
   - Keep changes additive and backward-compatible wherever possible
   - Prefer module-local extension over editing large shared files
3. **Validate locally** per the codebase's testing SOP: per-module unit tests + lint/formatter auto-fixes. Skip heavy integration tests locally if an online build pipeline is available.
4. **Open PR** via Bitbucket API (or equivalent) with a structured description:
   - WHY / WHAT / IMPACT / TEST RESULTS / ROLLBACK / RISK sections
   - Link the Jira issue in the PR description
5. **Update Jira**: transition the issue to `In Review`, post a comment linking the PR.

If implementation is blocked (compile errors, ambiguous spec, auth failure), transition the Jira issue back to `To Do` and post a comment explaining why. Do NOT leave issues stuck in `In Progress` without a linked PR.

**Tools**[__must__]:
- /task <request>
- Jira issue transition (via MCP-Atlassian or equivalent)
- PR creation (via MCP-Bitbucket or equivalent)

### Phase 6b -- Monitor Pull Requests
[__depends on__ Phase 6; __goto__ Phase 6b __wait__ 1h]

Monitor all open PRs linked to the Epic's issues. For each open PR:

- **CI failure**: triage as genuine (fix + push follow-up commit), pre-existing flake (document + re-trigger), or infrastructure (re-trigger). Do NOT re-trigger the same pipeline more than twice without escalating.
- **Reviewer comments**: accept and fix valid issues (push follow-up commit), discuss ambiguous points, or respectfully justify with evidence.
- **Merged**: transition the linked Jira issue to `Done`, post a closing comment.
- **Declined/Superseded**: transition the linked Jira issue back to `To Do`, post a comment explaining.
- **Stale** (no updates for 7+ days): rebase to latest main, post a comment requesting review.

**Tools**[__must__]:
- /monitor --type pull_request

**Tools**[__optional__]:
- Jira issue transition (via MCP-Atlassian or equivalent)
- PR comment/update (via MCP-Bitbucket or equivalent)
