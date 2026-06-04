# SOP Creation from Runbook

Convert a free-form operational runbook (typically a Confluence page or markdown
runbook) into a well-architected, automation-ready SOP that follows the
`agent_foundation` SOP convention. Replaces ad-hoc copy-paste-and-reformat with a
disciplined, live-verified process that produces SOPs ready for repeat execution.

[__keywords__] sop creation, runbook to sop, automate runbook, sop from confluence, sop authoring, standardize runbook
[__example_requests__]
- turn this Confluence runbook into a SOP: <url>
- automate the runbook at <url>
- convert this manual process into a reusable SOP
- create an SOP from the page at <url>

## Reference example
A real end-to-end application of this SOP can be reviewed at:
- **Source runbook**: https://hello.atlassian.net/wiki/spaces/~71202088d1945544364701b4921ae3c07480c4/pages/7049101435
- **Resulting SOP**: `CoreProjects/OpenStartup/src/openteam/server/resources/sops/atlassian/search/SOP.md`
- **Resolved-issues archive**: same directory, under `issues/resolved/`
- **Orchestrator implementation**: `CoreProjects/xtenant_refresh_automation/`

The example shows what "good" looks like for this SOP's output, including the
Verified Reference table pattern (Phase 4 below) and the resolved-vs-live issues
split (Phase 8 below).

## Anti-patterns this SOP exists to prevent
[__important__]:

1. **Trusting CLI defaults without region/scope verification** — e.g. `get-blueprint -n <name>`
   silently returns one variant when several exist. Always verify.
2. **Claiming a step "works" because the dry-run succeeded** — dry-runs validate
   syntax, not authorization. Always exercise the real surface (live `get`
   against the real env, even if no write).
3. **Mixing resolved and live limitations in "Known Issues"** — degrades signal.
   Always split.
4. **Hard-coding service identifiers without traceability** — every UUID / job-id /
   table-name in the SOP MUST cite its source (runbook URL, registry, or
   verified CLI output).
5. **Writing a phase-by-phase SOP without a dependency graph** — leads to false
   sequencing claims. Make parallelism explicit.

## Phase 0 -- Setup:
[__initial__]

Collect the two inputs required to begin runbook conversion:

1. **`runbook_url`** — the source URL (Confluence page URL, markdown file path,
   or other addressable runbook). MUST be a single canonical source; if the user
   has multiple runbooks for the same process, ask which is authoritative.
2. **`target_sop_path`** — the destination directory where `SOP.md` (and
   optional `sop.config.json`) will be written. By convention this is one of:
   - `{{ session_root_path }}/sops/<slug>/` for project-local SOPs
   - `<agent_foundation_root>/src/agent_foundation/resources/sops/<slug>/` for
     framework-level SOPs (one level deep so the loader discovers them)
   - `<openstartup_root>/src/openteam/server/resources/sops/<group>/<slug>/` for
     organizational grouping (note: loader sees only one level deep — see Phase 7)

When asking for `runbook_url`, use a `clarification` conversation tool with
`expected_input_type: "url"`. The output variable MUST be named `runbook_url`.

When asking for `target_sop_path`, use a `clarification` conversation tool with
`expected_input_type: "path"` and `prefix` set to `{{ session_root_path }}`.
The output variable MUST be named `target_sop_path`.

If the user wants the SOP discoverable by the framework SOP loader, warn them
about the one-level-deep constraint (see Phase 7 for details) and offer the
framework-level destination as the default.

## Phase 1 -- Runbook ingestion and entity extraction:
[__depends on__ Phase 0]

[__requires user input__] **STOP: Before fetching the runbook, you MUST use a
`confirmation` conversation tool to get explicit user approval.** Present the
`runbook_url` and the extraction plan (below) and let the user approve or
redirect (e.g. "actually start from this related page first").

Once confirmed, fetch the runbook content and perform structured extraction.

**Fetch tool**[__must__] choose ONE based on URL type:
- Confluence page → `mcp__atlassian__get_confluence_page` (saves HTML to a local file
  so you can grep/expand without re-fetching)
- Local markdown → `open_files`
- Other web URL → if no fetcher is available, surface this honestly to the user
  and stop; do NOT guess content from the URL alone

**Entities to extract** (write to `{{ session_root_path }}/.sop_creation/entities.json`):

| Entity type | Examples | Why it matters |
|---|---|---|
| **URLs** | View Run links, dashboard links, PR links, source-table links | Used in Verified Reference table (Phase 4) and per-phase Tools sections |
| **Workflow / job identifiers** | Blueprint UUIDs, Databricks job IDs, MLflow experiment IDs, registry names | These are the primary verification targets in Phase 2 |
| **Code / data identifiers** | Repo + PR numbers, table fully-qualified names, S3 paths, DBFS URIs | Permission-gated; must be probed in Phase 2 |
| **Human / team references** | Owner handles, channel names, group names, on-call rotations | Captured in Maintainers section of output SOP |
| **Cardinality hints** | "4 regions", "3 environments", "per-tenant" | Surfaces sharded/parallel work that the runbook may treat as serial |
| **Implicit dependencies** | "After X succeeds, do Y" / "while X runs, also Y" | Used to construct Phase 3's dependency graph |
| **Cycle metadata** | Last-updated date, last-run timestamp, version numbers, change-log entries | Helps distinguish "current state" from "historical artefact" |

**Output of this phase**: `entities.json` plus a short Markdown summary listing
each entity category with counts. Present the summary to the user with a
`confirmation` tool before proceeding — they will spot omissions you missed.

## Phase 2 -- Live automation-surface probe:
[__depends on__ Phase 1]

[__requires user input__] **STOP: Before running any probe, present the probe
plan via a `confirmation` conversation tool.** Probes are READ-ONLY by
contract, but the user must confirm scope (which environments, which
identifiers, which CLI versions).

For each automation surface implicated by the entities (Phase 1), run the
**minimal read-only command** that proves authorization works **end-to-end**
in the target environment. Record `OK` / `FAIL <error>` per surface.

Common probe patterns:

| Surface | Probe command | What success proves |
|---|---|---|
| ML Studio workflow blueprint | `atlas ml workflow get-blueprint -n <name> -u <usecase> -t ADHOC -e prod` | SSAM read on that usecase |
| Databricks job | `databricks jobs get <job_id>` | At least `CAN_VIEW` on that job |
| Databricks SQL table | `SELECT COUNT(*) FROM <fqn>` via Statements API | SELECT grant on table |
| ML Registry version | `atlas ml registry version list -r <component>` | Registry read on component |
| Bitbucket repo | `mcp__bitbucket__invoke_tool bitbucketRepository action=get` | Read access to repo |
| Confluence page | `mcp__atlassian__invoke_tool get_confluence_page` | Confluence read |
| POCO-gated APIs (any) | The first failing API + `atlas poco logs get -s <service>` | Surfaces the canonical SSAM group required |

**Rules**:
- A `--dry-run` flag does NOT count as a probe — it bypasses auth.
- A probe that returns a non-empty JSON body counts as `OK`.
- If a probe returns 403 / unauthorized, capture the `decisionId` (POCO services
  emit this) and add it to the gap report.
- If a probe times out or DNS-fails, capture both the hostname and the error;
  some hosts are corporate-network-gated and may resolve for the user but not
  for an automated session.

**Output**: `{{ session_root_path }}/.sop_creation/probe_report.md` with one row
per surface. **Present to user** before proceeding — if any blocker rows exist,
the user decides whether to (a) request access first, (b) document the gap and
proceed with manual fallback, or (c) abort.

## Phase 3 -- Phase decomposition and dependency graph:
[__depends on__ Phase 2]

Translate the runbook's narrative into a phase-by-phase decomposition. Each
output phase MUST have:

- A short, action-oriented heading: `## Phase N -- <verb-led action>:`
- A dependency tag: `[__depends on__ Phase X]` (or `[__initial__]` for Phase 0)
- A parallelism note when applicable: `[__parallel with__ Phase Y]`
- A one-sentence purpose statement
- An explicit "Tools" subsection if any CLI/MCP/API call is involved
- A "Verification" subsection (how do you know the phase succeeded?)
- A "Failure modes" subsection (what can go wrong?)

**Heuristics for boundary detection**:
- A new phase begins when the runbook switches **principal** (different tool,
  different team, different system) OR introduces a **wait gate** (must wait for
  prior phase to complete).
- A "phase" that is purely human judgment (e.g. "visually verify the chart
  looks reasonable") still becomes a phase, marked `[__requires user input__]`,
  because it gates downstream execution.
- Steps the runbook lists as "optional sanity checks" generally do NOT become
  full phases; capture them as Verification within the surrounding phase
  (exception: if the sanity check has a substantive cost — e.g. a separate
  training run — promote it to a phase, e.g. "Phase 1 baseline training").

**Dependency graph artefact**: produce a Mermaid graph in
`{{ session_root_path }}/.sop_creation/dependency_graph.md`. Present to the
user with a `confirmation` conversation tool before proceeding to Phase 4 — the
user will catch missing parallelism or false-serial claims.

## Phase 4 -- Verified Reference table construction:
[__depends on__ Phase 3]

This is the single most important defensive artefact in the resulting SOP.
For every identifier that the runbook treats as "the obvious one" (blueprint
ID, job ID, table name, registry component, etc.), the Verified Reference
table records:

| column | purpose |
|---|---|
| Phase | The phase that consumes this identifier |
| Surface | The CLI / API that consumes it (e.g. `atlas ml workflow run`) |
| Identifier kind | e.g. "blueprint UUID", "job ID", "table FQN" |
| Identifier value | The literal value, in monospace |
| Region / shard | If applicable (e.g. `us_west_2`) — many surfaces require per-region IDs |
| Verified at | Date of last live probe (UTC ISO date) |
| Verification cmd | The exact command used to verify |

**Why this table is mandatory**: The reference example's most expensive bug
(2026-06-02 single-region eval mistake) came from assuming a name-keyed
identifier was region-agnostic. The Verified Reference table forces every
identifier to be region-tagged, which makes the assumption visible at SOP
authoring time, not at execution time.

Place this table near the top of the output SOP, immediately after
Prerequisites. Sort rows by phase, then by region.

## Phase 5 -- SOP draft assembly:
[__depends on__ Phase 4]

Assemble the draft SOP at `{{ target_sop_path }}/SOP.md` following the
canonical structure:

```
# <Title>
<one-paragraph purpose statement>

[__keywords__] <comma-separated>
[__example_requests__]
- <how users will phrase the trigger>

## Reference
- <source runbook URL>
- <prior cycle URLs if any>
- <related code / scripts>

## Prerequisites
[__must__]:
- <hard-required accesses, CLIs, MCPs>

## Verified Reference  (from Phase 4)

## Phase 0 -- Setup
...
## Phase N -- <action>:
[__depends on__ Phase ...]
...

## Known limitations
> Resolved issues are archived in `issues/resolved/`.

1. <limitation> -- <workaround> -- <path to close the gap, if any>

## End-to-end automation script
[__see__] <path-to-orchestrator-script-if-any>

## Cost & timing reference
- Wall-clock: ~<duration>
- Compute: ~$<estimate>
- Human time: ~<minutes>

## Maintainers
- <handle>: <role>
```

**Also write `sop.config.json`** in the same directory if the SOP is intended
to be loaded by the framework SOP loader (i.e. one level under a known SOP
base directory). Required fields: `name`, `display_name`, `version`,
`description`. Strongly recommended: `available_modes`, `requires_tools`,
`labels`.

## Phase 6 -- Self-audit pass:
[__depends on__ Phase 5]

Before declaring the SOP done, run a critical-thinking checklist against the
draft. Each item below must produce an explicit pass/fail finding; do not
hand-wave.

1. **Every identifier in the SOP appears in the Verified Reference table** — grep
   the draft for UUIDs / job IDs / table names and confirm 100% coverage.
2. **No phase claims a step "automates" something without a Phase 2 probe row** —
   if you claim `databricks jobs run-now <id>` works, the probe report must
   contain `OK databricks jobs get <id>` for the same id.
3. **No phase says "depends on X" if X is upstream of a different branch** —
   walk the dependency graph and assert each `[__depends on__]` matches.
4. **Every "manual" step has a documented path to automation** (or an explicit
   "intentionally manual" justification with reason).
5. **Permission gaps are listed in Known limitations with the exact request URL
   or Slack channel** for closing the gap.
6. **No marketing language** — phrases like "fully automated", "production
   grade", "ready to ship" are red flags. Either prove them (link to a
   completed dry-run) or remove them.
7. **No "TODO" / "TBD" / "FIXME" left in the body** — these must be either
   resolved, escalated to Known limitations, or filed as separate tickets.

**Output**: `{{ session_root_path }}/.sop_creation/self_audit_report.md` with
one row per check + finding + (if fail) the line numbers in the draft. Present
to user via a `confirmation` conversation tool — they decide whether to fix
findings, escalate them to Known limitations, or accept them.

## Phase 7 -- Loader-discoverability check:
[__depends on__ Phase 6]

The framework `load_all_sops` loader is **one-level-deep** under each
configured base directory. It iterates `base_dir.iterdir()` and looks for
`child / "SOP.md"`. Implications:

- `agent_foundation/.../sops/<slug>/SOP.md` → ✅ loaded as SOP `<slug>`
- `agent_foundation/.../sops/<group>/<slug>/SOP.md` → ❌ NOT loaded (too deep)

If the user wants the new SOP runtime-discoverable, the placement chosen in
Phase 0 MUST be one level deep relative to a known SOP base. Verify by:

1. Reading the loader configuration to enumerate `base_dirs` (sources include
   `agent_foundation/resources/sops/`, plus any `extra_dirs` registered by the
   active project, e.g. `openteam/server/resources/sops/`).
2. Confirming `<chosen_base>/<slug>/SOP.md` resolves; if the user chose a
   deeper path for human-organizational reasons, surface this honestly and
   offer three options:
   - **Promote** — flatten to a top-level slug for discoverability (sacrifices
     hierarchy)
   - **Stay nested** — accept that the SOP is human-only, not loader-loaded
   - **Patch the loader** — extend `load_all_sops` to walk recursively (touches
     `agent_foundation` core; out of scope for most cycles)

Document the chosen placement and rationale in the SOP's Reference section.

## Phase 8 -- Issue-tracking infrastructure:
[__depends on__ Phase 7; __optional__]

If the runbook has any historical issues worth preserving (e.g. the resolved
single-region eval mistake in the reference example), create:

```
{{ target_sop_path }}/issues/
└── resolved/
    └── <YYYY-MM-DD>-<short-slug>.md
```

Each resolved-issue file MUST include:

- Date discovered + cycle context
- Severity
- What happened (the bug or surprise)
- How it surfaced (the detection path — credit the user critical-thinking
  challenge if applicable, since this is the most common detection vector)
- Root cause
- Resolution (what changed)
- Prevention going forward (where in the SOP the guard now lives)
- Why filed as resolved (the explicit claim that the bug cannot recur)

Add a one-line pointer at the top of the SOP's "Known limitations" section:
`> Resolved issues are archived in [\`issues/resolved/\`](issues/resolved/).`

For currently-live issues (not yet resolved), use `issues/open/<slug>.md` with
the same template minus the "Resolution" / "Why filed as resolved" sections,
plus an "Owner" and "Target resolution date" field.

## Phase 9 -- Reference orchestrator implementation:
[__depends on__ Phase 6; __optional__]

If the SOP is end-to-end automatable (most phases have CLI/MCP/API surfaces),
generate a reference orchestrator script alongside the SOP. Conventions:

- Path: `{{ session_root_path }}/<sop-slug>_automation/` (sibling to the SOP, not
  inside it — keeps the SOP directory pure documentation)
- Entry point: `<sop-slug>_orchestrator.py`
- Must support `--dry-run` (prints what would be executed without doing it)
  and `--execute` (the real run)
- Must NOT bake credentials into the script — read from the user's existing
  CLI configs (e.g. `~/.databrickscfg`, `atlas auth`)
- Must print every command before executing so the run is auditable
- Phase boundaries in the script MUST match phase boundaries in the SOP, with
  the same names. A divergence between SOP phase names and script phase
  names is a bug.

The reference example's orchestrator at
`CoreProjects/xtenant_refresh_automation/refresh_xtenant_orchestrator.py`
demonstrates these conventions.

## Cost & timing reference
[__info__]:
- **Wall-clock to produce a SOP**: ~1-2 hours for a runbook of moderate size
  (~30 entities, ~6-10 phases)
- **Live-probe time**: ~10-20 minutes (Phase 2) — dominates if many auth
  surfaces need exercise
- **Self-audit pass**: ~10-15 minutes (Phase 6) — pure thinking, no compute
- **Optional orchestrator generation**: ~30 minutes additional (Phase 9)

## Maintainers
- **This SOP**: maintained by the `agent_foundation` core team
- **Reference example**: see the maintainers in
  `CoreProjects/OpenStartup/src/openteam/server/resources/sops/atlassian/search/SOP.md`
