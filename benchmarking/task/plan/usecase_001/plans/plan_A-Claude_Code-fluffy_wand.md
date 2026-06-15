# Implementation Plan: atlassian-agi Data Builder Pipeline

## Context

The goal is to build a systematic, scalable data pipeline that extracts project-level execution traces from Atlassian's Hello tenant for post-training an enterprise-intelligence LLM. Currently, 23 case studies exist (hand-authored, ~2.2h each), but only 2 are fully fleshed out (GORDIAN and nucbc_authz). To reach the v0 SFT target of **300 case studies** (150 failure + 150 matched success), we need to automate the ~70% of each dossier that is deterministically derivable from APIs, leaving only the ~30% (narrative, counterfactual, caveats) for LLM enrichment.

An existing comprehensive design plan exists at `data/src/_docs/_plan/00_PLAN_data_builder.md` with a 5-layer architecture. The bootstrap is done: `core/entity.py` (Pydantic Project model with HealthSignals, FrontierCategory enums) and `cli.py` (typer stubs). **Everything else needs to be built.**

### Pre-existing API clients to reuse (read-only patterns):
- `CoreProjects/atlassian-packages/ai-lab-atlassian-agent/jira_client.py` (1107 lines, async httpx)
- `CoreProjects/atlassian-packages/ai-lab-atlassian-agent/confluence_client.py` (async httpx, CQL)
- `CoreProjects/atlassian-packages/ai-lab-atlassian-agent/bitbucket_client.py` (async httpx)
- `CoreProjects/atlassian-packages/ai-lab-atlassian-agent/config.py` (env-var auth pattern)

### Key architectural decisions:
1. **Vendor read-only subsets** of existing API clients (they're not pip-installable; carrying CRUD methods is unsafe)
2. **Direct REST APIs only** (MCP tools only work inside Rovo sessions; the CLI runs standalone)
3. **Anthropic SDK directly** for L2/L4 LLM calls (the planned `ai-gateway` package doesn't exist in workspace)
4. **All output** goes to `data/opportunity-studies/tony/` (extending existing directory)
5. **Two-pass L1 scoring** to avoid N+1 query explosion on 15K candidates

---

## Implementation Plan

### Phase 1: Core Infrastructure (~50 files across core/)

**Location:** `atlassian_agi_data_builder/core/`

#### 1a. Configuration (`core/config.py`)
- Unified `DataBuilderConfig` dataclass loaded from env vars
- Sub-configs: `JiraConfig`, `ConfluenceConfig`, `BitbucketConfig`, `SlackConfig`, `TwgConfig`
- Required: Jira credentials. Optional (with degradation warnings): Confluence, Bitbucket, Slack, TWG
- Separate `OPSJ_URL`/`OPSJ_API_TOKEN` for HOT tickets (they live on `ops.internal.atlassian.net`)

#### 1b. API Clients (`core/api/`)
Thin, read-only httpx wrappers modeled on existing clients. Each has:
- Tenacity retry with exponential backoff (2/4/8s, max 5 retries)
- Per-service asyncio.Semaphore (Jira: 5, Confluence: 5, Atlas/Compass: 3, Slack: 2)
- Rate-limit 429 handler (reads `Retry-After` header)
- Generic async paginator supporting both offset-based and cursor-based patterns

| File | Lines | Source API |
|---|---|---|
| `core/api/base_client.py` | ~120 | Abstract base (httpx AsyncClient, Basic Auth, _request, _paginate) |
| `core/api/jira_client.py` | ~150 | Jira REST v3 (search_projects, search_issues, get_issue_changelog) |
| `core/api/confluence_client.py` | ~100 | Confluence v2 (get_spaces, search_content, get_page_versions) |
| `core/api/atlas_client.py` | ~180 | Atlas/Townsquare REST (search_projects, search_goals, get_project_updates) |
| `core/api/compass_client.py` | ~120 | Compass REST v2 (search_components, get_dependencies) |
| `core/api/bitbucket_client.py` | ~80 | Bitbucket Cloud 2.0 (list_repos, list_pull_requests) |
| `core/api/slack_client.py` | ~80 | Slack Web API via httpx (search_messages, get_channel_history) |
| `core/api/__init__.py` | ~40 | ClientRegistry (lazy init, cached_property per client, close_all) |

#### 1c. Cache (`core/cache.py`, ~150 lines)
- Disk-backed: parquet for tabular, JSON for individual responses
- 24-hour TTL by default, configurable
- Hierarchical keys: `l0/atlas_projects`, `l1/hs01/HOT-12345`
- Storage: `data/.cache/`

#### 1d. Checkpoint (`core/checkpoint.py`, ~120 lines)
- Rerun-safe: `data/.checkpoint.json` with `{layer: {step: {status, timestamp, metadata}}}`
- `is_done(layer, step)` / `mark_done(layer, step, metadata)` / `mark_failed(layer, step, error)`
- Idempotency: if step is done, skip entirely and use cached output

#### 1e. LLM Client (`core/llm_client.py`, ~100 lines)
- **Initially a stub** with a `LLMClient` protocol/ABC that returns placeholder responses
- Real implementation swapped in during Slice 3 using `anthropic` SDK (add `anthropic>=0.25` to deps)
- Supports Haiku (L2 triage, ~$0.01/call) and Sonnet (L4 enrichment, ~$1-3/call)
- Cost tracking per call via `usage` response field
- Async with semaphore-controlled concurrency

#### 1f. Logging (`core/logging.py`, ~40 lines)
- structlog with ISO timestamps, JSON in CI / console in terminal

---

### Phase 2: L0 Discovery (~6 files)

**Location:** `atlassian_agi_data_builder/l0_discovery/`

**Goal:** Enumerate ALL Hello-tenant projects into `data/l0_candidates.parquet` (~5K-15K rows).

| File | Purpose |
|---|---|
| `base.py` | Abstract `L0Source` with `enumerate()` -> DataFrame, checkpoint-aware `run()` |
| `atlas_projects.py` | REST `/gateway/api/watermelon/` with TWG CLI fallback |
| `atlas_goals.py` | Same pattern for goals |
| `jira_projects.py` | `/rest/api/3/project/search` with full pagination |
| `compass_components.py` | `/gateway/api/compass/v2/components` (largest source, ~5K-10K) |
| `confluence_spaces.py` | `/wiki/api/v2/spaces` |
| `dedup.py` | Cross-source dedup (normalize names, group by name+team, keep richest record) |
| `runner.py` | Orchestrate all sources concurrently, merge, dedup, write parquet |

**Output columns:** `project_id` (source-prefixed: `atlas_proj:uuid`, `jira:10001`, etc.), `project_type`, `name`, `description`, `status`, `owner_aaid`, `team_id`, `created_at`, `updated_at`, `last_activity_at`, `source_url`

**Per-source parquets** under `data/.cache/l0/` for incremental re-runs.

---

### Phase 3: L1 Health Scoring (~15 files)

**Location:** `atlassian_agi_data_builder/l1_health/`

**Goal:** Score all candidates with 12 health signals and rank to a ~500 shortlist.

#### Signal tier classification (avoids N+1 explosion):
- **Tier 0 (free, from L0 data):** HS02 (staleness), HS12 (outcome label) — zero API calls
- **Tier 1 (batch queries):** HS01 (HOT linkage via single JQL), HS04 (scope churn), HS06 (dependency fan), HS10 (ownership changes) — O(1) queries
- **Tier 2 (per-candidate, expensive):** HS03, HS05, HS07, HS08, HS09, HS11 — only computed for top ~2K after Pass 1

**Two-pass approach:**
1. Pass 1: Tier 0 + Tier 1 on all candidates → preliminary score → take top 2K
2. Pass 2: Tier 2 on top 2K only → final composite score

| File | Signal | Tier | Data source |
|---|---|---|---|
| `signals/hs01_hot_linkage.py` | HOT incident count | 1 | JQL on opsj |
| `signals/hs02_staleness.py` | Days since update | 0 | L0 `updated_at` |
| `signals/hs03_status_oscillation.py` | Green→red oscillations | 2 | Atlas project update history |
| `signals/hs04_scope_churn.py` | Epic add/remove ratio | 1 | Jira changelog (batch JQL) |
| `signals/hs05_cross_team_depth.py` | Distinct teams touching project | 2 | TWG (fallback: 0) |
| `signals/hs06_dependency_fan.py` | Inbound+outbound deps | 1 | Atlas project links + Compass |
| `signals/hs07_pr_iteration.py` | Review round count | 2 | Bitbucket per repo |
| `signals/hs08_page_oscillation.py` | High-edit-then-abandon pages | 2 | Confluence page versions |
| `signals/hs09_slack_velocity.py` | Message velocity spikes | 2 | Slack channel history |
| `signals/hs10_ownership_changes.py` | Lead/assignee handovers | 1 | Jira changelog (batch) |
| `signals/hs11_loom_density.py` | Videos near milestones | 2 | Loom search or TWG |
| `signals/hs12_outcome_label.py` | Terminal status exists | 0 | L0 `status` |
| `scorer.py` | Weighted composite + category assignment | — | All signals |
| `runner.py` | Two-pass orchestrator | — | Orchestrates passes |

**Scoring formula:**
```
score = 3.0*HS1 + 1.0*HS3 + 0.7*HS4 + 1.5*HS5 + 1.0*HS6 + 0.5*HS7 + 0.3*HS8 + 0.5*HS9 + 0.8*HS10 + 0.3*HS11
       - 5.0*(1-HS12)  [no outcome label penalty]
       - 3.0*(HS2>365)  [1+ year stale penalty]
```

**Category assignment (multi-label, rule-based):**
- UC1 if HS3 >= 2 AND HS4 > 0.3
- UC2 if HS1 >= 2
- UC3 if HS4 > 0.5 AND HS12 = 1
- UC4 if HS8 >= 5 AND HS1 >= 1
- UC5 if HS5 >= 4 AND HS10 >= 3
- Plus 10 health categories (dependency_fragile, silent_drift, ownership_chaos, etc.)

**Validation:** The 23 hand-authored cases must all rank in the top 200.

**Output:** `data/l1_shortlist.parquet` (all candidates scored, sorted descending)

---

### Phase 4: L2 Triage + Pair Finder (~6 files)

**Location:** `atlassian_agi_data_builder/l2_triage/` and `pair_finder/`

#### 4a. LLM Classifier (`l2_triage/classifier.py`)
- Takes top 500 from L1 shortlist
- Per-candidate: 20-line context (name + description + last 3 updates + linked issue summary)
- Haiku-class model confirms/rejects category, outputs confidence score
- Batches of 20, asyncio.Semaphore(20), checkpointed to `data/l2_triage_checkpoint.jsonl`
- Estimated cost: ~$2-5 for 500 candidates
- Output: top 150 confirmed failures (by is_failure=True, highest confidence)

#### 4b. Pair Finder (`pair_finder/`)
- `feature_vector.py`: 6-dim vector [size_category, team_count, sprint_count, dependency_fan_in, lifecycle_phase, primary_uc_category] — min-max normalized to [0,1]
- `nearest_neighbor.py`: For each of 150 failures, find nearest success case from candidate pool (HS12=1, HS1=0) by cosine distance <= 0.2
- Tie-break: temporal proximity (within 6 months preferred), then domain match
- Constraint: each success paired with at most 2 failures (prevents over-representation)

**Output:** `data/l2_winners.parquet` (150 failures + 150 paired successes = 300)

---

### Phase 5: L3 Substrate Mining (~15 files + 10 templates)

**Location:** `atlassian_agi_data_builder/l3_substrate/`

**Goal:** For each of 300 winners, produce 9 of the 12 canonical dossier files — fully deterministic, no LLM.

#### Builder modules (one per file type):

| Module | Output file | Primary data source | Fallback |
|---|---|---|---|
| `team_inventory.py` | `02_team_and_people_inventory.md` | TWG `worked_on`+`reports_to`, Atlas owners | Jira assignees, Confluence authors |
| `timeline_builder.py` | `03_timeline_and_trace.md` | Atlas updates + Jira changelog + Confluence versions + Slack | Partial timeline from available sources |
| `schema_emitter.py` | `04_trace_schema.yaml` | Template populated from all other builders | Static template with project anchors |
| `substrate_indexer.py` | `07_substrate_artifact_index.md` | Cross-source rollup from 08-12 | — |
| `jira_inventory.py` | `08_jira_inventory.md` | JQL: `project={key}` + linked issues | Empty with honest gap |
| `confluence_inventory.py` | `09_confluence_inventory.md` | CQL: `space={key}` + ancestor traversal | Empty with honest gap |
| `slack_inventory.py` | `10_slack_inventory.md` | Slack conversations.list + search | Channels only, no content (RAI) |
| `bitbucket_inventory.py` | `11_bitbucket_inventory.md` | Compass component -> repo -> PR | Inferred from Confluence/Slack refs |
| `loom_inventory.py` | `12_loom_video_inventory.md` | Loom search + Confluence embed scan | Often empty (honest gap) |

#### Jinja2 templates (`l3_substrate/templates/`)
10 template files (`.md.j2` and `.yaml.j2`) matching the exact GORDIAN canonical format: section headers, markdown table structures, "## Intent", "## Honest gaps", "## Count Summary" in every file.

#### Execution strategy (per project):
```
Batch 1 (parallel): [08, 09, 10, 11, 12]  — independent inventory extraction
Batch 2:            [02]                   — team inventory (uses inventory data for people discovery)
Batch 3:            [03]                   — timeline (aggregates events from all inventories)
Batch 4 (parallel): [04, 07]              — schema + index (depend on everything above)
```

**Per-project time:** ~5-15 min (API-bound). 300 projects x asyncio.Semaphore(16) = ~3 hours wallclock.

**Output directory:** `data/opportunity-studies/tony/{nn}_{slug}/` (9 files per directory)

---

### Phase 6: L4 LLM Enrichment (~5 files)

**Location:** `atlassian_agi_data_builder/l4_enrichment/`

**Goal:** Fill the 5 LLM-requiring files per project using Sonnet-class agent.

#### Design:
- **Single LLM call per project** (not 5 separate calls) — cheaper, more coherent
- Full L3 dossier (9 files, ~5-8K tokens) as context
- Structured output with `===FILE_START:filename===` / `===FILE_END===` delimiters
- Per-project cost cap: $5; total budget: $900

#### Output files per project:
1. `01_project_arc.md` — narrative arc (<=200 words)
2. `05_frontier_lab_counterfactual.md` — moat argument (<=300 words)
3. `06_honest_caveats.md` — substrate gaps (<=150 words)
4. `README.md` — 1-paragraph framing (<=100 words)
5. `_hypothesis_sections.md` — hypothesis/UC-mapping for deterministic files

#### Files:
- `agent.py` — EnrichmentAgent class, context builder, cost tracker
- `parser.py` — Parse delimited LLM output into individual files, validate word counts + references
- `runner.py` — Orchestrate enrichment for all projects with checkpointing
- `prompts/enrichment_system.txt` — System prompt (grounding rules, RAI gates)
- `prompts/enrichment_user.txt` — User prompt template with dossier context + file specifications

**Concurrency:** asyncio.Semaphore(8), estimated ~6 min wallclock for 300 projects.

---

### Phase 7: Evaluation (~5 files)

**Location:** `atlassian_agi_data_builder/evaluation/`

| File | Purpose | Key metric |
|---|---|---|
| `corpus_quality.py` | File presence, anchor fill rate, size checks, section structure | All 14 files present, >200 bytes each |
| `pair_quality.py` | Structural similarity, substrate density ratio, category coherence | Success >= 50% substrate density of failure |
| `sft_readiness.py` | Cross-validate L3 against 23 hand-authored gold-truth cases | >= 85% data overlap on deterministic fields |
| `gold_truth_mapping.yaml` | Maps case study slugs to L0 project IDs | Manual construction after L0 run |
| `runner.py` | Orchestrate all checks, output reports to `evaluation/*.json` | — |

---

### Phase 8: CLI + Integration

Update `cli.py` to wire all runners to the typer commands:

```python
data-builder discover [--force]        # L0
data-builder score [--top 500]         # L1
data-builder triage [--top 300]        # L2
data-builder mine [--project-id X]     # L3 (single) or --all
data-builder enrich [--project-id X]   # L4 (single) or --all
data-builder evaluate                  # validation
data-builder all [--top 300]           # full pipeline
```

Update `pyproject.toml` with new dependencies: `anthropic>=0.25`, `numpy>=1.26`, `pyyaml>=6.0`

---

## File Manifest (~55 new files)

```
atlassian_agi_data_builder/
├── core/
│   ├── config.py                          NEW (~80 lines)
│   ├── logging.py                         NEW (~40 lines)
│   ├── cache.py                           NEW (~150 lines)
│   ├── checkpoint.py                      NEW (~120 lines)
│   ├── llm_client.py                      NEW (~100 lines)
│   └── api/
│       ├── __init__.py                    NEW (ClientRegistry, ~40 lines)
│       ├── base_client.py                 NEW (~120 lines)
│       ├── jira_client.py                 NEW (~150 lines)
│       ├── confluence_client.py           NEW (~100 lines)
│       ├── atlas_client.py                NEW (~180 lines)
│       ├── compass_client.py              NEW (~120 lines)
│       ├── bitbucket_client.py            NEW (~80 lines)
│       └── slack_client.py                NEW (~80 lines)
├── l0_discovery/
│   ├── __init__.py                        NEW
│   ├── base.py                            NEW (~60 lines)
│   ├── atlas_projects.py                  NEW (~100 lines)
│   ├── atlas_goals.py                     NEW (~80 lines)
│   ├── jira_projects.py                   NEW (~80 lines)
│   ├── compass_components.py              NEW (~100 lines)
│   ├── confluence_spaces.py               NEW (~70 lines)
│   ├── dedup.py                           NEW (~80 lines)
│   └── runner.py                          NEW (~80 lines)
├── l1_health/
│   ├── __init__.py                        NEW
│   ├── signals/
│   │   ├── __init__.py                    NEW
│   │   ├── base.py                        NEW (~50 lines)
│   │   ├── hs01_hot_linkage.py            NEW (~80 lines)
│   │   ├── hs02_staleness.py              NEW (~30 lines)
│   │   ├── hs03_status_oscillation.py     NEW (~100 lines)
│   │   ├── hs04_scope_churn.py            NEW (~90 lines)
│   │   ├── hs05_cross_team_depth.py       NEW (~70 lines)
│   │   ├── hs06_dependency_fan.py         NEW (~70 lines)
│   │   ├── hs07_pr_iteration.py           NEW (~80 lines)
│   │   ├── hs08_page_oscillation.py       NEW (~70 lines)
│   │   ├── hs09_slack_velocity.py         NEW (~90 lines)
│   │   ├── hs10_ownership_changes.py      NEW (~80 lines)
│   │   ├── hs11_loom_density.py           NEW (~60 lines)
│   │   └── hs12_outcome_label.py          NEW (~30 lines)
│   ├── scorer.py                          NEW (~120 lines)
│   └── runner.py                          NEW (~100 lines)
├── l2_triage/
│   ├── __init__.py                        NEW
│   ├── classifier.py                      NEW (~150 lines)
│   └── runner.py                          NEW (~100 lines)
├── pair_finder/
│   ├── __init__.py                        NEW
│   ├── feature_vector.py                  NEW (~100 lines)
│   └── nearest_neighbor.py                NEW (~120 lines)
├── l3_substrate/
│   ├── __init__.py                        NEW
│   ├── base.py                            NEW (~60 lines)
│   ├── jira_inventory.py                  NEW (~150 lines)
│   ├── confluence_inventory.py            NEW (~120 lines)
│   ├── slack_inventory.py                 NEW (~100 lines)
│   ├── bitbucket_inventory.py             NEW (~100 lines)
│   ├── loom_inventory.py                  NEW (~80 lines)
│   ├── team_inventory.py                  NEW (~120 lines)
│   ├── timeline_builder.py                NEW (~150 lines)
│   ├── schema_emitter.py                  NEW (~100 lines)
│   ├── substrate_indexer.py               NEW (~100 lines)
│   ├── runner.py                          NEW (~120 lines)
│   └── templates/
│       ├── 02_team_and_people_inventory.md.j2
│       ├── 03_timeline_and_trace.md.j2
│       ├── 04_trace_schema.yaml.j2
│       ├── 07_substrate_artifact_index.md.j2
│       ├── 08_jira_inventory.md.j2
│       ├── 09_confluence_inventory.md.j2
│       ├── 10_slack_inventory.md.j2
│       ├── 11_bitbucket_inventory.md.j2
│       └── 12_loom_video_inventory.md.j2
├── l4_enrichment/
│   ├── __init__.py                        NEW
│   ├── agent.py                           NEW (~150 lines)
│   ├── parser.py                          NEW (~80 lines)
│   ├── runner.py                          NEW (~100 lines)
│   └── prompts/
│       ├── enrichment_system.txt          NEW
│       └── enrichment_user.txt            NEW
├── evaluation/
│   ├── __init__.py                        NEW
│   ├── corpus_quality.py                  NEW (~120 lines)
│   ├── pair_quality.py                    NEW (~100 lines)
│   ├── sft_readiness.py                   NEW (~150 lines)
│   ├── gold_truth_mapping.yaml            NEW (manual post-L0)
│   └── runner.py                          NEW (~80 lines)
├── core/entity.py                         UPDATE (minor: add to_parquet_row helper)
├── core/__init__.py                       UPDATE (exports)
└── cli.py                                 UPDATE (wire runners)
pyproject.toml                             UPDATE (add anthropic, numpy, pyyaml deps)
```

**Estimated total: ~4,500 lines of new Python code + ~500 lines of Jinja2 templates.**

---

## Implementation Sequence: Thin Slice First

**Strategy:** Build a minimal working version of ALL 5 layers for a single project (GORDIAN), validate output matches the hand-authored exemplar, then widen to scale.

### Slice 1: Single-project vertical (GORDIAN) — Weeks 1-3

| Step | Deliverable | Validation |
|---|---|---|
| 1a | Core: config, base_client, jira_client, confluence_client, cache, checkpoint | Authenticated API calls work |
| 1b | L0: jira_projects source only (simplest) + runner | Parquet has GORDIAN project |
| 1c | L1: HS02 (staleness) + HS12 (outcome) + scorer (2 signals only) | GORDIAN scores > 0 |
| 1d | L2: **Stub** classifier (pass-through, no LLM) + manual project selection | GORDIAN selected as winner |
| 1e | L3: jira_inventory + confluence_inventory + timeline_builder (3 of 9 builders) | Compare output vs hand-authored GORDIAN files — >=70% overlap |
| 1f | L4: **Stub** enrichment (copy existing hand-authored narrative files) | Directory has all 14 files |
| 1g | Evaluation: corpus_quality on GORDIAN only | 14/14 files present, sizes reasonable |

### Slice 2: Full L0-L1 funnel + remaining L3 builders — Weeks 3-5

| Step | Deliverable | Validation |
|---|---|---|
| 2a | L0: all 5 source enumerators + dedup | l0_candidates.parquet has 5K-15K rows |
| 2b | L1: all 12 signals + two-pass scorer | 23 gold-truth cases rank in top 200 |
| 2c | L3: remaining 6 builders (slack, bitbucket, loom, team, schema, substrate index) + all Jinja2 templates | GORDIAN L3 output matches exemplar >=85% |
| 2d | Pair finder: feature_vector + nearest_neighbor | Pairs look structurally reasonable |

### Slice 3: LLM layers + scale — Weeks 5-7

| Step | Deliverable | Validation |
|---|---|---|
| 3a | L2: real LLM classifier (Haiku) replaces stub | 150 confirmed failures from 500 shortlist |
| 3b | L4: real enrichment agent (Sonnet) replaces stub | Spot-check 10 arcs; >=70% pass rate |
| 3c | Full pipeline run: `data-builder all --top 300` | 300 dossiers in tony/ |
| 3d | Evaluation: all 3 quality checks | Reports pass thresholds |

---

## Honest Caveats

1. **opsj cross-site:** HOT tickets live on `ops.internal.atlassian.net`, not `hello.atlassian.net`. Needs separate credentials.
2. **TWG is SSAM-gated:** Pipeline must degrade gracefully without TWG. HS05 (cross-team depth) and team_inventory fall back to Jira/Confluence inference.
3. **Success cases are harder:** Failures have explicit substrate (HOT tickets, PIR pages). Successes are "silent execution." Pair quality eval enforces >=50% substrate density.
4. **LLM enrichment budget:** $900 cap for 300 projects. Single-call-per-project design keeps cost at ~$1-3/project.
5. **Rate limits:** Atlassian MCP/REST APIs have undocumented throttles. Cache-first design + tenacity retry mitigate.
6. **Atlas REST API shape:** The Townsquare REST endpoints are internal and may change. Atlas client needs resilient fallback to TWG CLI.
7. **Ground truth is only 23 cases (2 exemplar, 21 stubs).** SFT readiness validation is limited by gold-truth coverage.

---

## Verification

After implementation:
1. Run `data-builder all --top 300` end-to-end
2. Verify `data/opportunity-studies/tony/` has 300+ directories, each with 14 files
3. Run `data-builder evaluate` — all three reports should pass thresholds
4. Spot-check 10 random dossiers against hand-authored exemplars for quality
5. Verify total LLM cost stayed under $900
6. Verify the 23 existing STUB case studies were expanded to EXEMPLAR format
