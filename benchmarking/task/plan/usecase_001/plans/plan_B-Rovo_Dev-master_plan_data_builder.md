# Atlassian-AGI Data Builder — Plan v1.0

> **Created:** 2026-06-06 by tony + Rovo Dev
> **Purpose:** Systematic, scalable extraction of project-grounded execution traces from Hello tenant for SFT/RL post-training of an enterprise-intelligence LLM.
> **Reference:** Counters the frontier-lab "expert traces + RL" thesis (OpenAI/Anthropic/DeepMind partnering with Mercor to mine enterprise execution traces); see `atlassian_packages/_plan/atlassian_data_moat_vision/` for the strategic context.
> **Scope:** This is the engineering plan for `atlassian-agi/data/src/atlassian_agi_data_builder/` — the Python package that builds the dataset that the case studies under `atlassian-agi/data/opportunity-studies/tony/` are currently authored by hand.

---

## §0. TL;DR — what this document commits to

1. **Build a 5-layer data pipeline** in `atlassian-agi/data/src/atlassian_agi_data_builder/`:
   - **L0 Discovery** — enumerate ALL Hello-tenant projects (Atlas projects, Jira projects, Compass services, Goals, Confluence spaces) into a candidate set of ~5,000-15,000 candidates
   - **L1 Rule-based health scoring** — apply 12 heuristic signals to rank candidates into a ~500-project shortlist (3 frontier-trace + 5 health-issue + 5 success-pairing categories)
   - **L2 LLM triage** — run a small-model (Haiku/Llama-8B) classifier on the L1 shortlist to confirm category + score-on-signature → ~150 winners
   - **L3 Substrate mining** — run the GORDIAN-canonical 12-file dossier extraction pipeline per winning project (deterministic Python; no LLM), producing the same shape as the 23 hand-authored cases
   - **L4 LLM enrichment** — agent fills narrative gaps, drafts frontier-counterfactual, writes honest caveats (the parts that need judgment)
2. **Target corpus size for v0 post-training:** **150-300 case studies** (honest justification in §6)
3. **Build a project-as-entity dataclass** that becomes the single source of truth across the pipeline
4. **Make every layer rerunnable, idempotent, and checkpointable** — incremental updates as Hello data evolves; no full re-runs
5. **Honest dampener:** layers L0-L3 are deterministic + cheap (~$0-10 per 500 projects); L4 LLM enrichment is the expensive layer ($500-2,000 for 150 projects with Sonnet-class model). Don't run L4 on projects that fail L2 — that's the whole point of the funnel.

---

## §1. Critical-thinking framing — why a pipeline, not more agents

### 1.1. The honest scaling problem

The 23 hand-authored case studies in `opportunity-studies/tony/` took **~50 hours of manual + AI-assisted authoring** (multiple sessions, multiple agents, ~30 hours of subagent compute). That's **~2.2 hours per case study** average. Scaling this to 150-300 cases via the same approach = **330-660 hours**. Not feasible.

But the 23 cases share a **uniform 12-file canonical shape** (post-v2.58 reconstruction). About **70% of each file is deterministically derivable** from Jira/Confluence/Bitbucket/Slack queries: the inventory tables (08-12), the trace schema (04), the timeline event log (03 §TWG-Reconstructed), the substrate artifact index (07), and the team/people inventory (02). The remaining **30% (narrative arc, frontier counterfactual, honest caveats)** requires LLM judgment.

**Insight:** if we build a deterministic pipeline that does the 70% mechanically, an LLM agent only needs to handle the 30% — and 30% × 150 cases × ~10 min agent work = **25 hours of agent time**, not 330. **An order-of-magnitude scaling unlock.**

### 1.2. Why the funnel is more critical than the miner

The user's framing was correct: *"we cannot run LLM on all projects at once"*. Hello tenant has ~5,000-15,000 projects/services across Atlas, Jira, Compass, Goals. Even at $0.01 per LLM call, scoring all = $50-150 + hours of latency. **Rule-based L1 filtering is the critical innovation** — it must reduce 5,000-15,000 to 500 candidates using only cheap-to-compute structural signals (no LLM, no embedding lookups, no per-project deep fetch). Then L2 LLM classification on 500 is cheap (~$5) and produces the ~150 winners worth full L3+L4 treatment.

### 1.3. Why pairwise success/failure framing matters for training

User's §5 insight is correct and important: **failure-only datasets teach pattern recognition but not policy**. To train an LLM to "decide what a competent project leader would do", we need pairs:
- ❌ HOT-302430 Identity Gatekeeper (Sev1 5-cascade) ↔ ✅ a similar dependency-fan-in service that DIDN'T cascade
- ❌ ZTP-Fireworks (Sev1 sandbox runtime) ↔ ✅ a similar in-flight platform launch with clean GA
- ❌ Patroni pod-label drift (architectural gap) ↔ ✅ a similar dual-source-of-truth system that was caught at design review

This converts the corpus from **classification training** (binary failed/succeeded) into **counterfactual decision training** (this trajectory vs. that trajectory). The latter is what frontier labs call "expert traces" — the *behaviors* that distinguish good outcomes from bad.

**Implication:** for every failure case in the corpus, we need ≥1 matched success case. Target ratio = **1:1 failure:success**. Current corpus is 23 failures + 0 explicit successes = badly imbalanced for RL signal.

---

## §2. Five-Layer Pipeline — Detailed Design

### L0. Discovery — Enumerate all candidates

**Goal:** produce a CSV of ALL Hello-tenant projects: `(project_id, project_type, name, status, owner, created_at, updated_at, last_activity_at)` — target ~5,000-15,000 rows.

**Sources (via existing MCP tools):**
| Source | MCP tool | Expected count | Project-type label |
|---|---|---|---|
| Atlas projects | `mcp__atlassian_project__search_projects` | ~2,000-4,000 | `atlas_project` |
| Atlas goals | `mcp__atlassian_goal__search_goals` | ~500-1,500 | `atlas_goal` |
| Jira projects | `mcp__atlassian__get_jira_projects` | ~200-500 | `jira_project` |
| Compass services/components | `mcp__compass__component_search` | ~5,000-10,000 | `compass_component` |
| Confluence spaces | `mcp__atlassian__get_confluence_spaces` | ~500-1,500 | `confluence_space` |
| Teams | `mcp__atlassian_team__search` | ~500-2,000 | `team` (not project — but owner of projects) |

**Implementation:** `src/atlassian_agi_data_builder/l0_discovery/` — one module per source, each emits rows to a parquet file `data/l0_candidates/<source>.parquet`. Idempotent: each module checkpoints by `(source, last_run_timestamp)` and only fetches new/updated.

**Honest gap:** the MCP tools paginate; some have `first` limits of 100. For full enumeration we need GraphQL pagination loops with cursor management. **Estimated wallclock for first full L0 run: 30-60 min.**

### L1. Rule-Based Health Scoring & Triage

**Goal:** reduce ~5,000-15,000 candidates to ~500 shortlist using 12 cheap signals — NO LLM, NO deep fetch.

**The 12 health signals (rule-based scorers):**

| # | Signal | Why it matters | Cheap to compute? |
|---|---|---|---|
| **HS1** | **Sev1/Sev2 HOT linkage count** | Projects with linked production incidents are UC2 (incident predictor) + UC5 (coordination) candidates | Yes — query `pir_link` field |
| **HS2** | **Days-since-update vs. expected cadence** | Stalled/abandoned projects = UC1 (project always green until red) negative-outcome candidates | Yes — `updated_at` |
| **HS3** | **Status oscillation count** (green→amber→green→red in N weeks) | The "always green until red" signature — UC1 prime | Yes — Atlas project_update history |
| **HS4** | **Scope churn ratio** (epics added/removed mid-flight ÷ initial epics) | UC1 scope-volatility signal; UC3 project-critic | Yes — Jira issue create/delete events |
| **HS5** | **Cross-team coordination depth** (distinct teams touching project ÷ depth-of-hierarchy) | UC5 horizontal coordination signal | Yes — TWG `worked_on` edges |
| **HS6** | **Dependency fan-in/fan-out** (linked projects) | UC5 dependency fragility | Yes — Atlas project dependencies API |
| **HS7** | **PR iteration count per epic** (PRs-per-ticket × revision-rounds) | UC1 PR-review-risk signal | Yes — Bitbucket via TWG |
| **HS8** | **Confluence page-version oscillation** (page edited >10x then no further edits) | Requirement-thrash signal — UC4 swing-project meme | Yes — page-version table |
| **HS9** | **Slack channel velocity spikes** (messages/day stddev > 3× baseline) | UC5 escalation-pattern signal | Yes — Slack count-by-day per channel |
| **HS10** | **Ownership change count** (assignee/lead handovers per month) | UC5 coordination fragility | Yes — Jira changelog |
| **HS11** | **Loom video density around milestones** (videos created within 7d of major status flip) | UC1 demo/escalation signal; also enriches the trace | Yes — Loom search by date+team |
| **HS12** | **Outcome label exists** (`closed`, `cancelled`, `shipped` status terminus reached) | Needed for ANY supervised training (we need ground truth) | Yes — Atlas/Jira status |

**Scoring model:**
```python
score = (
    3.0 * HS1   # heavy weight — incidents are most valuable
    + 1.0 * HS3 + 0.7 * HS4   # UC1 signals
    + 1.5 * HS5 + 1.0 * HS6 + 0.8 * HS10  # UC5 signals
    + 0.5 * HS7 + 0.3 * HS8 + 0.5 * HS9 + 0.3 * HS11  # supporting
    - 5.0 * (1 - HS12)  # heavy penalty if no outcome label
    - 3.0 * (HS2 > 365)  # heavy penalty if 1+ year stale (likely abandoned)
)
```

**Output:** `data/l1_shortlist.parquet` with all ~5,000-15,000 candidates scored, sorted descending. **Take top 500.**

**Category assignment (rule-based, multi-label):**
- `frontier_uc1_green_to_red` if HS3 ≥ 2 AND HS4 > 0.3
- `frontier_uc2_incident_cluster` if HS1 ≥ 2
- `frontier_uc3_project_critic` if HS4 > 0.5 AND HS12 = 1
- `frontier_uc4_swing_project` if HS8 ≥ 5 AND HS1 ≥ 1
- `frontier_uc5_coordination` if HS5 ≥ 4 AND HS10 ≥ 3
- `health_dependency_fragile` if HS6 ≥ 5 (NEW — see §4)
- `health_silent_drift` if HS2 between 90-365d AND HS12=0 (NEW — abandoned-without-decision)
- `health_ownership_chaos` if HS10 ≥ 5 (NEW)
- `health_scope_explosion` if HS4 > 2.0 (NEW)
- `health_post_launch_decay` if status was `shipped` then HS1 spike within 30d (NEW)

**Implementation:** `src/atlassian_agi_data_builder/l1_health/` — pure-pandas scorer over the L0 parquet.

### L2. LLM Triage Classifier

**Goal:** confirm category assignment on the 500 shortlist via small-model LLM (Haiku-class) — ~$5 total.

**Per-candidate input:** project name + 20-line description (concatenated from Atlas description + last 3 project updates + linked issue summary). Output: `(category_confirmed, confidence, frontier_signature_match)`.

**Why use LLM here:** rule-based signals can confuse "active, high-coordination, succeeding project" with "in-trouble project". LLM at this stage uses semantic signals from project description + recent updates to disambiguate.

**Pairwise pair-finder:** for each confirmed failure-case in top 150, query L0 for **structurally-similar projects with HS12=1 and zero/low HS1** — pick 1-2 success matches. This is the §5 user insight made concrete.

**Output:** `data/l2_winners.parquet` — top 150 confirmed failure cases + 150 paired success cases = **300 case candidates**.

**Implementation:** `src/atlassian_agi_data_builder/l2_triage/` — async-batch via the existing ai_gateway in the workspace (see `atlassian_packages/ai-gateway/`).

### L3. Substrate Mining (the GORDIAN-canonical 12-file builder)

**Goal:** for each of 300 winners, produce the 12-file dossier shape — fully deterministic, no LLM.

**One module per canonical file:**

| Canonical file | L3 module | Source | Output format |
|---|---|---|---|
| `02_team_and_people_inventory.md` | `l3.team_inventory` | TWG `worked_on`+`reports_to` joins; Atlas project owners; ACL members | Markdown table |
| `03_timeline_and_trace.md` | `l3.timeline_builder` | Atlas project_updates + Jira changelog + Confluence page-versions + Slack thread density | Markdown table + YAML event log |
| `04_trace_schema.yaml` | `l3.schema_emitter` | Static template populated with project anchors | YAML |
| `07_substrate_artifact_index.md` | `l3.substrate_indexer` | Cross-source rollup from 08-12 | Markdown |
| `08_jira_inventory.md` | `l3.jira_inventory` | JQL `project = X OR linkedissue = X` + custom-field extraction | Markdown table |
| `09_confluence_inventory.md` | `l3.confluence_inventory` | CQL `space = X OR ancestor = X OR contributor IN team` | Markdown table |
| `10_slack_inventory.md` | `l3.slack_inventory` | Slack `conversations.list` filtered to project name + verbatim top threads | Markdown table |
| `11_bitbucket_inventory.md` | `l3.bitbucket_inventory` | Repo discovery via Compass component → PR search by linked-issue | Markdown table |
| `12_loom_video_inventory.md` | `l3.loom_inventory` | Loom search by team + by date-near-milestones | Markdown table |

**For files requiring LLM (5 of 12) — these go to L4:**
- `01_project_arc.md` (narrative — needs judgment)
- `05_frontier_lab_counterfactual.md` (moat argument — needs judgment)
- `06_honest_caveats.md` (gaps — needs judgment)
- `README.md` (1-paragraph framing — needs judgment)
- The `### Hypothesis` / `### UC mapping` sections inside the deterministic files (needs judgment)

**Implementation:** `src/atlassian_agi_data_builder/l3_substrate/` — one module per file type. Each module emits a `dict` payload that a templater (Jinja2) renders into the canonical markdown shape.

**Estimated wallclock per project: 5-15 min** (dominated by API rate limits, not compute). 300 projects × 10 min = **50 hours sequential, ~3 hours with 16x parallelism.**

### L4. LLM Enrichment

**Goal:** fill the 5 LLM-requiring files per project using a Sonnet-class agent + the L3 deterministic output as context.

**Per-project agent prompt:**
> *You are given the L3 deterministic dossier for project X (12 files, of which 5 are stubs). Fill in: project arc narrative (≤200 words), frontier-lab counterfactual (what the labs can't see; ≤300 words), honest caveats (≤150 words), README framing (≤100 words). Ground every claim in a specific L3 file/line reference. NO new facts.*

**Cost estimate:** 300 projects × ~5K tokens prompt + ~3K tokens completion × Sonnet pricing ≈ **$300-900 total**.

**Implementation:** `src/atlassian_agi_data_builder/l4_enrichment/` — async batch agent over the L3 outputs; integrates with ai_gateway.

---

## §3. Frontier-Trace Category Targeting (responding to user §3)

For each of the 5 frontier-lab vertical/horizontal scenarios named by the user, **target distribution** in the 300-project corpus:

| Frontier scenario | Target case count | Pairing (fail:success) | Current 23-case count |
|---|---|---|---|
| **UC1 — "Always green until red"** | 60 | 30 fail + 30 success | 4 (01, 08, 09, partial 15) |
| **UC2 — Incident predictor** | 80 | 40 fail + 40 success | 7 (02, 03, 10, 11, 12, 13, 14, 19, 20, 22, 23) |
| **UC3 — Project critic / decomposition risk** | 50 | 25 fail + 25 success | 3 (04, 15, 16) |
| **UC4 — Swing project / requirement drift** | 50 | 25 fail + 25 success | 4 (05, 17, 18, partial 04) |
| **UC5 — Org coordination optimizer** | 60 | 30 fail + 30 success | 5 (06, 07, 21, 22, 23) |
| **Total** | **300** | **150 fail + 150 success** | **23 (almost all fail)** |

**Current 23-case corpus is 0% success-case coverage.** Adding success cases is the #1 expansion priority.

---

## §4. Additional Health-Issue Categories (responding to user §4 — innovation)

Beyond the 5 user-named frontier scenarios, these health signatures are worth corpus-grounding:

| New category | Why it's interesting for SFT/RL | Sample structural signature |
|---|---|---|
| **Dependency fragility cascade** | Trains the model on "what makes a service load-bearing for many others" | HS6 ≥ 5 + recent HS1 in any dependent |
| **Silent drift abandonment** | Distinguishes "killed for good reason" vs "starved without decision" — the latter is org-coordination failure | HS2 between 90-365d AND HS12=0 AND HS9 dropping |
| **Ownership chaos** | "This thing has had 4 PMs and 3 EMs in 6 months" — the team-health signal | HS10 ≥ 5 within 180 days |
| **Scope explosion** | The "started as a 2-week thing, became a 6-month thing" pattern | HS4 > 2.0 over project lifetime |
| **Post-launch decay** | Successful launch followed by silent rot in production — UC2 sibling | shipped status THEN HS1 spike within 30d |
| **Acquisition integration drag** | Acquired-company project that stalls inside parent org | tag `acquired` + HS2 increasing + HS10 ≥ 2 |
| **Compliance/regulatory pivot** | Project that pivoted hard mid-flight due to external regulation | HS4 spike correlated with external news/SOC2/GDPR ticket spike |
| **Talent attrition signal** | Project losing 2+ tech leads / staff engineers in 90 days | TWG `left_org` edges within team within 90d |
| **Toxic dependency adoption** | Project that adopted a now-deprecated/critically-vulnerable framework | Bitbucket dependency graph + Snyk/Compass advisory match |
| **Re-org casualty** | Project orphaned by a re-org (PM/EM/director departed → no clear replacement) | TWG team-membership transition + HS10 spike + HS2 increasing |

**These 10 NEW categories** would slot into the L1 scorer and expand the corpus richness beyond the 5 frontier scenarios.

---

## §5. Pairwise Success/Failure Pair Construction (responding to user §5)

User intent (paraphrased): *"For every failure case, we need a structurally-similar success case. So the model learns to distinguish trajectories, not just classify outcomes."*

### Pair-construction algorithm

For each L2 failure-confirmed project `P_fail`:
1. **Feature vector** = `[size_category, team_count, sprint_count, dependency_fan_in, lifecycle_phase_at_failure, primary_uc_category]`
2. **Candidate pool** = projects with `HS12=1` (closed/shipped) AND `HS1=0` (no Sev1/Sev2) within feature-vector cosine distance ≤ 0.2
3. **Top-2 nearest** = candidate success pairs
4. **Tie-breaker** = same time period (±6 months) and same broad domain (Identity, Jira, Atlas, etc.)

### Worked example (using current corpus)

- ❌ `22_identity_gatekeeper_service` (HOT-302430, 5-cascade Sev1) →
  - Candidate ✅ pair 1: **Atlassian Account Service** (similar 7-product fan-in, no recent Sev1 since 2025-12) — would teach the model what *prevented* the cascade
  - Candidate ✅ pair 2: **Login Service v2 migration** (similar architectural transition, completed without incident) — would teach migration-execution policy

- ❌ `03_ZTP_sandbox_incident_cluster` (Fireworks, in-flight, Sev1 during dogfood) →
  - Candidate ✅ pair 1: **OpenAPI Platform launch** (similar in-flight platform GA, clean) — would teach launch-readiness policy

- ❌ `01_GORDIAN_delivery_health` (status oscillation, mid-program) →
  - Candidate ✅ pair 1: an enterprise migration program of similar size that hit GA on time

### Implementation

`src/atlassian_agi_data_builder/pair_finder/` — pulls L0 candidates filtered by `(HS12=1, HS1=0)` and runs nearest-neighbor on the feature vector. Output: `data/pairs.parquet` with `(fail_project_id, success_project_id, similarity, common_features)`.

---

## §6. Target corpus size — honest justification (responding to user closing question)

**The user asked: "How many projects do you think needed?"**

### My honest answer: **150-300 projects for v0; 1,000-3,000 for v1; 10,000+ for v2 frontier-competitive.**

**Reasoning:**

#### v0 (150-300 cases) — supervised SFT seed corpus
- Recent literature on domain-SFT shows **~150-300 high-quality demonstrations** is the threshold where a Sonnet/GPT-4-class base model starts generalizing on narrow domain tasks (cf. LIMA paper, Tülü3, etc.)
- 150 fail + 150 success = enough for pairwise contrast learning at v0
- This is what we can realistically produce in **8-12 weeks** with the proposed pipeline
- **v0 outcome:** a fine-tuned model that can do single-project judgment ("is this project in trouble?", "what's the risk in this decomposition?")

#### v1 (1,000-3,000 cases) — RL-ready corpus with environment simulators
- For RL, we need **a distribution of trajectories**, not just point examples — typically 5-10x the SFT corpus size
- Need 5+ pair examples per sub-pattern within each UC (not just 1-2) — drives count to ~1,500
- Need cross-team multi-quarter sequences for the simulators — drives count to ~3,000
- **v1 outcome:** an RL-trained model that can recommend *interventions* (project plans, escalation triggers, dependency restructurings) with quantified expected outcomes

#### v2 (10,000+ cases) — frontier-competitive
- Anthropic's Excel-trace deal with financial institutions reportedly captured **~10K-50K execution traces**
- Frontier-competitive enterprise-intelligence training would need similar scale across project types
- At 10K+ cases, the model learns **across-industry generalization** (not just Hello-tenant patterns)
- **v2 outcome:** a model that could power a commercial "AI Project Director" product line — the actual moat

**Critical-thinking honesty check:**
- The v0 number (150-300) is the **minimum where SFT signal beats prompt engineering**
- The v1 number (1,000-3,000) is the **threshold where RL becomes the dominant training paradigm**
- The v2 number (10,000+) is the **frontier-competitive bar**
- **Going below 150 is wasted effort.** Going above 10,000 has diminishing returns until multi-org data is added.

**For this plan's v0 target: 300 cases is the right answer.** That's what the data-builder will be designed for.

---

## §7. Package architecture

```
atlassian-agi/data/src/
├── pyproject.toml                       # uv/poetry-managed; deps: pandas, pyarrow, jinja2, httpx, tenacity, structlog
├── README.md
├── _docs/_plan/
│   ├── 00_PLAN_data_builder.md          # this doc
│   ├── 01_data_schema.md                # the project-as-entity dataclass + parquet schemas (TBD)
│   ├── 02_l1_signal_definitions.md      # exact SQL/Python for each HS1-HS12 scorer (TBD)
│   ├── 03_l3_template_specs.md          # Jinja2 templates per canonical file (TBD)
│   └── 04_evaluation_plan.md            # how we validate corpus quality + SFT readiness (TBD)
└── atlassian_agi_data_builder/
    ├── __init__.py
    ├── core/                            # shared types, MCP client wrappers, caching
    │   ├── entity.py                    # Project dataclass — single source of truth
    │   ├── mcp_client.py                # thin wrappers around the existing MCP servers
    │   ├── cache.py                     # disk-backed cache (parquet + json), idempotent fetch
    │   └── checkpoint.py                # rerun-safe checkpoint manager
    ├── l0_discovery/
    │   ├── atlas_projects.py
    │   ├── jira_projects.py
    │   ├── compass_components.py
    │   ├── confluence_spaces.py
    │   └── runner.py                    # CLI: `data-builder discover`
    ├── l1_health/
    │   ├── signals/                     # one module per HS1-HS22
    │   │   ├── hs01_hot_linkage.py
    │   │   ├── hs02_staleness.py
    │   │   ├── ...
    │   ├── scorer.py                    # rule-based scoring + category assignment
    │   └── runner.py                    # CLI: `data-builder score`
    ├── l2_triage/
    │   ├── classifier.py                # async LLM triage via ai_gateway
    │   ├── pair_finder.py               # nearest-neighbor success-pair search
    │   └── runner.py                    # CLI: `data-builder triage`
    ├── l3_substrate/
    │   ├── jira_inventory.py
    │   ├── confluence_inventory.py
    │   ├── slack_inventory.py
    │   ├── bitbucket_inventory.py
    │   ├── loom_inventory.py
    │   ├── team_inventory.py
    │   ├── timeline_builder.py
    │   ├── schema_emitter.py
    │   ├── substrate_indexer.py
    │   ├── templates/                   # Jinja2 templates for each canonical file
    │   │   ├── 03_timeline_and_trace.md.j2
    │   │   ├── 08_jira_inventory.md.j2
    │   │   └── ...
    │   └── runner.py                    # CLI: `data-builder mine <project_id>`
    ├── l4_enrichment/
    │   ├── agent.py                     # the LLM agent (calls ai_gateway with the L3 dossier)
    │   ├── prompts/
    │   │   ├── project_arc.txt
    │   │   ├── frontier_counterfactual.txt
    │   │   ├── honest_caveats.txt
    │   │   └── readme_framing.txt
    │   └── runner.py                    # CLI: `data-builder enrich <project_id>`
    ├── pair_finder/                     # used by l2 + also runnable standalone for analysis
    │   └── nearest_neighbor.py
    ├── evaluation/
    │   ├── corpus_quality.py            # check: are all 12 files present? are anchors filled?
    │   ├── pair_quality.py              # check: are pairs structurally similar?
    │   └── sft_readiness.py             # check: cross-validates against the 23 hand-authored cases
    └── cli.py                           # entry point: `data-builder [discover|score|triage|mine|enrich|all]`
```

---

## §8. Phased delivery plan

| Phase | Deliverable | Duration | Validation |
|---|---|---|---|
| **P0 — Bootstrap** | `pyproject.toml`, `core/entity.py`, `core/mcp_client.py`, `core/cache.py`, CI/lint | 1 day | `data-builder --help` runs |
| **P1 — L0 Discovery** | All 5 source enumerators + `runner.py` + `l0_candidates.parquet` populated | 3-5 days | Total candidate count is 5,000-15,000 and matches manual cross-check on 10 known projects |
| **P2 — L1 Scoring** | All 12 HS scorers (frontier 5 + new health 7) + `scorer.py` + ranking output | 5-7 days | The 23 hand-authored cases all rank in top 200 of `l1_shortlist.parquet` (gold-truth recall test) |
| **P3 — L2 Triage + pair-finder** | LLM classifier + pair finder + `l2_winners.parquet` (300 candidates) | 3-5 days | Spot-check 20 random L2 picks: ≥80% correctly categorized; pairs are structurally similar by manual review |
| **P4 — L3 Substrate miners** | All 9 deterministic inventory builders + Jinja2 templates | 7-10 days | Re-run L3 against the 23 hand-authored projects → ≥85% character-overlap on the deterministic fields (compare to current files) |
| **P5 — L4 LLM enrichment** | Agent + prompts + per-project runner | 3-5 days | Generated arcs are reviewed by user; pass rate ≥70% on 10 spot-checks |
| **P6 — Full pipeline run** | All 300 projects through L0→L4 | 1 week wallclock | Corpus quality eval passes; pair quality eval passes |
| **P7 — Corpus → SFT** | Convert 300 case studies into JSONL training pairs | 2-3 days | First SFT run on a 2B-class model shows non-zero signal vs. prompt baseline |

**Total: ~6-8 weeks of focused engineering** to deliver v0 corpus + first SFT signal.

---

## §9. Honest caveats and known risks

1. **MCP rate limits.** Atlassian's MCP tools have undocumented per-tenant throttles. L0 + L3 will hit them. Mitigation: tenacity-backed retry with exponential backoff; cache-first design; build a "fetch budget" estimator.

2. **HOT incidents live on `ops.internal.atlassian.net`, not `hello.atlassian.net`.** Verified during the 23-case authoring sessions. The L3 miner needs to handle this cross-site reference correctly — `hello`-only JQL returns empty for HOT issues even though the PIR pages exist on hello.

3. **TWG access is SSAM-gated.** We hit this during the v2.55-v2.60 sessions; full event-log mining via Socrates SQL requires `twg-graph-explorer-readonly` SSAM grant. **Pipeline must work both with and without SSAM** — L3 modules should mark "TWG-blocked" gracefully and fall back to Atlassian/Slack/Loom-only enumeration (which is what the current 23 cases use).

4. **Success-case discovery is harder than failure-case discovery.** Failures have HOT tickets, PIR pages, post-mortems — explicit substrate. Successes have... silent execution. The pair finder will need creative signals: "shipped on time per Atlas target_date" + "zero linked Sev1/Sev2 in 30 days post-ship" + "team consensus stable". **Risk: we end up with 150 well-documented failures and 150 thinly-documented successes.** Mitigation: explicitly score success-case "evidence richness" and only include those with ≥50% the substrate-density of failures.

5. **L4 LLM enrichment is the most expensive component.** Budget cap = $1,500 for v0; if a single project's enrichment exceeds $5, route it to a smaller model first.

6. **Ground truth uncertainty.** "Did project X actually succeed?" is sometimes a judgment call (e.g., shipped-on-time-but-was-abandoned-after-6-months). The L1 HS12 scorer flags definite successes/failures; the gray zone goes to manual review.

7. **Privacy and access.** All data stays in Hello tenant. The data-builder must NEVER write tenant data outside `atlassian-agi/data/`. The SFT JSONL output should be reviewed for PII before any training run.

8. **The 23 existing hand-authored cases are the gold truth.** The L3 miner's output must be evaluated against them. If L3 misses anchors that the human author found, the L1 scorer probably needs additional signals or the L3 module needs additional source coverage.

---

## §10. Decision points requiring user input (before P0)

1. **Confirm v0 target: 300 projects (150 fail + 150 success)?** Or adjust?
2. **Confirm dest for L3 output:** should it write directly to `atlassian-agi/data/opportunity-studies/tony/<NN>_<slug>/` (same shape as existing 23) or to a separate `opportunity-studies/auto/` (so hand-authored vs auto-generated are separable)?
3. **Confirm we use the existing ai_gateway (`atlassian_packages/ai-gateway/`) for L2 + L4 LLM calls?** (vs. a separate Anthropic/OpenAI API key path)
4. **Confirm pyproject/uv vs poetry vs pants?** (Other repos in this workspace use a mix)
5. **Confirm CLI invocation pattern:** `data-builder ...` vs `python -m atlassian_agi_data_builder ...` vs `uv run data-builder ...`

---

## §11. Success criteria for this plan

This plan succeeds if:
- ✅ A reader who is not the author can take this plan and start building P0 within 2 hours
- ✅ Every layer has a clear, scoped, testable deliverable
- ✅ The dependency on existing systems (MCP tools, ai_gateway, TWG, SSAM) is explicit
- ✅ The cost estimates are honest (compute + dollars + wallclock)
- ✅ The v0 corpus target (300) is justified, not waved-at
- ✅ The honest caveats are not buried at the end of long sections — they're prominent

---

## §12. Change log

- **v1.0 (2026-06-06 00:57)** — Initial plan. Authored by Rovo Dev based on user §1-§6 specification. Ready for user review before P0 implementation begins.
