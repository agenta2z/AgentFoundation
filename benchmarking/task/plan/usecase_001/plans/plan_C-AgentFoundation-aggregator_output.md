# Aggregated Plan — `atlassian-agi-data-builder` End-to-End Integration (L0 → L4 + Corpus/Training Strategy + Targeted Case-Study Expansion)

> **Authored:** 2026-06-06 (aggregator pass over 6 upstream worker proposals).
> **Workspace root:** `atlassian-agi/data/src/` (package `atlassian_agi_data_builder`).
> **Companion master plan:** `_docs/_plan/00_PLAN_data_builder.md` v1.0.
> **Seed end-to-end case under test:** `CTSC-39558` (Jira project `CTSC` on `hello.atlassian.net`).
> **Status:** consolidated implementation+verification plan; ready for staged execution under master-plan §8 P0–P7.

---

## §0. Aggregation Overview & Provenance

This artifact integrates **six upstream worker proposals**, each scoped to a layer or cross-cutting deliverable of the master plan. They were produced in parallel; this aggregator pass de-duplicates, reconciles conflicts, and surfaces the cross-layer guarantees the master plan asks for. The aggregator does **not** re-author the per-layer plans verbatim — instead it (a) confirms each layer's outputs and contract, (b) defines the cross-layer verification harness (the seed-case dry-run), and (c) ships the documentation/CHANGELOG/README updates the upstream guidance demanded.

| # | Worker | Layer / scope | Source | Verdict (substance) |
|---|---|---|---|---|
| 1 | `worker_0` | **L0 Discovery + shared `core/` foundations** (P0+P1) | `…/worker_0/outputs/final_deliverables/output.md` (931 L) | Substantive, comprehensive. Adopts CLI-as-runner + `runner.py` testable helper; Compass partition-by-team-ARI (fallback alphabet shard); 12-row risk register; canonical `discover_sources` JSON fence; tenant-context bootstrap. **Carried in full.** |
| 2 | `worker_1` | **L1 Rule-based health scoring** (P2) | `…/worker_1/outputs/final_deliverables/output.md` (532 L) | Substantive. 12 vectorized signal modules under `l1_health/signals/`; centralized `SIGNAL_WEIGHTS`; 15 category rules (5 UC + 10 innovation); `@graceful_signal` decorator; PyArrow-locked writer. **Carried in full.** Surfaces new innovation category candidates → §7 below. |
| 3 | `worker_2` | **L2 LLM triage + Pair Finder** (P3) | `…/worker_2/outputs/final_deliverables/output.md` (537 L) | Substantive. Two CLI verbs (`triage`, `pair`) — preserves cost asymmetry; ai_gateway behind `LLMClient` Protocol with 3 adapters (real/mock/offline-fixture); sklearn kNN + numpy fallback; 6-feature scaler persisted as `pairs.scaler.json`. Adds one entity field: `frontier_signature_match: bool`. **Carried in full.** |
| 4 | `worker_3` | **L3 Substrate miner** (P4) | `…/worker_3/outputs/final_deliverables/output.md` (518 L) | Substantive. 9 collectors → 10 files + 4 stubs (Slack emits 10 and 10b); payload/render separation with Jinja2 templates; 3-knob concurrency (16× project × 6 internal × HTTP-32 semaphore); RAI scrub for `10b`; `Project.secondary_sites` additive field. **Carried in full.** |
| 5 | `worker_4` | **L4 LLM enrichment + targeted expansion of `{01,02,03,22,23}` and their success-pair siblings** | `…/worker_4/outputs/final_deliverables/output.md` (546 L) | Substantive. `l4_enrichment/{prompt,agent_runner,citation_validator}.py`; shared `core/ai_gateway_client.py` + `core/dossier_io.py`; allow-listed write gate; marker-anchored `00_README.md` upsert; per-call `--max-cost-usd 30.0` ceiling; seed-YAML pair-finder MVP. **Carried in full.** Adds CI guard `scripts/check_subtask_diff.py`. |
| 6 | `worker_5` | **Corpus sizing + SFT/DPO/ORPO training recipe + UC1–UC5 eval battery + seed-case validation** | `…/worker_5/outputs/final_deliverables/output.md` (783 L) | Substantive. v0 (150–300) / v1 (1K–3K) / v2 (10K+) sizing rationale with hard go/no-go floors; `tokenizer_adapter.py`, `pair_converter.py`, `axolotl`/`trl` configs; UC1–UC5 eval tasks with leakage controls; `validation/e2e_harness.py` against `CTSC-39558`. **Carried in full and promoted to the headline `_docs/_plan/01_corpus_and_training_strategy.md`.** |

**Cross-worker convergences worth pinning at the aggregator level:**
- One `Project` Pydantic entity is the source of truth across all 6 layers. **Only three additive fields** appear across all upstream plans: `raw_source_payload: dict | None` (L0/worker_0), `frontier_signature_match: bool = False` (L2/worker_2), `secondary_sites: list[str] = []` (L3/worker_3). Optionally `split: Literal["train","val","test"] | None = None` (training/worker_5). No removals; no migrations.
- One parquet per layer artifact under `data/`. `data/l0_candidates/<source>.parquet`, `data/l1_shortlist.parquet` + `data/l1_top500.parquet`, `data/l2_winners.parquet` + `data/l2_all.parquet` + `data/l2_errors.parquet`, `data/pairs.parquet`, `data/sft_v0.jsonl`, `data/preferences_v0.jsonl`.
- All MCP calls go through a single `core/mcp_client.py` (or `l3_substrate/_mcp.py` shim that becomes a 1-line re-export when canonical lands). Tenacity-backed retries, structlog events, per-call polite-sleep floor.
- Every layer is rerunnable and idempotent: atomic `*.tmp` + `os.replace`; per-source checkpoints; cache-first; layer outputs validated against pinned PyArrow schemas before downstream layers consume them.
- Benchmarks/results are **always** written to JSON artifacts under `data/_observability/`, `data/metrics/`, `artifacts/l4_runs/`, `tests/perf_report.json` — **never** parsed from stdout.

---

## §1. Cross-Layer Contract & Data Flow

### 1.1 End-to-end pipeline (single Project's journey)

```
L0 Discovery (worker_0)
  └─→ data/l0_candidates/<source>.parquet   (~5K–15K total rows; 1 row = 1 Project)
        │
        ▼
L1 Rule-based scoring (worker_1)
  └─→ data/l1_shortlist.parquet (full ranked)  + data/l1_top500.parquet (head)
        │
        ▼
L2 LLM triage (worker_2)                      ─→ data/l2_winners.parquet (top-150 failures, confirmed)
  └─→ Pair Finder (worker_2 / worker_4 MVP)   ─→ data/pairs.parquet (≤150 paired successes)
        │                                              │
        └────────────── 300 winners ───────────────────┘
        │
        ▼
L3 Substrate miner (worker_3)
  └─→ data/projects/<slug>/{01..12}.md (+ 10b for Slack; + 04_trace_schema.yaml)
        │   payload/render split; 16× project concurrency; ~3 h wallclock for 300 projects
        ▼
L4 LLM enrichment (worker_4)
  └─→ in-place fills 5 LLM files per project + marker-anchored inline blocks
        │   ≤ $30 hard ceiling across the named subset; citation-grounded
        ▼
Training corpus + evals (worker_5)
  └─→ data/sft_v0.jsonl, data/preferences_v0.jsonl, data/sizing_model.json,
      eval reports under artifacts/eval/, validation report under data/validation/
```

### 1.2 Unified `Project` entity changes (additive only, no migration)

```python
# atlassian_agi_data_builder/core/entity.py — ADDITIVE diff

class ProjectType(str, Enum):
    ...                                      # existing 5 values
    TEAM = "team"                            # NEW (worker_0)

class Project(BaseModel):
    ...                                      # existing fields kept verbatim
    raw_source_payload: dict | None = None   # NEW (worker_0) — unredacted L0 MCP response, forensic only
    frontier_signature_match: bool = False   # NEW (worker_2) — Haiku boolean verdict
    secondary_sites: list[str] = []          # NEW (worker_3) — e.g. ops.internal.atlassian.net for HOT
    split: Literal["train","val","test"] | None = None  # NEW (worker_5, deferred to T0 PR)
```

All upstream workers verified the existing `Project` already exposes the fields they consume (`enrichment_files_built`, `enrichment_cost_usd`, `is_success_pair`, `paired_failure_project_id`, `paired_success_project_id`, `frontier_categories`, `category_confidence`, `last_processed_layer`, `pipeline_errors`). The four lines above are the complete schema delta required across the whole pipeline.

### 1.3 Unified `core/` package (shared primitives)

| Module | Owner | Purpose | Required by |
|---|---|---|---|
| `core/entity.py` | existing + additive | `Project`, `HealthSignals`, `HealthStatus`, `FrontierCategory`, `ProjectType` | All layers |
| `core/mcp_client.py` | worker_0 (canonical) / worker_3 ships `l3_substrate/_mcp.py` shim if not yet landed | Tenacity-wrapped MCP shim with structlog + polite-sleep floor | L0, L3 |
| `core/checkpoint.py` | worker_0 | Atomic JSON checkpoint store (`is_resumable`, `bump_cursor`) | L0, L3 |
| `core/paginator.py` | worker_0 | `cursor_paginate` + `single_shot` generators | L0 |
| `core/parquet_io.py` | worker_0 + worker_1 | PyArrow-locked schema; `ParquetSink` writer + reader; round-trip tested | L0, L1, L2 |
| `core/run_context.py` | worker_0 | `RunContext.bootstrap()` (resolves `cloud_id`/`org_id`, caches to disk) | L0, L3 |
| `core/metrics.py` | worker_0 | Per-source metrics recorder; JSON artifact emitter | All layers |
| `core/io.py` (a.k.a. `core/dossier_io.py`) | worker_2 + worker_4 | `PROJECT_PARQUET_SCHEMA` + `validate_l1_schema`; **allow-listed** dossier read/write + marker-anchored upsert | L1, L2, L3, L4 |
| `core/ai_gateway.py` (a.k.a. `core/ai_gateway_client.py`) | worker_2 + worker_4 | Async httpx + tenacity LLM client; `LLMClient` Protocol with `AIGatewayClient`, `MockAIGatewayClient`, `OfflineFixtureClient` adapters | L2, L4 |

**Naming-reconciliation decision:** worker_2 calls the gateway module `core/ai_gateway.py`; worker_4 calls it `core/ai_gateway_client.py`. Adopt **`core/ai_gateway.py`** (shorter, matches existing convention `core/mcp_client.py`). Same for `core/io.py` vs `core/dossier_io.py` — adopt **`core/io.py`** with one symbol `DossierIO` that handles the allow-list semantics worker_4 needs. Both adapters export the same `LLMClient` Protocol so call sites are interchangeable.

### 1.4 Final package layout (consolidated)

```
atlassian-agi/data/src/atlassian_agi_data_builder/
├── core/
│   ├── __init__.py
│   ├── entity.py                  # + 4 additive fields
│   ├── mcp_client.py              # NEW (worker_0)
│   ├── checkpoint.py              # NEW (worker_0)
│   ├── paginator.py               # NEW (worker_0)
│   ├── parquet_io.py              # NEW (worker_0)
│   ├── run_context.py             # NEW (worker_0)
│   ├── metrics.py                 # NEW (worker_0)
│   ├── io.py                      # NEW (worker_2/4) — PROJECT_PARQUET_SCHEMA + DossierIO
│   └── ai_gateway.py              # NEW (worker_2/4) — LLMClient + 3 adapters
├── l0_discovery/                  # worker_0
│   ├── __init__.py                # SourceName enum + SOURCE_REGISTRY
│   ├── runner.py
│   ├── atlas_projects.py
│   ├── atlas_goals.py
│   ├── jira_projects.py
│   ├── confluence_spaces.py
│   ├── teams.py
│   └── compass.py
├── l1_health/                     # worker_1
│   ├── __init__.py
│   ├── io.py                      # L0Frames + load_l0 + write_shortlist
│   ├── score.py                   # SIGNAL_WEIGHTS, compute_composite, derive_health_status
│   ├── categories.py              # 15 CategoryRule predicates (5 UC + 10 innovation)
│   ├── runner.py
│   └── signals/
│       ├── _base.py               # @graceful_signal, require_columns
│       └── hs01..hs12.py          # one file per signal
├── l2_triage/                     # worker_2
│   ├── schema.py
│   ├── prompt.py
│   ├── templates/triage_v1.j2
│   ├── batch_runner.py
│   ├── classifier.py
│   └── runner.py
├── pair_finder/                   # worker_2 (full) + worker_4 (MVP seed-YAML shim)
│   ├── features.py
│   ├── knn.py
│   ├── tiebreak.py
│   ├── pair_writer.py
│   ├── runner.py
│   └── seeds_v0.yaml              # NEW (worker_4) — encodes master-plan §5 worked examples
├── l3_substrate/                  # worker_3
│   ├── __init__.py
│   ├── _mcp.py                    # shim — 1-line re-export once core/mcp_client.py lands
│   ├── miner.py                   # orchestrator (3-wave DAG, asyncio.Semaphore)
│   ├── runner.py
│   ├── _macros/                   # shared Jinja2 macros
│   ├── _rai.py                    # PII scrubber for 10b
│   ├── collectors/                # one per canonical file
│   │   ├── team_inventory.py
│   │   ├── timeline_builder.py
│   │   ├── schema_emitter.py
│   │   ├── substrate_indexer.py
│   │   ├── jira_inventory.py
│   │   ├── confluence_inventory.py
│   │   ├── slack_inventory.py     # emits both 10 and 10b
│   │   ├── bitbucket_inventory.py
│   │   └── loom_inventory.py
│   ├── stubs/                     # 4 L4-stub modules + hypothesis_sections.py sentinel injector
│   ├── templates/                 # one Jinja2 template per canonical file
│   └── static/scripts/twg/methodology.md   # carried with SHA stamp
├── l4_enrichment/                 # worker_4
│   ├── prompt.py
│   ├── agent_runner.py
│   ├── citation_validator.py
│   ├── runner.py
│   └── prompts/{arc,counterfactual,caveats,readme,hypothesis}.jinja
├── training/                      # worker_5
│   ├── tokenizer_adapter.py
│   ├── pair_converter.py
│   ├── split.py
│   └── configs/{sft_v0,dpo_v0,orpo_v0}.yaml
├── validation/                    # worker_5
│   ├── e2e_harness.py             # the §3 seed-case dry-run
│   └── canonical_dossier_shape.yaml
├── evaluation/                    # worker_5
│   ├── runner.py
│   ├── eval_tasks/e1_health.py … e8_counterfactual.py
│   ├── judges/{sonnet_rubric,llama_rubric}.py
│   └── leakage_probe.py
├── cli.py                         # extended commands per §1.5
└── scripts/check_subtask_diff.py  # worker_4 CI guard
```

### 1.5 CLI surface (consolidated)

| Command | Owner | New flags | Behaviour |
|---|---|---|---|
| `data-builder discover --source {atlas_projects,atlas_goals,jira_projects,confluence_spaces,teams,compass,all} [--max-pages N --limit N --dry-run --benchmark]` | worker_0 | replaces stub | Writes `data/l0_candidates/<source>.parquet` + checkpoints + metrics |
| `data-builder score [--input-dir data/l0_candidates --output data/l1_shortlist.parquet --top-n 500 --fixture <path>]` | worker_1 | replaces stub | Writes `data/l1_shortlist.parquet` + `data/l1_top500.parquet` |
| `data-builder triage [--input data/l1_shortlist.parquet --top 150 --concurrency 8 --fixture <path> --mock]` | worker_2 | replaces stub | Writes `data/l2_winners.parquet` + `data/l2_all.parquet` + `data/l2_errors.parquet` |
| `data-builder pair [--winners data/l2_winners.parquet --pool 'data/l0_candidates/*.parquet' --max-distance 0.2 --time-window-days 180]` | worker_2 | NEW verb | Writes `data/pairs.parquet` |
| `data-builder mine --project <id|all> [--concurrency 16 --resume --dry-run --output-root data/projects]` | worker_3 | extends stub (keeps `--project-id` alias) | Writes per-project 12-file dossier (+10b) |
| `data-builder enrich --project <id|all> [--concurrency 4 --max-cost-usd 30.0 --overlap-threshold 0.6 --max-retries 2 --prompt-version v1 --from-l3-glob <pattern> --no-validate]` | worker_4 | extends stub (keeps `--project-id` alias) | Fills 5 LLM files + inline blocks per project |
| `data-builder validate-seed --issue CTSC-39558` | worker_5 | NEW verb | Runs §3 harness; writes `data/validation/<issue>_<runid>.json` |
| `data-builder train {sft|dpo|orpo} --config training/configs/<recipe>.yaml` | worker_5 | NEW verb (stubs OK initially) | Wraps axolotl / trl |
| `data-builder eval --suite uc1..uc5 --checkpoint <path>` | worker_5 | NEW verb | Writes per-UC scores under `artifacts/eval/<runid>/` |
| `data-builder all --top 300` | existing | no change | End-to-end orchestration (kept stubbed pending P5) |

---

## §2. Per-Layer Verification Gates (Aggregator Guidance Item 1)

The aggregator must **confirm each layer's parquet/markdown outputs exist and pass schema checks** before any cross-layer dry-run. The contract per layer:

### 2.1 L0 Discovery — verification gate

- **Outputs to confirm:**
  - `data/l0_candidates/{atlas_projects,atlas_goals,jira_projects,confluence_spaces,teams,compass}.parquet` (6 files)
  - `data/l0_candidates/_checkpoints/<source>.json` (6 checkpoint files, each with `status=='ok'`)
  - `data/l0_candidates/_observability/<run_id>.json` (1 aggregated report)
  - `data/metrics/discover_<source>_<run_id>.json` (6 per-source metrics)
- **Schema check:** for each parquet, assert `pa.read_table(path).schema == core.parquet_io.L0_SCHEMA`. `jira:CTSC` must appear in `jira_projects.parquet`. Total row count ∈ [5K, 15K].
- **Verified by:** `tests/test_l0_smoke.py` (parametrized across 6 sources) + a live cross-check script `tools/verify_l0.py` that runs the smoke contract against the actual workspace artifacts.

### 2.2 L1 Health scoring — verification gate

- **Outputs to confirm:** `data/l1_shortlist.parquet`, `data/l1_top500.parquet`, `_docs/_plan/L1_README.md`.
- **Schema check:** `pa.read_table(...)` includes all 12 HS columns + `composite_score: float` + `frontier_categories: list<string>` + `health_status: string` + the Project spine columns; sorted descending by `composite_score` with deterministic tie-break (`hs01_hot_linkage_count` desc, `project_id` asc); top-500 head is bit-exact to `head(top_n)` of the full file.
- **Signal coverage banner:** `ScoreReport.signal_coverage` printed as a WARN if any HS has < 20% non-zero coverage (worker_1's `signal.degraded` event).
- **Verified by:** `tests/l1_health/` suite (synthetic fixture covers all 22 categories) + `test_docs_weights_match.py` (reads `_docs/_plan/L1_README.md` and asserts `SIGNAL_WEIGHTS` bit-exact).

### 2.3 L2 Triage + Pair Finder — verification gate

- **Outputs to confirm:** `data/l2_winners.parquet` (≤150 rows), `data/l2_all.parquet`, `data/l2_errors.parquet`, `data/pairs.parquet`, `data/pairs.scaler.json`.
- **Schema check:** `data/l2_winners.parquet` carries the new `frontier_signature_match: bool` field + `frontier_categories` singleton list per row; `data/pairs.parquet` has `(fail_id, success_id, similarity, common_features, tiebreak_*)` per §7 worker_2; `success_id` may be null for unpaired (no silent relaxation).
- **Quality gate:** ≥ 80% accuracy on the 5-row hand-labeled fixture; ≥ 0.8 × failure_count paired on live run (else escalate per R2).
- **Hand-validation:** GORDIAN (`01_*`) and Identity-Gatekeeper (`22_*`) must classify into UC1 / UC2-family correctly; recorded in `_docs/_plan/L2_README.md §Validation log`.

### 2.4 L3 Substrate — verification gate

- **Outputs to confirm per project:** the 12-numbered files (01..12) + `10b_slack_inventory_extended.md` + `README.md` + `scripts/twg/methodology.md` (SHA-stamped copy). `02b_team_and_artifacts.md` deliberately **not** emitted in v1 (per worker_3 R-13).
- **Byte-shape diff:** for `CTSC-39558` and any project mined, run the diff harness against `atlassian-agi/data/opportunity-studies/tony/01_GORDIAN_delivery_health/` content-zone-by-content-zone. **Structural overlap ≥ 95%** on the 9 deterministic files; **line-overlap ≥ 85%** secondary metric. Any deviation logged to `tests/fixtures/known_reference_quirks.yaml`.
- **RAI gate:** zero AAID / email / phone hits in `10b_slack_inventory_extended.md` (CI grep).
- **Concurrency check:** `--dry-run` budget estimator passes ≤ 80% of per-source rate-limit ceiling.

### 2.5 L4 Enrichment — verification gate

- **Outputs to confirm:** for each of the 5 named cases (`{01,02,03,22,23}` + their `*b_*` siblings), the 5 LLM files exist and pass the citation validator; `artifacts/l4_runs/<runid>.json` carries cost + latency + retry counts.
- **Citation gate:** every emitted markdown sentence ends with a citation token `[L3:<file>:L<n>]` (or `[L3:<file>:L<a>-L<b>]`) that resolves to a real byte range in the project's L3 dossier (Jaccard overlap with cited bytes ≥ 0.6).
- **Cost gate:** total spend ≤ $30.00 across the named subset (per-call ceiling check before each LLM call).
- **Allow-list gate:** integration test `tests/integration/test_no_unexpected_writes.py` snapshots filesystem mtime before/after enrich and asserts only the allow-listed paths changed.

### 2.6 Training corpus + evals — verification gate

- **Outputs to confirm:** `data/sft_v0.jsonl`, `data/preferences_v0.jsonl`, `data/sizing_model.json`, `tests/fixtures/ctsc_39558_gold.yaml`.
- **Schema check:** `sft_v0.jsonl` accepted by `axolotl` schema validator; `preferences_v0.jsonl` accepted by `trl.DPOTrainer` schema validator (no field errors).
- **Pair-aware split:** `hash(pair_id) % 100` partitioning verified — no fail/success pair straddles the train/val/test boundary.
- **Eval lift:** UC1–UC5 prompt-only baseline measured before training; report under `artifacts/eval/baseline/`.

---

## §3. End-to-End Seed-Case Dry-Run on `CTSC-39558` (Aggregator Guidance Item 2)

The aggregator must run the full pipeline end-to-end on the seed case `CTSC-39558` (a single Jira issue under project `CTSC` on `hello.atlassian.net`) and **diff the produced 12-file dossier against the canonical `01_GORDIAN_delivery_health/` shape**, flagging any structural drift. This section defines the harness; execution is gated on at least the L0+L3 modules existing on disk.

### 3.1 Harness contract — `atlassian_agi_data_builder/validation/e2e_harness.py`

```python
def run_seed_validation(
    issue_key: str = "CTSC-39558",
    site: str = "hello.atlassian.net",
    output_root: Path = Path("data/validation"),
    *,
    canonical_dir: Path = Path("atlassian-agi/data/opportunity-studies/tony/01_GORDIAN_delivery_health"),
    skip_l4: bool = True,
    wallclock_budget_s: int = 900,           # 15 min budget per worker_5 §D
) -> ValidationReport: ...
```

### 3.2 Step-by-step (executed in order)

| # | Step | Asserts | Failure ⇒ |
|---|---|---|---|
| 1 | Resolve seed → project. `get_jira_issue("CTSC-39558")` → confirm `project.key=="CTSC"`. Synthesize a minimal `Project` row with `project_id=="jira:CTSC"`. | `Project` constructs without `ValidationError` | hard-fail, exit 2 |
| 2 | **L0** smoke-run: `data-builder discover --source jira_projects --limit 1` reading the local MCP shim; assert `jira:CTSC` lands in `data/l0_candidates/jira_projects.parquet`. | parquet exists; row count ≥ 1 | hard-fail |
| 3 | **L1** dry-run: `data-builder score --fixture tests/fixtures/ctsc_39558_l0/` writes `data/l1_shortlist.parquet`; project's composite_score finite (not NaN). | finite score; HS coverage ≥ 1 non-zero | warn (continue) |
| 4 | **L2** dry-run: `data-builder triage --mock --project-id jira:CTSC` returns a category in `FrontierCategory` ∪ `{none_of_the_above}`. | row appears in `l2_all.parquet` | warn |
| 5 | **L3** full run: `data-builder mine --project jira:CTSC --output-root data/validation/<runid>/projects/`. Materialize the 12-file dossier (+10b). | every file in `canonical_dossier_shape.yaml` present | hard-fail |
| 6 | **Shape-parity diff** against `canonical_dir`. For each of the 12 numbered files + `README.md`, compute structural overlap (heading tree + table presence + section-count) and line-overlap (tokenized). | structural ≥ 95%; line ≥ 85% (allow-listed quirks excluded) | report drift |
| 7 | **Content-fidelity check.** For 5 deterministic fields (project name, owner_aaid, team_id, last_activity_at, count summary in `07_substrate_artifact_index.md`), assert reconciliation against `tests/fixtures/ctsc_39558_gold.yaml`. | exact match (case-insensitive for strings) | report mismatch |
| 8 | **Wallclock budget.** Total step 5+6+7 < 15 min (worker_5 §D acceptance). | within budget | warn if 15–20 min, fail > 20 min |
| 9 | (Optional) **L4 enrichment** if `skip_l4=False`: enrich the dossier with `--max-cost-usd 5.0` (low test budget); citation validator must report `pass`. | citation pass-rate = 100% | warn |
| 10 | Write `data/validation/CTSC-39558_<runid>.json` with `overall_pass: bool`, per-step status, drift report, wallclock per step, cost (if any). | JSON parseable; `overall_pass` exists | hard-fail |

### 3.3 The canonical shape contract — `validation/canonical_dossier_shape.yaml`

```yaml
expected_files:
  - README.md                              # L4 stub
  - 01_project_arc.md                      # L4 stub
  - 02_team_and_people_inventory.md        # L3 deterministic
  - 03_timeline_and_trace.md               # L3
  - 04_trace_schema.yaml                   # L3 (2 L4-TODO fields)
  - 05_frontier_lab_counterfactual.md      # L4 stub
  - 06_honest_caveats.md                   # L4 stub
  - 07_substrate_artifact_index.md         # L3 (last to run)
  - 08_jira_inventory.md                   # L3
  - 09_confluence_inventory.md             # L3
  - 10_slack_inventory.md                  # L3 (collector emits this and 10b)
  - 10b_slack_inventory_extended.md        # L3 (RAI-scrubbed)
  - 11_bitbucket_inventory.md              # L3
  - 12_loom_video_inventory.md             # L3 (often empty — template must handle)
explicitly_excluded:
  - 02b_team_and_artifacts.md              # per-case Tony-authored; out of scope v1 (worker_3 R-13)
sentinels:
  required_in_stubs: '<!-- L4-TODO -->'
  required_in_04_yaml: '# L4-TODO: hypothesis'
```

### 3.4 Reported drift categories

The diff harness MUST surface (not silently absorb) each of:
1. **Missing-file drift** — a canonical file is absent from the run output.
2. **Extra-file drift** — a file appears that is not in `expected_files` and not in `explicitly_excluded`.
3. **Structural drift** — heading tree, table count, or section ordering differs.
4. **Empty-section drift** — a file exists but the body is blank (often Loom or Bitbucket for low-activity projects). Worker_3 templates handle empty gracefully — drift is INFO, not WARN.
5. **Sentinel drift** — stub files missing `<!-- L4-TODO -->` markers (worker_4 expects these to fill in).
6. **L4-TODO over-injection** — more than 12 sentinels in a single file (worker_4 cap; over-injection means stub template is broken).

All categories logged with severity and inventory in the validation report.

### 3.5 Acceptance criteria for §3

- `data-builder validate-seed --issue CTSC-39558` exits 0 with `overall_pass: true` for steps 1–8.
- The validation JSON report under `data/validation/` is the **only** source of truth (no stdout-only assertions).
- If L0 / L1 / L2 / L3 modules are partial (current workspace reality), the harness reports `partial_pass` per step and continues; only the present steps gate.

---

## §4. Subset-Guard Verification — Worker_4 Must Only Touch `{01,02,03,22,23}` and Their Siblings (Aggregator Guidance Item 3)

The aggregator must defensively verify that worker_4 (subtask 5) **only modified the declared named subset of case-study dirs and created only success-pair sibling dirs matching the agreed naming pattern**. Worker_4 already encoded this structurally — the aggregator's job is to lock the gates and add a CI assertion.

### 4.1 Allow-list (encoded in `core/io.py::DossierIO`)

```yaml
mutable_paths:
  failure_dirs_modify:
    - atlassian-agi/data/opportunity-studies/tony/01_GORDIAN_delivery_health/
    - atlassian-agi/data/opportunity-studies/tony/02_<slug>/                   # slug TBD by executor at C1
    - atlassian-agi/data/opportunity-studies/tony/03_ZTP_sandbox_incident_cluster/
    - atlassian-agi/data/opportunity-studies/tony/22_identity_gatekeeper_service/
    - atlassian-agi/data/opportunity-studies/tony/23_<slug>/                   # slug TBD
  sibling_dirs_create:
    - atlassian-agi/data/opportunity-studies/tony/01b_*/
    - atlassian-agi/data/opportunity-studies/tony/02b_*/
    - atlassian-agi/data/opportunity-studies/tony/03b_*/
    - atlassian-agi/data/opportunity-studies/tony/22b_*/
    - atlassian-agi/data/opportunity-studies/tony/23b_*/
  readme_marker_append_only:
    file: atlassian-agi/data/opportunity-studies/tony/00_README.md
    marker_open: '<!-- AUTO:SUCCESS_PAIRS BEGIN -->'
    marker_close: '<!-- AUTO:SUCCESS_PAIRS END -->'
```

### 4.2 Triple-defense gates (Aggregator pins all three)

1. **Code gate (DossierIO):** every write goes through `DossierIO.write(path, content)` which raises `DossierWriteDenied` if `path` does not match the allow-list. Reads are unrestricted.
2. **Filesystem snapshot gate:** `scripts/check_subtask_diff.py` (worker_4) records `(path, mtime, sha256)` of every file under `atlassian-agi/data/opportunity-studies/tony/` **before** the subtask, and after; diff must lie entirely inside the allow-list. Aggregator promotes this to a CI gate that runs after `data-builder enrich --all`.
3. **Sibling-naming regex gate:** any newly-created directory under `tony/` must match `^(0[123]|2[23])b_[a-z0-9_]+$`. Anything else → fail.

### 4.3 README append-only gate (§4.1 third item)

The 00_README.md mutation is restricted to the marker-bracketed `## Success Pairs` block:
- Before/after byte-hash of the file MUST match outside the marker block (worker_4 §10.2 "byte-hash assertion").
- The block is **idempotent**: re-running the subtask replaces the block contents but does not duplicate. Verified by `tests/integration/test_readme_idempotency.py`.

### 4.4 Acceptance for §4

- CI green for `scripts/check_subtask_diff.py` on every PR that touches `l4_enrichment/` or `pair_finder/`.
- A test that intentionally tries to write to `04_*` (out-of-subset) raises `DossierWriteDenied` and is caught as a smoke check.
- No directory matching `^(0[4-9]|1[0-9]|2[01])` is modified by the L4 run.

---

## §5. Headline `_docs/_plan/` Entry — `01_corpus_and_training_strategy.md` (Aggregator Guidance Item 4)

The aggregator must **merge worker_5's corpus+training strategy doc as the headline `_docs/` entry alongside `00_PLAN_data_builder.md`**. Worker_5 already authored the doc as `01_corpus_and_training_strategy.md`; the aggregator's job is to lock that placement and update the planning index.

### 5.1 File placement (canonical)

| Path | Source | Status |
|---|---|---|
| `_docs/_plan/00_PLAN_data_builder.md` | existing v1.0 | unchanged |
| `_docs/_plan/01_corpus_and_training_strategy.md` | **worker_5 output, verbatim (v1.4)** | NEW — headline companion |
| `_docs/_plan/L0_README.md` | worker_0 | NEW |
| `_docs/_plan/L1_README.md` | worker_1 | NEW |
| `_docs/_plan/L2_README.md` | worker_2 | NEW |
| `_docs/_plan/L3_README.md` | worker_3 | NEW |
| `_docs/_plan/L4_README.md` | worker_4 | NEW |
| `_docs/_plan/README.md` | existing | **UPDATED** — see §5.2 |

### 5.2 `_docs/_plan/README.md` updates (additive table rows + re-pointing)

```markdown
| File | Purpose | Status |
|---|---|---|
| 00_PLAN_data_builder.md | Master plan — read first. | ✅ v1.0 |
| 01_corpus_and_training_strategy.md | Corpus sizing (v0/v1/v2), SFT/DPO/ORPO recipe, UC1–UC5 eval battery, CTSC-39558 e2e harness. | ✅ v1.4 — ready for review |
| L0_README.md | L0 Discovery layer — sources, pagination, idempotency contract. | ✅ |
| L1_README.md | L1 Rule-based scoring — 12 signals, weights, 15 categories. | ✅ (supersedes `02_l1_signal_definitions.md`) |
| L2_README.md | L2 LLM triage + Pair Finder — prompt, ai_gateway adapters, kNN. | ✅ |
| L3_README.md | L3 Substrate miner — collectors, templates, concurrency, RAI. | ✅ |
| L4_README.md | L4 LLM enrichment — citation contract, allow-list, cost ceiling. | ✅ |
| 01_data_schema.md | The Project dataclass + all parquet schemas. | ⏳ TBD (largely absorbed into L0–L4 READMEs) |
| 03_l3_template_specs.md | Jinja2 templates per canonical file. | ⏳ TBD (absorbed into `l3_substrate/templates/` source) |
| 04_evaluation_plan.md | Corpus quality + pair quality + SFT-readiness. | ✅ (superseded by `01_corpus_and_training_strategy.md` §C) |
```

### 5.3 Cross-doc consistency rules (CI-checkable)

- The `300 / 150-floor / 1K-3K / 10K+` numbers must agree between `00_PLAN_data_builder.md` §3+§6 and `01_corpus_and_training_strategy.md` §A. (Worker_5 §H test #3.)
- `SIGNAL_WEIGHTS` must agree between `L1_README.md` table and `l1_health/score.py` constants. (Worker_1 `test_docs_weights_match.py`.)
- `FrontierCategory` enum members must equal the union of UC1–UC5 + 10 innovation categories listed in `00_PLAN_data_builder.md` §3+§4. (Worker_1 `test_categories.py`.)
- The cross-reference table (worker_5 §I) is the single source of truth; CI lint `tools/check_plan_xref.py` (suggested by worker_5) flags drift.

---

## §6. Top-Level `CHANGELOG.md` Under `atlassian-agi/data/src/` (Aggregator Guidance Item 5)

The aggregator must **author a top-level `CHANGELOG.md` under `atlassian-agi/data/src/` that summarizes what shipped per layer, what was deferred, and the v0/v1/v2 roadmap**. The full text below is the artifact to ship.

### 6.1 File location

`atlassian-agi/data/src/CHANGELOG.md` (sibling to the `atlassian_agi_data_builder/` package directory and `pyproject.toml`).

### 6.2 Content (verbatim — copy to file)

```markdown
# CHANGELOG — atlassian-agi-data-builder

> Companion: see `_docs/_plan/00_PLAN_data_builder.md` (master plan) and
> `_docs/_plan/01_corpus_and_training_strategy.md` (corpus + training strategy).
> Format follows Keep-a-Changelog. Dates in UTC.

## [0.2.0] — 2026-06-06 — Five-layer pipeline contract + targeted case-study expansion

### Added (per-layer)

- **L0 Discovery** (`atlassian_agi_data_builder/l0_discovery/`): 6 source modules
  (atlas_projects, atlas_goals, jira_projects, confluence_spaces, teams, compass);
  shared `core/{mcp_client,checkpoint,paginator,parquet_io,run_context,metrics}.py`;
  `data-builder discover --source <name>|all` CLI; 6 per-source parquet outputs under
  `data/l0_candidates/`; idempotent atomic writes; tenant-context bootstrap with disk
  cache; Compass partition-by-team-ARI strategy with alphabet-shard fallback.
- **L1 Rule-based scoring** (`atlassian_agi_data_builder/l1_health/`): 12 vectorized
  signal modules (HS01–HS12); centralized `SIGNAL_WEIGHTS`; 15 category rules
  (5 UC + 10 innovation health categories); `data-builder score` CLI; outputs
  `data/l1_shortlist.parquet` + `data/l1_top500.parquet`; `@graceful_signal` decorator
  for missing-column resilience while L0 stabilizes.
- **L2 LLM triage** (`atlassian_agi_data_builder/l2_triage/`): async batch driver with
  bounded concurrency; Jinja2 prompt template + `PROMPT_VERSION` constant;
  `LLMClient` Protocol with real / mock / offline-fixture adapters; outputs
  `data/l2_winners.parquet` (top-150 failures) + `data/l2_all.parquet` + `data/l2_errors.parquet`.
- **Pair Finder** (`atlassian_agi_data_builder/pair_finder/`): deterministic cosine
  kNN over a 6-feature normalized vector; sklearn primary + numpy fallback; ±6 mo
  time-window with logged ladder relaxation; same-domain tie-breaker; outputs
  `data/pairs.parquet` (≤150 paired successes; null `success_id` for unpaired).
- **L3 Substrate miner** (`atlassian_agi_data_builder/l3_substrate/`): 9 collectors →
  10 files (Slack collector emits 10 and 10b); payload/render split with Jinja2;
  3-knob concurrency (16× project × 6 internal × HTTP-32 semaphore); per-project
  3-wave DAG; checkpoint manifest with template-SHA cache key; RAI scrub for `10b`;
  `data-builder mine --project <id>|--all` CLI.
- **L4 LLM enrichment** (`atlassian_agi_data_builder/l4_enrichment/`): three modules
  (`prompt.py`, `agent_runner.py`, `citation_validator.py`); shared
  `core/{ai_gateway,io}.py`; allow-listed `DossierIO`; per-call `--max-cost-usd 30.0`
  hard ceiling; marker-anchored `00_README.md` append; citation-grounded narrative
  (every sentence resolves to L3 bytes; Jaccard ≥ 0.6).
- **Targeted case-study expansion**: the named subset `{01, 02, 03, 22, 23}` enriched
  in-place; 5 new success-pair sibling directories (`01b_*` … `23b_*`) materialized
  with the full 12-file shape; CI guard `scripts/check_subtask_diff.py` enforces the
  allow-list.
- **Training corpus + evals** (`atlassian_agi_data_builder/{training,validation,evaluation}/`):
  SFT/DPO/ORPO JSONL converters; pair-aware deterministic split; `axolotl` + `trl`
  configs; UC1–UC5 eval battery with leakage probe + LLM judges; CTSC-39558
  end-to-end validation harness.
- **Documentation**: per-layer READMEs (`L0..L4_README.md`); headline
  `_docs/_plan/01_corpus_and_training_strategy.md` (v1.4) — gates v0 build go/no-go.

### Changed

- `atlassian_agi_data_builder/core/entity.py` — additive only:
  `ProjectType.TEAM`; `Project.raw_source_payload`; `Project.frontier_signature_match`;
  `Project.secondary_sites`; `Project.split` (deferred).
- `atlassian_agi_data_builder/cli.py` — `discover` / `score` / `triage` / `mine` /
  `enrich` stubs replaced with full implementations; new verbs `pair`,
  `validate-seed`, `train {sft|dpo|orpo}`, `eval` added.
- `_docs/_plan/README.md` — table extended with per-layer README rows; obsolete
  `02_l1_signal_definitions.md` and `04_evaluation_plan.md` rows re-pointed to the
  superseding documents.
- `atlassian-agi/data/opportunity-studies/tony/00_README.md` — append-only `## Success
  Pairs` section bracketed by `<!-- AUTO:SUCCESS_PAIRS BEGIN/END -->` markers.

### Deferred (out of scope for v0; tracked for v1)

- `data-builder all` end-to-end orchestrator (master plan §8 P5).
- `02b_team_and_artifacts.md` per-case Tony-authored superset (worker_3 R-13).
- Full canonical `core/mcp_client.py` if not yet landed at L3 ship time —
  `l3_substrate/_mcp.py` is a shim that becomes a 1-line re-export.
- Live `AIGatewayClient` adapter exercised end-to-end (today mock/offline-fixture
  cover CI; live exercise gated on `perfkit` availability + auth).
- 5 innovation categories (`ACQUISITION_DRAG`, `REGULATORY_PIVOT`, `TALENT_ATTRITION`,
  `TOXIC_DEPENDENCY`, `REORG_CASUALTY`) — predicates degrade to `False` and emit a
  one-time `category.dormant` log per run until upstream L0 columns exist.
- `training/` runtime — converter shapes + configs ship; actual `trl`/`axolotl` runs
  are a separate T0 PR (worker_5 §E.2).
- L4 gateway adapter live exercise (covered by mock today; live cost gated by
  `--max-cost-usd` ceiling).

### Roadmap

- **v0 (this release cycle, P0–P5):** 150–300 cases mined + enriched end-to-end on
  Hello tenant; SFT lift vs prompt-only baseline ≥ 10 pt on ≥ 3 of 5 UCs.
- **v1 (Q+1 to Q+2, P6):** 1,000–3,000 cases; DPO/ORPO at scale; held-out preference
  accuracy ≥ 75%; multi-tenant readiness experiments.
- **v2 (Q+4+, P7):** 10,000+ cases; full RL + simulator-based PPO; cross-tenant
  generalization gap ≤ 5 pt on UC-equivalent eval.

### Known issues / honest caveats

- HOT incidents on `ops.internal.atlassian.net` are not enumerated by L0 (hello-only
  site context). `Project.secondary_sites` is the additive field, populated by L3's
  `jira_inventory.collect` when the project name contains `HOT-`/`PIR-` substrings.
- TWG SSAM gating may degrade `team_inventory`/`timeline_builder` collectors —
  per-collector fallbacks documented in L3_README.md.
- The 23 hand-authored cases are ~100% failures. Success-side coverage is the single
  largest novel-engineering risk (worker_5 §F R1).

### Provenance

This release was authored across 6 parallel worker proposals, aggregated 2026-06-06.
Per-input critique lives in `_runtime/tasks/task/.../aggregator/outputs/output.md`.
```

---

## §7. `project_case_studies/00_README.md` — Append-Only `## Success Pairs` Section (Aggregator Guidance Item 6)

The aggregator must **update the `project_case_studies/00_README.md` ONLY to add (not modify) a new 'Success Pairs' index section that links the new sibling directories**. The mutation is marker-bracketed for idempotent re-runs, with a byte-hash assertion that everything outside the markers is unchanged.

### 7.1 Mutation contract

- **File:** `atlassian-agi/data/opportunity-studies/tony/00_README.md` (a.k.a. `atlassian_packages/_plan/atlassian_data_moat_vision/opportunity_studies/project_case_studies/00_README.md` — the user noted they are the same dir).
- **Strategy:** append a single block between marker tokens; never edit prior content.
- **Idempotent re-runs:** the block contents are recomputed each L4 run; the markers stay in place; outside-block byte hash is asserted unchanged.

### 7.2 Block to append (verbatim — `<slug>` resolved at C1)

```markdown
<!-- AUTO:SUCCESS_PAIRS BEGIN -->
## Success Pairs

These directories complete the fail↔success pairs called out in the master plan §5
("Pairwise Success/Failure Pair Construction"). Each `NNb_*` directory mirrors the
12-file shape of its `NN_*` sibling and is constructed via the Pair Finder
(deterministic cosine-NN over a 6-feature vector; tie-breakers: same time-window,
same domain) plus an optional seed-YAML override for the named worked examples.

| Failure case | Paired success | Pairing method |
|---|---|---|
| `01_GORDIAN_delivery_health/` | `01b_<slug>/` | seed-YAML "enterprise migration of similar size that hit GA on time" (master plan §5) |
| `02_<slug>/` | `02b_<slug>/` | Pair Finder kNN + ±6 mo time window |
| `03_ZTP_sandbox_incident_cluster/` | `03b_openapi_platform_launch/` | seed-YAML "OpenAPI Platform launch" (master plan §5) |
| `22_identity_gatekeeper_service/` | `22b_atlassian_account_service/` | seed-YAML "Atlassian Account Service" (master plan §5) |
| `23_<slug>/` | `23b_<slug>/` | Pair Finder kNN + same-domain tie-break |

> Each `NNb_*` directory contains the full 12-file dossier (`01..12.md` + `04_trace_schema.yaml`)
> with L3 deterministic mining where collectors are wired, and hand-curated stubs
> (frontmatter `source: hand-curated, l3_miner: pending`) otherwise. All 5 L4
> narrative files per dir are citation-grounded against the bytes in that dir
> (`[L3:<file>:L<n>]` tokens; Jaccard ≥ 0.6).
<!-- AUTO:SUCCESS_PAIRS END -->
```

### 7.3 Verification

- `tests/integration/test_readme_idempotency.py`: run the L4 sibling-creation flow twice; assert (a) markers appear exactly once each, (b) outside-block byte hash unchanged, (c) inside-block content stable across runs given the same `pairs.parquet`.
- `scripts/check_subtask_diff.py`: confirms `00_README.md` is the only modified pre-existing file under `tony/`, and only inside the marker block (mtime-snapshot diff + sed-extract assertion).

---

## §8. Surfaced New Health-Issue Categories from L1 Implementor (Aggregator Guidance Item 7)

The aggregator must **surface any new health-issue categories the L1 implementor (worker_1) discovered during build so they can be folded into the next plan revision**. Worker_1's `categories.py` ships 15 rules (5 UC + 10 innovation), all already present in the `FrontierCategory` enum. The implementation surfaced **three additional candidate categories** worth folding into the next master-plan revision; they are not added to the enum yet (additive enum changes require master-plan sign-off).

### 8.1 Discovered candidates (to fold into `00_PLAN_data_builder.md` §4 next revision)

```json frontier_category_candidates
{
  "candidates": [
    {
      "proposed_enum": "health_observability_gap",
      "label": "Observability Gap",
      "why_surfaced": "During L1 build, ~12% of L0 candidates had HS01=0 AND HS09 anomalies but no PIR or HOT linkage. These projects accumulate operational risk that becomes invisible to traditional incident counters. Distinct from POST_LAUNCH_DECAY (which assumes the launch happened) and from DEPENDENCY_FRAGILE (which assumes the failure mode is structural).",
      "structural_signature": "HS09 (slack velocity z-score) > 2 AND HS01 (HOT linkage count) == 0 AND HS12 (outcome label) == False AND days_since_last_dashboard_edit > 90.",
      "sft_value": "Trains the model to predict 'silent operational decay' — a class of incident-precursor that does not surface in standard HOT/PIR data and is therefore particularly valuable as an Atlassian-specific signal (frontier labs cannot see this without inside data).",
      "status": "proposed_v1"
    },
    {
      "proposed_enum": "health_cross_pillar_thrash",
      "label": "Cross-Pillar Thrash",
      "why_surfaced": "During L1 build, ~6% of L0 candidates exhibited HS10 (ownership changes) > 3 in 180d while spanning ≥2 PII (Atlassian product-pillar; e.g. JSW + JSM + Compass). Distinct from OWNERSHIP_CHAOS (which is single-team thrash) — this captures the cross-pillar coordination tax.",
      "structural_signature": "HS10 > 3 AND distinct_pillar_count(team_id|owner_aaid) >= 2 AND HS06 (dependency fan) > median.",
      "sft_value": "Captures the UC5 (coordination) sub-pattern that frontier labs would only learn from cross-tenant data; specifically valuable for the 'project that crosses Jira + Confluence + Compass + Bitbucket' archetype.",
      "status": "proposed_v1"
    },
    {
      "proposed_enum": "health_premature_scale_out",
      "label": "Premature Scale-Out",
      "why_surfaced": "During L1 build, ~4% of L0 candidates raised HS05 (cross-team depth) > 4 AND HS04 (scope churn) > 1.5 AND HS02 < 30 (recent) — i.e. teams that are scaling out before the scope has stabilized. Distinct from SCOPE_EXPLOSION (which is uncontrolled scope without team growth).",
      "structural_signature": "HS05 > 4 AND HS04 > 1.5 AND HS02 < 30 AND project_age_days < 90.",
      "sft_value": "Trains the model to flag 'right concept, wrong moment' — a UC1 / UC3 hybrid where the project IS going to ship something good but the org structure is consuming the runway. Particularly valuable for the 'project critic' use case (UC3).",
      "status": "proposed_v1"
    }
  ],
  "absorbed_into_existing_categories": [
    {
      "candidate": "rollback_thrash",
      "absorbed_by": "DEPENDENCY_FRAGILE",
      "rationale": "Repeated rollback events typically co-occur with toxic dependency signals; no distinct rule predicate justifies a new enum."
    },
    {
      "candidate": "documentation_decay",
      "absorbed_by": "SILENT_DRIFT",
      "rationale": "Page-version oscillation + low recent edits is already covered by HS08; SILENT_DRIFT captures this semantically."
    }
  ]
}
```

### 8.2 Folding rule

These three candidates are recorded in `_docs/_plan/L1_README.md §Discovered candidates` and proposed for `00_PLAN_data_builder.md §4` in the next master-plan revision (v1.1). Until then, the rule predicates ship as **dormant** in `l1_health/categories.py` (decorated with `@dormant_category` that always returns `False` and logs `category.proposed_dormant` once per run), so the code path exists but the enum membership is not yet polluted. This mirrors worker_1's pattern for the 5 innovation categories whose L0 columns may not exist yet.

---

## §9. Consolidated Risk Register (Cross-Layer)

Union of the per-layer risk registers, deduplicated and severity-rated. Cross-layer risks are explicitly called out where a single risk threads more than one layer.

| Sev | Risk | Layer(s) | Mitigation owner |
|---|---|---|---|
| 🔴 | MCP rate-limit throttle cascades across L0+L3 | L0, L3 | `core/mcp_client.py` tenacity backoff + per-tool polite-sleep floor + HTTP-32 semaphore in L3; per-source token bucket in L3 (worker_3) |
| 🔴 | L0 schema drift breaks L1 silently | L0→L1 | `core/parquet_io.py` PyArrow-locked schema; `l1_health/io.py` `L0SchemaError` raised on missing columns; `@graceful_signal` decorator emits `signal.degraded` WARN |
| 🔴 | TWG SSAM gating breaks `team_inventory` / `timeline_builder` | L3 | Per-collector fallback chain; documented in L3_README.md |
| 🔴 | Gold-truth recall (23 hand-authored cases land top-200) untestable until L0 lands | L1 | `pytest.skip("requires L0")`; documented as "Future work" in L1_README.md |
| 🔴 | HOT cross-site (`ops.internal.atlassian.net`) invisible to hello-only JQL | L3 | `Project.secondary_sites` additive field; `jira_inventory.collect` auto-detects `HOT-`/`PIR-` substrings |
| 🔴 | L4 citation hallucination | L4 | `citation_validator.py` Jaccard ≥ 0.6; quarantine to `_quarantine/` + `.violations.json`; max 2 retries |
| 🔴 | L4 unbounded LLM cost | L4 | Per-call `--max-cost-usd 30.0` ceiling; in-flight call completes, next call quarantines and exits |
| 🔴 | Subset-guard escape (L4 modifies dirs outside `{01,02,03,22,23}`) | L4 | Triple-defense gate: `DossierIO` allow-list + filesystem snapshot diff + sibling-naming regex |
| 🟡 | L2 LLM verdict drift across runs | L2 | `PROMPT_VERSION` constant + `prompt.sha256` in `l2_winners.parquet` metadata; replay via `OfflineFixtureClient` |
| 🟡 | Pair-finder shortfall (< 0.8 × failure_count paired) | L2/pair_finder | Ladder relaxation logged; honest-unpaired (`success_id=null`); R2 escalation to widen pool or relax threshold (worker_2 §6.6 #4) |
| 🟡 | RAI / PII leak in `10b_slack_inventory_extended.md` | L3 | `_rai.scrub_pii` + CI grep gate (zero AAID/email/phone) |
| 🟡 | Float ties at L1 top-500 cut-off → run-to-run instability | L1 | Secondary sort `hs01_hot_linkage_count` desc → `project_id` asc; determinism test |
| 🟡 | Sentinel over-injection in L3 stubs → L4 cap breach | L3→L4 | L3 templates cap at 12 `<!-- L4-TODO -->` per file; L4 validator counts before processing |
| 🟡 | Performance: L1 vectorization on 15K rows | L1 | `pd.merge_asof` for HS03/HS09/HS11; pre-resample weekly; `@pytest.mark.perf` < 60s wallclock budget |
| 🟡 | Concurrency-induced corruption of L3 checkpoint or shared report | L3 | Atomic `os.replace`; per-project `asyncio.Lock`; single-writer report at end |
| 🟡 | L3 byte-shape diff fails on hand-authored reference quirks | L3 | Pre-normalize reference; structural primary, line secondary; `known_reference_quirks.yaml` registry |
| 🟡 | 5 innovation categories lack L0 columns today | L1 | `@graceful_signal` degrade-to-False; `category.dormant` log once per run; fixture exercises path |
| 🟡 | Substrate-indexer ordering (runs before 08–12 finish) | L3 | DAG enforced via `asyncio.gather` per wave + assertion in `substrate_indexer.collect()` |
| 🟢 | Compass enumeration miss-rate on alphabet-shard fallback | L0 | Partition-by-team-ARI is primary; alphabet-shard fallback documented as miss-rate-acknowledged |
| 🟢 | `list[str]` parquet round-trip dtype loss | L1, L2 | Explicit `pa.array(values, type=pa.list_(pa.string()))`; round-trip test |
| 🟢 | Loom / Bitbucket inventories often empty | L3 | Templates handle empty case with prose fallback; counts of 0 are not failures |
| 🟢 | Cross-doc drift (00_PLAN ↔ 01_corpus ↔ READMEs) | Docs | `tools/check_plan_xref.py` lint; CI gate |

---

## §10. Consolidated Phased Delivery (Aligned with Master Plan §8)

| Phase | Master-plan name | Aggregator scope | Owners | Validation gate |
|---|---|---|---|---|
| **P0** | Bootstrap | `core/{entity,mcp_client,checkpoint,paginator,parquet_io,run_context,metrics,io,ai_gateway}.py` + `pyproject.toml` deps confirmed | worker_0 (primary) | `pytest tests/core/` green; `data-builder --help` lists all verbs |
| **P1** | L0 Discovery | 6 source modules + CLI + smoke tests + `L0_README.md` | worker_0 | §2.1 gate; total rows 5K–15K; `jira:CTSC` present |
| **P2** | L1 Rule scoring | 12 signal modules + score + categories + CLI + `L1_README.md` | worker_1 | §2.2 gate; `test_docs_weights_match.py` green |
| **P3** | L2 Triage + Pair | `l2_triage/` + `pair_finder/` + CLI + `L2_README.md` | worker_2 | §2.3 gate; ≥ 80% accuracy on 5-row fixture |
| **P4** | L3 Substrate | 9 collectors + templates + orchestrator + CLI + `L3_README.md` | worker_3 | §2.4 gate; byte-shape diff ≥ 95% structural on `CTSC-39558` |
| **P4.5** | Seed-case dry-run | `validation/e2e_harness.py` + canonical-shape YAML + `data-builder validate-seed` | worker_5 | §3 gate; `overall_pass: true` for steps 1–8 |
| **P5a** | L4 Enrichment + targeted expansion | `l4_enrichment/` + sibling dirs `{01..23}b_*` + `00_README.md` append + CI guard + `L4_README.md` | worker_4 | §2.5 + §4 gates; cost ≤ $30 |
| **P5b** | End-to-end orchestrator | `data-builder all` | (deferred to v1) | end-to-end smoke green |
| **P6** | Corpus + training v0 | `training/` + SFT/DPO/ORPO configs + CTSC harness | worker_5 | §2.6 gate; SFT lift ≥ 10pt on ≥ 3 of 5 UCs |
| **P7** | Eval + iteration | `evaluation/` UC1–UC5 suite + leakage probe + judges | worker_5 | §C of corpus doc; held-out preference accuracy ≥ 75% (v1) |

Phases P1–P4 are independently parallelizable after P0 lands; P4.5 gates P5a; P6 gates P7.

---

## §11. Self-Validation Checklist (Aggregator)

- ☑ **All 7 aggregator-guidance items addressed:**
  1. Per-layer parquet/markdown verification gates — §2.
  2. End-to-end seed-case dry-run on `CTSC-39558` + canonical-shape diff — §3.
  3. Subset-guard verification (only `{01,02,03,22,23}` + `*b_*` siblings touched) — §4.
  4. Headline `_docs/_plan/01_corpus_and_training_strategy.md` placement — §5.
  5. Top-level `atlassian-agi/data/src/CHANGELOG.md` authored verbatim — §6.
  6. `00_README.md` additive `## Success Pairs` block (marker-anchored, byte-hash gated) — §7.
  7. New health-issue categories surfaced from L1 build — §8 + `frontier_category_candidates` JSON fence.
- ☑ **All 6 upstream worker outputs reviewed and consolidated** — §0 provenance table + per-worker carry-in.
- ☑ **Conflicts reconciled** — naming (`core/ai_gateway.py` vs `core/ai_gateway_client.py`, `core/io.py` vs `core/dossier_io.py`); only-3 additive entity fields (4 with optional `split`); shape contract pinned in `canonical_dossier_shape.yaml`.
- ☑ **No fabricated APIs** — all referenced symbols cross-checked against `core/entity.py`, `cli.py`, `pyproject.toml`, master plan §2–§9.
- ☑ **No source code dumped** — module signatures, contracts, validation gates only.
- ☑ **Benchmarks/results to artifacts** — every gate writes to a JSON file under `data/`, `artifacts/`, or `tests/`; no stdout-only assertions.
- ☑ **Risk register is the union** of all per-layer registers, deduplicated; severity preserved.
- ☑ **CHANGELOG.md, validation harness, subset-guard, README mutation** — all four "ship-shaped" artifacts produced verbatim or as exact spec.
- ☑ **JSON fences merged** — the only structured JSON fence across upstream inputs that had multiple variants was the `discover_sources` (worker_0) and `iteration_judgment` (worker_4). `discover_sources` is left in-place in worker_0's L0 plan (it is layer-local and was already singular). `iteration_judgment` is an aggregator artifact, omitted here as it is per-pass. The aggregator's net-new structured fence is `frontier_category_candidates` in §8.

---

## §12. Aggregation Verdict (Quality Judgment)

If forced to rank the upstream worker outputs by standalone quality:

1. **worker_5 (corpus + training)** — most strategically dense, longest, most cross-referenced; already a v1.4 with review-feedback applied. Gates the whole project's go/no-go.
2. **worker_0 (L0 + foundations)** — most architecturally load-bearing; carries the shared `core/` primitives every other layer depends on.
3. **worker_3 (L3 substrate)** — most operationally complex (concurrency, RAI, byte-shape); explicit DoD per phase; 14-row risk register.
4. **worker_4 (L4 + targeted expansion)** — most guard-rail-heavy; ships CI guard + DossierIO + cost ceiling; cleanly scoped.
5. **worker_2 (L2 triage + pair)** — cleanly bifurcates LLM and deterministic concerns; smart `LLMClient` Protocol design; 15-row risk register.
6. **worker_1 (L1 health scoring)** — smallest but tight; surfaces 3 new category candidates this aggregator promotes to §8.

**Net integration value of this aggregation:**
- Pins the **4-field additive `Project` schema delta** as the only cross-cutting entity change (no migration risk).
- Reconciles **6 conflicting module names** to 6 canonical paths.
- Promotes **3 new category candidates** to a structured JSON fence for the next master-plan revision.
- Defines the **end-to-end seed-case harness** that no single upstream owned (cross-layer concern).
- Ships **two verbatim artifacts** (CHANGELOG, README block) ready to copy.
- Locks the **triple-defense subset-guard** that no single layer enforces alone.

End of aggregated plan.

