# task — Run Verification Catalog

> **Purpose**: Tool-specific post-run verification for `/task` (the production task tool). Use **together with** the common catalog at `../VERIFICATION.md` (covers A1–A19 / O-1 – O-22 that apply to any BTA/Dual-based tool).
>
> Only **historical, documented** observations appear in §2. Speculative "what could go wrong" items are intentionally excluded — VERIFICATION docs catalog observed reality, not imagined risk.
>
> **Last updated**: 2026-05-27

---

## Tool Profile

| Property | Value |
|----------|-------|
| **Default topology** (plan-only) | `breakdown-multiflow-plan.yaml` — **Dual{BTA{MFDual}}** |
| **Default topology** (full PTI) | `breakdown-multiflow-plan-then-implement.yaml` — plan stage + PTI implement stage, wrapped in outer Dual review |
| **Nesting depth** | 3 levels: outer Dual → BTA workers (MFDual) → per-MFDual {peer flows + inner BTA aggregator} |
| **Aggregator count** | `1 outer BTA aggregator + N_workers inner MFDual aggregators` |
| **Breakdown inferencer** | `RovoDevCLI` by default (cascaded via `_params.default_inferencer`) |
| **Worker inferencer (per BTA child)** | `MultiFlowDualInferencer` (NOT a leaf — it's another orchestrator) |
| **Per-MFDual flow inferencers** | leaves rendered with `plan/main/initial.jinja2` / `followup.jinja2` |
| **Outer Dual fixer** | **LEAF** (lightweight; uses `plan/main/followup.jinja2`) — explicitly NOT the full BTA{MFDual} re-decomposition |
| **Canonical deliverable** | `output.md` |
| **Worker artifact** | `output.md` (per worker MFDual's `final_deliverables/`) |
| **Default `--max-breakdown`** | 3 (`_params.plan_max_breakdown`) |
| **Default `flow_max_dynamic_steps`** | 3 |
| **Default `consensus_max_iterations`** | 3 |
| **Required env** | None for plan-only with `RovoDevCLI`; may need `ROVOCHAT_*` / `JIRA_*` if YAML overrides cascade to RovoChat |

### Expected Workspace Tree (Plan-Only Mode)

```
task_<YYYYMMDD_HHMMSS>_<uuid>/
├── outputs/
│   ├── output.md                       ← symlink → propose/outputs/final_deliverables/output.md
│   ├── output_manifest.json
│   ├── final_deliverables/             ← contains the final symlink target
│   └── round_log.jsonl
├── logs/
├── artifacts/
└── children/
    ├── round_01/                        ← outer Dual review round (additional round_NN if consensus needed)
    └── propose/                         ← outer Dual.base = BTA
        └── children/
            ├── breakdown/               ← BTA breakdown phase
            │   └── outputs/breakdown_output.md
            ├── worker_0/                ← MFDual (NOT a leaf)
            │   ├── logs/.../MultiFlowDualInferencer-*.jsonl.parts/
            │   │     ├── InferenceInput/  InferenceResponse/  InferenceArgs/
            │   │     └── Round01/  Round02/  Round03/        ← per-round dynamic-step artifacts
            │   └── children/
            │       ├── propose/         ← Worker MFDual's internal BTA
            │       │   └── children/
            │       │       ├── aggregator/    ← INNER aggregator (per worker)
            │       │       ├── flow_0/        ← peer flow
            │       │       └── flow_1/        ← peer flow
            │       └── fixer_inferencer/      ← worker MFDual's own fixer leaf
            ├── worker_1/                ← (same shape as worker_0)
            ├── worker_2/                ← (same shape)
            └── aggregator/              ← OUTER BTA aggregator (synthesizes worker MFDual outputs)
```

---

## How to use

1. Run the tool (e.g., `./test_task.sh --background "design a microservices architecture for a notification system"`)
2. Auto-discover latest workspace:
   ```bash
   export TOOL=task
   export CANONICAL_DELIVERABLE=output.md
   export WS=$(ls -td /Users/tchen7/MyProjects/CoreProjects/OpenStartup/_runtime/tasks/task/task_* | head -1)
   echo "Auditing: $WS"
   ```
3. Run the **common audit body** (paste from `../VERIFICATION.md` §1 one-liner sanity script) — verifies A1–A19
4. Run the **tool-specific audit pack** below (TK-A-1 – TK-A-N) — adds task-only structural checks
5. If any check FAILS, consult `../VERIFICATION.md` §2 (common observations) FIRST, then §2 below (task-specific) — root cause may be tool-agnostic.

---

## §1 task-Specific Audit Pack

These rows extend the common audit with task-only structural concerns (multi-aggregator, dynamic rounds, dual-review chain, mode swap, deliverable promotion chain). Each row points to the historical observation (TK-O-N) it guards against.

| # | Check | Pass criterion | Guards |
|---|-------|----------------|--------|
| TK-A-1 | Aggregator count matches topology contract | Exactly `1 outer BTA aggregator + N_workers inner MFDual aggregators` exist with non-empty `InferenceInput/` | TK-O-1 |
| TK-A-2 | Round-numbered subdirs are populated and consecutive | For each `worker_N/.../jsonl.parts/Round<NN>/` directory, the next-lower `Round<NN-1>/` ALSO exists; no gaps; max round ≤ `flow_max_dynamic_steps` | TK-O-1 |
| TK-A-3 | Outer Dual fixer ran as LEAF, not full re-decomposition | If `round_01/` (or later round) exists at TOP level, the fixer's session log shows ONE inferencer call site (leaf path), NOT a nested `children/propose/children/{breakdown,worker_0,..}/` tree | TK-O-2 |
| TK-A-4 | Deliverable promotion chain intact (each hop has canonical content) | For each level `flow_N → MFDual worker → outer BTA aggregator → outer Dual → top` the corresponding `outputs/final_deliverables/output.md` (or symlink target) is non-empty AND content size is non-decreasing through the chain (synthesis grows or preserves) | TK-O-3 |
| TK-A-5 | `--plan` mode loaded plan-only YAML (not plan-then-implement) | When CLI used `--plan`, the workspace's `artifacts/topology_source.yaml` (or equivalent) resolves to `breakdown-multiflow-plan.yaml`, NOT `breakdown-multiflow-plan-then-implement.yaml` | TK-O-4 |
| TK-A-6 | Inner aggregators reference peer flows via `(See file: ...)` | For each `worker_N/children/propose/children/aggregator/.../InferenceInput/*.txt`, the prompt contains `(See file: ...)` references pointing to `flow_0/.../output.md` AND `flow_1/.../output.md` of the SAME worker_N subtree | (Common A7 extension) |
| TK-A-7 | No iteration runaway | `worker_N/.../jsonl.parts/Round<NN>/` count ≤ `flow_max_dynamic_steps`; top-level `round_NN/` count ≤ `consensus_max_iterations` | (Cost guard) |

### Quick wrapper

```bash
# Run common audit body (see ../VERIFICATION.md §1 one-liner)
# Then run task-specifics:

set -u
N_WORKERS=$(ls -d "$WS/children/propose/children/worker_"* 2>/dev/null | wc -l | tr -d ' ')
echo "N_workers detected: $N_WORKERS"

# TK-A-1: aggregator count
OUTER_AGG=$(ls -d "$WS/children/propose/children/aggregator" 2>/dev/null | wc -l)
INNER_AGG=$(ls -d "$WS/children/propose/children/worker_"*/children/propose/children/aggregator 2>/dev/null | wc -l)
echo "TK-A-1 outer aggregators: $OUTER_AGG (expected 1); inner aggregators: $INNER_AGG (expected $N_WORKERS)"

# TK-A-2: round-numbered subdirs consecutive
for w in "$WS/children/propose/children/worker_"*; do
  rounds=$(ls -d "$w"/logs/session/*/Round* 2>/dev/null | sed -E 's/.*Round0*([0-9]+)$/\1/' | sort -n)
  echo "TK-A-2 $(basename "$w") rounds: $(echo $rounds | tr '\n' ' ')"
done

# TK-A-3: outer Dual fixer is leaf
if [ -d "$WS/children/round_01" ]; then
  fixer_children=$(ls -d "$WS/children/round_01/children/"*/children 2>/dev/null | wc -l)
  echo "TK-A-3 fixer nested orchestrator depth (expect 0 for leaf): $fixer_children"
fi

# TK-A-4: promotion chain — count canonical output.md sizes at each level
for level in \
  "$WS/outputs/final_deliverables/output.md" \
  "$WS/children/propose/outputs/final_deliverables/output.md" \
  "$WS/children/propose/children/aggregator/outputs/output.md"; do
  if [ -e "$level" ]; then
    size=$(wc -c < "$level" 2>/dev/null | tr -d ' ')
    echo "TK-A-4 $level → $size bytes"
  else
    echo "TK-A-4 MISSING: $level"
  fi
done

# TK-A-5: mode swap — best-effort check of recorded source YAML
grep -h "breakdown-multiflow-plan" "$WS"/artifacts/*.yaml 2>/dev/null | head -2

# TK-A-6: inner aggregator file-ref pattern
for ia in "$WS/children/propose/children/worker_"*/children/propose/children/aggregator/logs/session/*.jsonl.parts/InferenceInput/*.txt; do
  refs=$(grep -c "(See file:" "$ia" 2>/dev/null)
  echo "TK-A-6 $(echo "$ia" | sed -E 's|.*(worker_[0-9]+).*|\1|') file-refs: $refs (expected ≥ 2 for flow_0+flow_1)"
done

# TK-A-7: iteration runaway
for w in "$WS/children/propose/children/worker_"*; do
  rcount=$(ls -d "$w"/logs/session/*/Round* 2>/dev/null | wc -l | tr -d ' ')
  echo "TK-A-7 $(basename "$w") round count: $rcount (cap = flow_max_dynamic_steps, default 3)"
done
top_rounds=$(ls -d "$WS/children/round_"* 2>/dev/null | wc -l | tr -d ' ')
echo "TK-A-7 outer Dual rounds: $top_rounds (cap = consensus_max_iterations, default 3)"
```

---

## §2 task-Specific Observation Catalog

> Each TK-O-N below is a **documented historical observation**, traceable to a specific plan, source-code comment, or preflight test. Speculative items are intentionally excluded.

### TK-O-1 — MFDual flow round-naming chain broken (Anomaly 8)
- **Look for**: per-flow round subdirs named with the prefix of an unrelated step (e.g., `flow_0_initial_round01/` when the expected name is `flow_0_round01/`); OR rounds nested under the wrong parent (`flow_0/children/default_followup_inferencer_round02/`); OR `flow_X_initial/` directory empty while its sibling `flow_X_initial_round01/` contains what should have been the initial step's output.
- **Source**: `AgentFoundation/_docs/_plan/mfdual_bug_fixes/mfdual_hollow_workspace_anomaly_7_fix_plan.md` (Anomaly 8: LWI round-naming chain refinement; Fix #13).
- **Distinguishes from healthy**: Healthy MFDual flow directory layout has exactly `flow_N/`, `flow_N/<initial>/`, `flow_N/round01/`, `flow_N/round02/`, … with monotonic, gap-free numbering and content in each round dir.

### TK-O-2 — Outer Dual fixer re-runs full BTA{MFDual} decomposition (cost explosion)
- **Look for**: Inside `$WS/children/round_NN/` (an outer Dual review-fix round), the fixer's subtree mirrors the full original BTA shape (`children/propose/children/{breakdown, worker_0, worker_1, ..., aggregator}/`), causing ~10–20× the expected LLM call volume per fix iteration.
- **Source**: Documented inline in the topology YAML self-doc — `OpenStartup/src/openteam/server/resources/tools/task/topologies/breakdown-multiflow-plan.yaml` lines 16–26 ("DualInferencer behavior makes fixer = base_inferencer … ~10-20× more LLM calls than needed for typical plan-quality feedback"). The lightweight-fixer wiring exists specifically to prevent regression.
- **Distinguishes from healthy**: A healthy fixer round is a single leaf inferencer that takes `plan/main/followup.jinja2` + prior plan + reviewer feedback and emits a refined plan; its session log contains a SINGLE inferencer call site without nested `children/` subtree.

### TK-O-3 — Deliverable promotion chain drops content at a hop
- **Look for**: The top-level `$WS/outputs/output.md` exists but its content is a tiny BTA "summary text" wrapper instead of being a symlink (or copy) of the substantive aggregator output; OR one of the intermediate hops (`worker_N/.../outputs/final_deliverables/output.md`, `outer aggregator/outputs/output.md`, etc.) is missing or empty despite later/earlier hops being populated.
- **Source**: `AgentFoundation/_docs/_plan/mfdual_bug_fixes/mfdual_hollow_workspace_anomaly_7_fix_plan.md` (Anomaly 7: hollow MFDual subtree + Bug 1 / unified_finalize_output work — `outputs/output.md` was summary text instead of symlinked canonical).
- **Distinguishes from healthy**: Each hop in the chain `flow_N → worker MFDual final_deliverables → outer BTA aggregator → outer Dual top` has a non-empty `output.md`, and the top-level `outputs/output.md` is either the canonical content or a symlink to it.

### TK-O-4 — `--plan` mode loaded plan-then-implement YAML by mistake
- **Look for**: Run was invoked with `--plan` but execution proceeds into the implementation stage; OR the outer Dual reviews an empty PTI implementation deliverable using `template_root_space=implementation` criteria (wrong review semantics, wasted iterations).
- **Source**: `OpenStartup/test/openteam/resources/tools/task/preflight/test_plan_mode_yaml_swap.py` (preflight regression test) — guards `executor._run_topology` swap from `breakdown-multiflow-plan-then-implement.yaml` → `breakdown-multiflow-plan.yaml` when `mode == "plan"`.
- **Distinguishes from healthy**: When `--plan` is used, the workspace's effective topology contains NO PTI implementation stage; only Dual{BTA{MFDual}} appears; review criteria explicitly load `plan/main/review.jinja2` (not `implementation/main/review.jinja2`).

---

## §3 Authoring Guide — Adding a NEW task-Specific Observation

Follow the same rules as the common catalog (`../VERIFICATION.md` §3):

1. **Observation, not cause.** Describe what an unhealthy run LOOKS LIKE in the workspace/log, not why it happened.
2. **Historical-only.** Add a TK-O entry ONLY if there is documented evidence the issue occurred — cite the source (a plan file, code comment, test, or a recorded run workspace). Do NOT add speculative "what could go wrong" entries.
3. **Linkable to an audit row.** Each TK-O-N should have at least one TK-A-M (or shared A-row) that detects it.
4. **Source citation required.** Each entry must include a `**Source**:` bullet pointing to the specific file/line/run that documents the observation.

---

## §4 Run Comparison — Historical Baselines

| Run ID | Date | Topology | Result | Notes |
|--------|------|----------|--------|-------|
| `task_20260524_015320_c7744338` | 2026-05-24 | plan-only (Dual{BTA{MFDual}}) | ✅ Full tree shape verified (worker_0..2, flow_0/1 per worker, inner+outer aggregators, round01 outer Dual review) | Used as the structural reference for §1 audit pack expected layout. |

(Add new baselines by appending rows here as runs accumulate.)
