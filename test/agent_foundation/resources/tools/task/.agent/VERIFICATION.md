# task — Run Verification Catalog

> **Purpose**: Tool-specific post-run verification for `/task` (the production task tool). Use **together with** the common catalog at `../../.agent/VERIFICATION.md` (covers A1–A20 / O-1 – O-24 that apply to any BTA/Dual-based tool).
>
> Only **historical, documented** observations appear in §2. Speculative "what could go wrong" items are intentionally excluded — VERIFICATION docs catalog observed reality, not imagined risk.
>
> **Last updated**: 2026-06-29 (added TK-A-17 / TK-O-12 covering the v8 breakdown-collapse failure mode + its fix; scoped TK-A-16 for deterministic guardrails. Earlier: merged the OS task-catalog additions — TK-A-9..16 / TK-O-6..11 covering the v5–v7 consensus-fix, audit-log, session-log, panelist, scaffold, and output-guardrail-judge issues; common catalog moved to `tools/.agent/`)

---

## Tool Profile

| Property | Value |
|----------|-------|
| **Default topology** (plan-only) | `breakdown-multiflow-plan.yaml` — **Dual{BTA{MFDual}}** |
| **Default topology** (full PTI) | `configs/default.yaml` — imports `breakdown-multiflow-plan.yaml` (plan stage) and adds the PTI implement stage, wrapped in outer Dual review. `--plan` on this file swaps to the standalone planner; plan-only presets (`full-plan`/`breakdown`/`multiple`) are a clean no-op. |
| **Nesting depth** | 3 levels: outer Dual → BTA workers (MFDual) → per-MFDual {peer flows + inner BTA aggregator} |
| **Aggregator count** | `1 outer BTA aggregator + N_workers inner MFDual aggregators` |
| **Breakdown inferencer** | `ClaudeCodeCLI` by default (cascaded via `_params.main_inferencer` ← `_params.default_inferencer`); overridable per-run via `DEFAULT_MAIN_INFERENCER` env (e.g. `RovoDevCLI`, as used by this run's sourced `.env`) |
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
├── run_state/                          ← per-run node state store (store.json) — state/definition-separation refactor
└── children/
    ├── round_01/                        ← outer Dual review round (additional round_NN if consensus needed)
    └── propose/                         ← outer Dual.base = BTA
        └── children/
            ├── breakdown/               ← BTA breakdown phase
            │   └── outputs/output.md
            ├── worker_0/                ← MFDual (NOT a leaf)
            │   ├── logs/.../MultiFlowDualInferencer-*.jsonl.parts/
            │   │     ├── InferenceInput/  InferenceResponse/
            │   │     └── Round01/ [Round02/ Round03/]      ← MFDual consensus rounds (≤ consensus_max_iterations); only as many as actually ran
            │   └── children/
            │       ├── propose/         ← Worker MFDual's internal BTA
            │       │   └── children/
            │       │       ├── aggregator/    ← INNER aggregator (per worker)
            │       │       ├── flow_0/        ← peer flow (children/{initial, round01[, round02]})
            │       │       └── flow_1/        ← peer flow
            │       ├── fixer_inferencer/      ← worker MFDual's own fixer leaf (empty unless a fix fired)
            │       ├── review_inferencer/     ← worker MFDual's own reviewer leaf
            │       └── round_01/              ← worker MFDual's own review round (children/{review}; more if consensus needed)
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
   export WS=$(ls -td /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_runtime/tasks/*/* | head -1)   # AF nests runs under tasks/<tool-name>/<run-id>; run dirs are <tool-name>_<YYYYMMDD_HHMMSS>_<uuid>, NOT task_*
   echo "Auditing: $WS"
   ```
3. Run the **common audit body** (paste from `../../.agent/VERIFICATION.md` §1 one-liner sanity script) — verifies A1–A19
4. Run the **tool-specific audit pack** below (TK-A-1 – TK-A-N) — adds task-only structural checks
5. If any check FAILS, consult `../../.agent/VERIFICATION.md` §2 (common observations) FIRST, then §2 below (task-specific) — root cause may be tool-agnostic.

---

## §1 task-Specific Audit Pack

These rows extend the common audit with task-only structural concerns (multi-aggregator, dynamic rounds, dual-review chain, mode swap, deliverable promotion chain). Each row points to the historical observation (TK-O-N) it guards against.

| # | Check | Pass criterion | Guards |
|---|-------|----------------|--------|
| TK-A-1 | Aggregator count matches topology contract | Exactly `1 outer BTA aggregator + N_workers inner MFDual aggregators` exist with non-empty `InferenceInput/` | TK-O-1 |
| TK-A-2 | Round-numbered subdirs are populated and consecutive | TWO round layers, each consecutive/gap-free: (a) MFDual consensus rounds at `worker_N/logs/session/*.jsonl.parts/Round<NN>/`, max ≤ `consensus_max_iterations`; (b) per-flow dynamic steps at `worker_N/children/propose/children/flow_M/children/{initial,round01,round02,...}`, max ≤ `flow_max_dynamic_steps` | TK-O-1 |
| TK-A-3 | Outer Dual fixer ran as LEAF, not full re-decomposition | If `round_01/` (or later round) exists at TOP level, the fixer's session log shows ONE inferencer call site (leaf path), NOT a nested `children/propose/children/{breakdown,worker_0,..}/` tree | TK-O-2 |
| TK-A-4 | Deliverable promotion chain intact (each hop has canonical content) | For each level `flow_N → MFDual worker → outer BTA aggregator → outer Dual → top` the corresponding `outputs/final_deliverables/output.md` (or symlink target) is non-empty, and the top is canonically promoted (symlinked to the aggregator's content). Size is NOT required to be monotonic — an integrate-and-distill aggregator may legitimately produce a SMALLER output than any single input (e.g. for a "brief plan" request); only non-emptiness + faithful coverage matter. | TK-O-3 |
| TK-A-5 | `--plan` mode resolved a plan-only topology (no implementation stage) | NO topology YAML is written to the workspace; detect plan-only STRUCTURALLY: the top inferencer in `logs/session` is `DualInferencer` over `BreakdownThenAggregate` over `MultiFlow` (Dual{BTA{MFDual}}), AND `grep -rl "PlanThenImplement\|executor_inferencer\|implementation/main" $WS` is empty (no PTI/implement subtree). AF full-PTI topology is `configs/default.yaml`, not the OS-era plan-then-implement YAML. | TK-O-4 |
| TK-A-6 | Inner aggregators reference peer flows via `(See file: ...)` / `(See outputs folder: ...)` | For each `worker_N/children/propose/children/aggregator/.../InferenceInput/*.txt`, the prompt contains a marker pair — `(See outputs folder: .../flow_N/outputs)` AND `(See file: .../flow_N/outputs/output.md)` — for BOTH flow_0 AND flow_1 of the SAME worker_N subtree | (Common A7 extension) |
| TK-A-7 | No iteration runaway | `worker_N/.../jsonl.parts/Round<NN>/` count ≤ `flow_max_dynamic_steps`; top-level `round_NN/` count ≤ `consensus_max_iterations` | (Cost guard) |
| TK-A-8 | Flow FOLLOWUP-step input references the prior step's full-output file on disk (own + peer), not just the lossy summary | For each `flow_M/children/round*/.../InferenceInput/*.txt` (round ≥ 01), the prompt MUST contain the own-flow on-disk reference — an `on disk at` / `previous full artifact` block pointing at the flow's own prior `children/{initial\|round*}/outputs/output.md`; AND, when a visible peer has already produced output, a `full peer artifact is available` reference to that peer's output. The referenced path MUST exist + be non-empty. Summary-text-only (no file path) FAILS. **Distinct from TK-A-6**: TK-A-6 audits the AGGREGATOR input (BTA-orchestrator resolution — always passed); TK-A-8 audits the FOLLOWUP-step input (MFI `_resolve_flow_output_path` — regressed under M7, fixed 2026-06-25). A passing TK-A-6 says NOTHING about TK-A-8. | TK-O-5 |
| TK-A-9 | Fix contributed to every non-consensus worker | For every MFDual worker whose last review was NOT consensus (`consensus_reached=false` in `round_log.jsonl`), the worker's `output.md` MUST differ from its `propose/outputs/.../output.md` (a fix actually changed the output). Byte-identical = fix silently skipped. | TK-O-6 |
| TK-A-10 | No CRITICAL-rejected output shipped | No worker's final `output.md` corresponds to a last-round review with unresolved CRITICAL severity. Check the last `audit_kind: "merged"` entry in each worker's `round_log.jsonl` — if `severity=CRITICAL`/`approved=false`, a fix entry MUST follow (or the worker is flagged `DEGRADED`). | TK-O-6 |
| TK-A-11 | Round-log entries carry decision fields | Every `review` entry in `round_log.jsonl` contains `consensus_reached`, `severity`, `approved`, `audit_kind`; every `fix` entry contains `audit_kind: "fix"` + `consensus_reached`. Bare `{round,phase,inferencer_class,timestamp}` rows FAIL. | TK-O-7 |
| TK-A-12 | Per-panelist audit entries present (panel mode) | In panel mode (2+ reviewers), each review round has per-panelist entries (`audit_kind: "panelist"`, with `panelist: panelist_NN`) PLUS one merged entry (`audit_kind: "merged"`). | TK-O-7 |
| TK-A-13 | Session logs for ALL CLI inferencer types | Every flow step (`initial`, `round01`, …) AND every reviewer/fixer leaf has a NON-empty `logs/session/` regardless of CLI type (Codex, Claude, RovoDev). A created-but-empty `logs/session/` FAILS. | TK-O-8 |
| TK-A-14 | Consistent panelist dirs | In panel mode the primary reviewer gets `review/children/panelist_00/` (not bare `review/`); all panelists have `panelist_NN/` dirs starting at `00` with no numbering gaps. | TK-O-9 |
| TK-A-15 | No empty scaffold dirs | No empty `children/review/` or `children/fix/` at the worker root, and no empty `children/guardrail/` (scaffold subdirs present but 0 files) for accepted (PASS) guardrail verdicts. | TK-O-10 |
| TK-A-16 | Output-guardrail judge actually JUDGED (not double-wrapped) | For **LLM/CLI judges** (the aggregator + flow leaves, whose `output_guardrail_inferencer` is a `${_params.main_inferencer}` CLI): each `children/guardrail/` has a non-empty `logs/session/` and the judge emitted a parseable verdict (`PASS`/`RESTART`/`RETRY_WITH_REFERENCE`/`CONTINUE`). The judge's `InferenceInput` is the judge prompt VERBATIM (starts with `You are a lightweight quality judge`), NOT wrapped in the planning template (`You are tasked with creating artifacts…`). No `already claimed by creator` CollisionError for the guardrail path. **Carve-out**: a *deterministic* guardrail (e.g. `SubtaskStructureJudge` on the breakdown leaf — pure-Python, duck-typed) intentionally makes **no** CLI call, so it creates **no** `children/guardrail/` dir and writes **no** session log — do NOT flag its absence; verify it via TK-A-17 (subtask count) instead. | TK-O-11 |
| TK-A-17 | Breakdown produced subtasks → flows actually spawned (never a SILENT 0) | A live breakdown yields ≥1 subtask: `children/propose/children/` has ≥1 `worker_NN/` AND an `aggregator/`, and the breakdown `outputs/output.md` is a JSON decomposition (not pure narration). If a breakdown legitimately yields 0 (transient/agent failure), the degrade MUST be LOUD: a `BREAKDOWN_EMPTY_DEGRADE` warning is present in the logs (never a silent skip-to-0-workers with exit 0). FAIL = `propose/children/` has only `breakdown/` (no workers, no aggregator) AND no `BREAKDOWN_EMPTY_DEGRADE` warning. | TK-O-12 |

### Quick wrapper

```bash
# Run common audit body (see ../../.agent/VERIFICATION.md §1 one-liner)
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
  "$WS/children/propose/children/aggregator/outputs/final_deliverables/output.md"; do
  if [ -e "$level" ]; then
    size=$(wc -c < "$level" 2>/dev/null | tr -d ' ')
    echo "TK-A-4 $level → $size bytes"
  else
    echo "TK-A-4 MISSING: $level"
  fi
done

# TK-A-5: plan-only resolution — NO topology yaml is written; detect structurally
echo "TK-A-5 PTI markers (expect 0): $(grep -rl 'PlanThenImplement\|executor_inferencer\|implementation/main' "$WS" 2>/dev/null | wc -l | tr -d ' ')"
echo "TK-A-5 top inferencer: $(ls "$WS"/logs/session/DualInferencer-*.jsonl >/dev/null 2>&1 && echo 'DualInferencer (Dual{BTA{MFDual}}) OK' || echo 'NOT Dual — check topology')"

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

# TK-A-8: flow FOLLOWUP-step input must reference the prior step's FULL-OUTPUT file
# (root-cause guard: MFI _format_followup_input own_path/peer_path must resolve from the
#  LIVE ctx-published per-run workspace, NOT a stale leaf instance _workspace).
fu_total=0; fu_ok=0
for fi in "$WS"/children/propose/children/worker_*/children/propose/children/flow_*/children/round*/logs/session/*.jsonl.parts/InferenceInput/*.txt; do
  [ -e "$fi" ] || continue
  fu_total=$((fu_total+1))
  if grep -qi "on disk at\|previous full artifact\|full peer artifact is available" "$fi"; then
    fu_ok=$((fu_ok+1))
  else
    echo "TK-A-8 FAIL (summary-only, no prior-output file ref): $(echo "$fi" | sed -E 's|.*(worker_[0-9]+).*(flow_[0-9]+).*(round[0-9]+).*|\1/\2/\3|')"
  fi
done
echo "TK-A-8 followup inputs with prior-output file ref: $fu_ok / $fu_total (expect all; 0/N = the M7 stale-_workspace regression, fixed 2026-06-25)"
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
- **Source**: Documented inline in the topology YAML self-doc — `agent_foundation/src/agent_foundation/resources/tools/task/configs/breakdown-multiflow-plan.yaml` (the "lightweight fixer" block: `fixer_inferencer` is declared as a single leaf `_target_: ${_params.main_inferencer}`, and the worker MFDual uses `fixer_strategy: winner` — reusing a winning flow leaf, not a re-decomposition; "DualInferencer behavior makes fixer = base_inferencer … ~10-20× more LLM calls than needed for typical plan-quality feedback"). The lightweight-fixer wiring exists specifically to prevent that cost regression.
- **Distinguishes from healthy**: A healthy fixer round is a single leaf inferencer that takes `plan/main/followup.jinja2` + prior plan + reviewer feedback and emits a refined plan; its session log contains a SINGLE inferencer call site without nested `children/` subtree.

### TK-O-3 — Deliverable promotion chain drops content at a hop
- **Look for**: The top-level `$WS/outputs/output.md` exists but its content is a tiny BTA "summary text" wrapper instead of being a symlink (or copy) of the substantive aggregator output; OR one of the intermediate hops (`worker_N/.../outputs/final_deliverables/output.md`, `outer aggregator/outputs/final_deliverables/output.md`, etc.) is missing or empty despite later/earlier hops being populated.
- **Source**: `AgentFoundation/_docs/_plan/mfdual_bug_fixes/mfdual_hollow_workspace_anomaly_7_fix_plan.md` (Anomaly 7: hollow MFDual subtree + Bug 1 / unified_finalize_output work — `outputs/output.md` was summary text instead of symlinked canonical).
- **Distinguishes from healthy**: Each hop in the chain `flow_N → worker MFDual final_deliverables → outer BTA aggregator → outer Dual top` has a non-empty `output.md`, and the top-level `outputs/output.md` is either the canonical content or a symlink to it.

### TK-O-4 — `--plan` mode loaded plan-then-implement YAML by mistake
- **Look for**: Run was invoked with `--plan` but execution proceeds into the implementation stage; OR the outer Dual reviews an empty PTI implementation deliverable using `template_root_space=implementation` criteria (wrong review semantics, wasted iterations).
- **Source**: AF `agent_foundation/src/agent_foundation/resources/tools/task/executor.py` (the `mode == "plan"` branch) — plan-only presets like `full-plan` are a clean no-op, while `configs/default.yaml` swaps to the standalone `breakdown-multiflow-plan.yaml`. (Migrated from the OpenStartup preflight `test_plan_mode_yaml_swap.py`.)
- **Distinguishes from healthy**: When `--plan` is used, the workspace's effective topology contains NO PTI implementation stage; only Dual{BTA{MFDual}} appears; review criteria explicitly load `plan/main/review.jinja2` (not `implementation/main/review.jinja2`).

### TK-O-5 — Flow followup-step input lacks the prior step's full-output file reference (agent refines from a lossy summary)
- **Look for**: For a flow at step ≥ 1, the followup `InferenceInput` (`flow_M/children/round*/.../InferenceInput/*.txt`) carries ONLY the `You previously produced this artifact (flow N, step K):` header + the summarized `<Response>` text, with NO `Your previous full artifact is on disk at: <path>` block and NO `The full peer artifact is available at: <path>` block for visible peers. Tell-tale: the round01 input is materially SMALLER than the flow's `children/initial/outputs/output.md` (e.g. ~11KB input vs ~21–30KB full output) — the agent refines from the summary, not the full artifact on disk.
- **Source**: `agent_foundation/.../flow_inferencers/multi_flow_inferencer.py` — `_format_followup_input` Part A (`if own_path:` / `if peer_path:`) is gated on `_resolve_flow_output_path`, which (pre-fix) read the flow-config leaf INSTANCES' `_workspace`. Under M7 the per-run workspace is published into the run-context (`_rc_child("step_{i}", workspace=...)`), NOT onto the leaf instance, so resolution returned `None` and the path blocks were silently dropped. Observed in run `wsfix8_20260624_045538_711816c2`: 0/7 followup inputs carried the block while 4 aggregator inputs carried 9 `(See file:)` refs. **Fixed 2026-06-25**: the resolver now reads a per-run `_latest_per_flow_path` map captured from the LIVE run-context (mirroring the `_latest_per_flow` text capture); guarded by `test_mfdual_path_aware_peer_visibility.py::TestFollowupPathUnderCtxPublishedWorkspace`. Same ctx-published-workspace-vs-stale-instance-`_workspace` class as the three M7 logging/naming/deferred-logger fixes.
- **Distinguishes from healthy**: A healthy followup input contains, in addition to the summary text, an explicit `on disk at` / `previous full artifact` path to the flow's own prior `children/{initial|round*}/outputs/output.md` (must exist + be non-empty), plus a `full peer artifact is available at` path for each visible peer that has produced output.
- **Contrast with TK-A-6 (why both rows exist)**: TK-A-6's `(See file:)` refs come from the BTA re-deriving each worker's path from the ORCHESTRATOR's OWN live workspace (`_bta_self._workspace.child(...)`) — an M7-correct path that always PASSED. TK-O-5 / TK-A-8 audit the FOLLOWUP-step input, whose path resolution went through the stale-leaf `_workspace`. A green TK-A-6 is NOT evidence for TK-A-8.

### TK-O-6 — Worker fix step silently skipped despite a CRITICAL review rejection (B1)
- **Look for**: A worker's `round_log.jsonl` has a review entry with `consensus_reached: false` and `severity: CRITICAL`/`MAJOR` but NO subsequent fix entry; the worker's `output.md` is byte-identical to `propose/outputs/.../output.md` (review/fix contributed nothing); `consensus_achieved: true` in the MFDual `InferenceResponse` despite the rejection.
- **Source**: Runs `multimodal_plan_3flow_20260627_140601_8d62955c` (v5) and `…_20260628_091301_3b63d7fd` (v6) — worker_00 hit this in both. Root cause: `fixer_strategy=winner` with NO winner detected → fixer fell back to the MFI orchestrator (not a `TemplatedInferencerBase`) → `_RoleDisabledError` → handler forced `consensus_reached=True`. **Fixed** in `multi_flow_dual_inferencer.py::_select_reviewer_and_fixer` (when `winner is None`, fall back to the first flow's `initial_inferencer` as fixer); confirmed in v7 (2 graceful FixerFallbacks, all workers ran fix cycles).
- **Distinguishes from healthy**: a non-consensus worker has fix entries in `round_log`, `output.md` differs from propose, and `total_iterations` reflects the consensus loop actually iterating.

### TK-O-7 — round_log.jsonl lacks decision fields / per-panelist entries (B2)
- **Look for**: `round_log.jsonl` entries contain only `{round, phase, inferencer_class, timestamp}` — missing `consensus_reached`/`severity`/`approved`/`audit_kind`/`panelist`. In panel mode, only one review entry per round (no per-panelist breakdown).
- **Source**: Run `…_20260627_140601_8d62955c` (v5) — all workers' round_logs were bare. **Fixed** in `dual_inferencer.py` by passing `extra=` dicts at the review (`:1604`) and fix (`:1823`) audit call sites, plus per-panelist audit entries (`audit_kind: "panelist"`) and a merged entry (`audit_kind: "merged"`).
- **Distinguishes from healthy**: per-panelist entries for each reviewer + a merged entry with the decision fields, and fix entries with `audit_kind: "fix"` + `consensus_reached`.

### TK-O-8 — Per-step session logs empty for CodexCLI / ClaudeCodeCLI (B3)
- **Look for**: Flow-step dirs (`flow_00/children/initial/logs/session/`, `flow_01/children/round01/logs/session/`) exist but contain 0 files — only RovoDevCLI (flow_02) writes session logs. The `logs/` dir is present (workspace was assigned) but the logger was never resolved for the other CLI types.
- **Source**: Run `…_20260627_140601_8d62955c` (v5) — flow_00 (Codex) + flow_01 (Claude) had 0 session files across all workers/steps; flow_02 (RovoDev) had 8/step. **Fixed** via workspace/ctx propagation so the deferred logger resolves for all CLI types; confirmed in v6/v7 (all three CLI types log 8 files/step).
- **Distinguishes from healthy**: every flow step has a non-empty `logs/session/` (InferenceInput + InferenceResponse) regardless of CLI type.

### TK-O-9 — Primary reviewer (panelist 0) has no dedicated workspace dir (B4)
- **Look for**: In panel mode the primary reviewer runs under bare `review/` while extras get `review/children/panelist_01/`, `panelist_02/` — no `panelist_00/`. Panelist-dir count ≠ reviewer count (e.g. 2 reviewers but 1 panelist dir).
- **Source**: Run `…_20260627_140601_8d62955c` (v5) — worker_01/02 had only `panelist_01/` (1 dir for 2 reviewers). **Fixed** in `dual_inferencer.py` (primary reviewer gets `panelist_00/` in panel mode; single-reviewer mode keeps bare `review/`); confirmed in v6/v7.
- **Distinguishes from healthy**: all reviewers have `panelist_NN/` dirs starting at `00`, no gaps.

### TK-O-10 — Empty scaffold dirs pollute the workspace tree (B5)
- **Look for**: Each worker has empty `children/review/` + `children/fix/` at the worker root (alongside the real `round_NN/children/review|fix/`); and/or empty `children/guardrail/` dirs (scaffold subdirs present, 0 files) created when the guardrail judge accepted (PASS) without writing.
- **Source**: v5 (6 empty review/fix scaffolds) + v7 (`…_20260628_152604_4fab40a1`, ~20 empty guardrail scaffolds). **Fixed** by (1) not creating the workspace in `_reassign_role_workspace` (review/fix scaffolds) and (2) removing the eager `ensure_dirs()` from `_run_output_guardrail` (guardrail scaffolds).
- **Distinguishes from healthy**: no empty `review/`, `fix/`, or `guardrail/` dirs — every dir that exists contains files.

### TK-O-11 — Output-guardrail judge double-wrapped → planned instead of judging; ran without context isolation (v7)
- **Look for**: A `children/guardrail/` whose `logs/` and `outputs/` are EMPTY but whose `_runtime/inferencer_cache/` holds a stream file containing **plan-writing narration** ("I'll investigate… write the consolidated plan") rather than a `PASS`/`RESTART` verdict; the judge's `InferenceInput` starts with the PLANNING template (`You are tasked with creating artifacts…`) with `You are a lightweight quality judge` buried inside `## Original User Request` (double-wrap); `already claimed by creator … CollisionError` followed by `Output guardrail judge failed … accepting output (fail-open)` in the log. **Auditor trap**: this judge session is easily mistaken for a SECOND aggregator invocation with a "default preamble" (the source of the spurious v7 "A16 fail" / "aggregator ran twice" findings) — a `winner_pick`/`<Winner>` grep on the judge cache is meaningless.
- **Source**: Run `multimodal_plan_3flow_20260628_152604_4fab40a1` (v7) — every guardrail judge (outer + 3 inner) was double-wrapped and never actually judged (silent fail-open). Root cause: `inferencer_base.py::_run_output_guardrail` rendered the complete judge prompt then called `judge.ainfer(prompt)` on a judge that inherited the plan template (→ re-wrapped → did planning), AND ran with no isolated run-context (→ claimed the caller's ctx node → CollisionError; cache landed under `guardrail/` while the session log resolved to the caller's workspace). **Fixed**: render `recovery/judge` via `self.template_manager` (unified with `_render_recovery_prompt`), neutralize the judge's `template_manager` so it executes the pre-rendered prompt verbatim, and run it under its own `_rc_child("guardrail")` with the workspace published (M7) — see `_prepare_guardrail_judge`. Guarded by `test_output_guardrail.py::TestGuardrailNoDoubleWrap` + `TestGuardrailUnifiedRendering`.
- **Distinguishes from healthy**: the judge emits a parseable verdict; `children/guardrail/logs/session/` + `outputs/` are populated; no CollisionError; the judge's `InferenceInput` is the judge prompt verbatim (no planning-template wrapper).

---

### TK-O-12 — Breakdown turn cut short → 0 subtasks → ENTIRE multi-flow pipeline silently skipped (v8)
- **Look for**: `children/propose/children/` containing ONLY `breakdown/` (no `worker_NN/`, no `aggregator/`); the breakdown `outputs/output.md` is pure narration ("I'll start by investigating the codebase…") with no JSON `subtasks` array; **no** `BREAKDOWN_EMPTY_DEGRADE` warning in the log; yet the run still exits 0 with a populated top-level `outputs/output.md` symlink (the outer Dual reviewed/fixed the raw narration into a single-agent plan). Run wall-clock is anomalously short (v8 = 41 min vs a healthy ~2 h). The aggregator + guardrail judge NEVER run, so a "0 collisions" health signal is meaningless.
- **Source**: Run `multimodal_plan_3flow_20260629_021219_1ca9970a` (v8). Root cause was NOT the agent choosing to write a file — the breakdown agent's final turn was killed mid-stream by the nested `claude` CLI's byte-stream idle watchdog (`CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS`, inherited at 20000 ms from the interactive session): a normal >20 s think-pause tripped a synthetic `API Error: Response stalled mid-stream` that ended the turn before any inline `<Response>` JSON was emitted (return code still 0). The BTA then parsed 0 subtasks and **silently** returned the raw narration (`if not sub_queries: return raw_output`), skipping all worker fan-out. **Fixed** with three layers: (1) `ClaudeCodeCliInferencer.byte_stream_idle_timeout_ms=120000` written into the spawned subprocess env (`_build_subprocess_env`) so a long think-pause no longer kills spawned agents; (2) a deterministic `SubtaskStructureJudge` (`output_guardrail_inferencer`) + `max_retry: 3` on the breakdown leaf (`breakdown-multiflow-plan.yaml` + `default.yaml`) so a narration/0-subtask breakdown is REJECTED and retried; (3) BTA now emits a LOUD `BREAKDOWN_EMPTY_DEGRADE` `self.log_warning` then degrades — never a silent skip. Guarded by `test/agent_foundation/common/inferencers/guardrails/test_subtask_structure_judge.py` (judge + failsafe) and `…/external/claude_code/test_byte_stream_idle_env.py` (scoped timeout). See also common-catalog **O-24**.
- **Distinguishes from healthy**: v9 (`…_20260629_103951_192264bc`) — breakdown `output.md` is a 12 KB JSON decomposition → 3 `worker_NN/` + `aggregator/` spawned; ran ~2 h; aggregator guardrail judge ran 61× with real verdicts; 0 `BREAKDOWN_EMPTY_DEGRADE`.

---

## §3 Authoring Guide — Adding a NEW task-Specific Observation

Follow the same rules as the common catalog (`../../.agent/VERIFICATION.md` §3):

1. **Observation, not cause.** Describe what an unhealthy run LOOKS LIKE in the workspace/log, not why it happened.
2. **Historical-only.** Add a TK-O entry ONLY if there is documented evidence the issue occurred — cite the source (a plan file, code comment, test, or a recorded run workspace). Do NOT add speculative "what could go wrong" entries.
3. **Linkable to an audit row.** Each TK-O-N should have at least one TK-A-M (or shared A-row) that detects it.
4. **Source citation required.** Each entry must include a `**Source**:` bullet pointing to the specific file/line/run that documents the observation.
5. **One audit row per independent resolution path.** When two independent code paths surface the "same" on-disk reference (e.g. aggregator vs per-step followup, or own-flow vs peer-flow), EACH needs its OWN content-level audit row — a green check on one is NOT evidence for the other (see TK-A-6 vs TK-A-8). Structural existence of a round dir (TK-A-2) is insufficient: a silently-dropped path block leaves the dir present but the prompt lossy. Treat `getattr(inf, "_workspace", ...)` used for path resolution on a SHARED/factory child instance as an M7 anti-pattern (resolve from the live ctx-captured per-run workspace instead) and flag it in review.

---

## §4 Run Comparison — Historical Baselines

| Run ID | Date | Topology | Result | Notes |
|--------|------|----------|--------|-------|
| `task_20260524_015320_c7744338` | 2026-05-24 | plan-only (Dual{BTA{MFDual}}) | ✅ Full tree shape verified (worker_0..2, flow_0/1 per worker, inner+outer aggregators, round01 outer Dual review) | Pre-refactor OpenStartup baseline (superseded by `wsfix8` below as the canonical reference). |
| `wsfix8_20260624_045538_711816c2` | 2026-06-24 | plan-only (Dual{BTA{MFDual}}) | ✅ 3 workers, 2 flows each, 1 outer + 3 inner aggregators (all with session logs), deliverable symlinked to canonical, post-M7 round naming healthy (`initial`/`round01`, 0 `step_*`) | **AF-native structural reference** — first healthy run after the state/definition-separation refactor + OS→AF migration + the three M7 fixes; matches the updated tree above. ⚠️ NOT clean on the followup-path channel (TK-O-5/TK-A-8): 0/7 flow-followup InferenceInputs carried the prior-output file ref (resolved pre-fix). The `_resolve_flow_output_path` M7 fix landed 2026-06-25; a fresh post-fix run is the TK-A-8 reference. |
| `wsfollowupfix1_20260625_225751_f2d000ad` | 2026-06-25 | plan-only (Dual{BTA{MFDual}}); flows = CodexCLI + RovoDevCLI | ✅ **TK-A-8 4/4** followup inputs carry the prior-step on-disk ref (own + peer); TK-A-6 3/3; all aggregators logged; `initial`/`round01` naming, 0 `step_*`; deliverable 29.6KB symlinked to canonical | **Post-fix TK-A-8 reference** — first run after the `_resolve_flow_output_path` M7 fix (2026-06-25); validates own_path + peer_path across heterogeneous codex/rovodev flows. |

(Add new baselines by appending rows here as runs accumulate.)
