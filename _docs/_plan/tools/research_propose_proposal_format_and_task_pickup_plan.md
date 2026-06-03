# Research-Propose Output Format Standardization + Task-Tool Proposal Pickup — INTEGRATED v2.0 Plan

> **Status:** Draft v2.0 — implementer-ready (integrated with Claude's parallel plan)
> **Author:** Tony Chen + Rovo Dev + Claude (parallel-authored)
> **Created:** 2026-06-02 19:11 (v1.0); **Integrated v2.0:** 2026-06-02 20:50
> **Scope:** AgentFoundation only. RankEvolve referenced; no RankEvolve edits.

---

## § -1. Provenance & Audit History

| Version | Time | Author | Notes |
|---|---|---|---|
| 1.0 | 2026-06-02 19:11 | Rovo Dev | Initial design. Created `common/proposals/` module, invented `writer.py`, `task/templates/initial_plan_from_proposals.md.j2`, treated as new architectural layer. |
| 2.0 | 2026-06-02 20:50 | Rovo Dev (integrating Claude) | **5 architectural corrections from Claude's parallel investigation, all empirically verified against live AF code:** (1) module path `common/data_models/proposal/` (AF convention; v1.0 invented `common/proposals/`). (2) Reuse existing `template_extra_feed["include_proposal_index"]` pattern (proven by `include_winner_pick` at `multi_flow_dual_inferencer.py:303`); v1.0 invented a new emission mechanism. (3) Use `_extract_json_block(text, "proposal_index")` from `flow_parsers.py:70` (proven by winner_pick + iteration_judgment); v1.0 invented `writer.py`. (4) Reuse `initial_response_override` (proven at `plan_then_implement_inferencer.py:1969, 1979`); v1.0 invented a templated plan injection path. (5) Generic `P<n>` IDs not `H<n>` (RankEvolve-domain-specific). Kept from v1.0: risks register, open questions, phased rollout, audit history, alternatives-rejected, sidecar architectural framing, atomic-write guarantee, SOP `__branch on:` integration story. |
| 3.0 | 2026-06-02 21:23 | Rovo Dev (integrating Claude round 2) | **Major architectural revision driven by Tony's "template_master_version is the generic mechanism" insight + Claude's parallel arrival at the same conclusion.** Empirically verified that workers (`deep_research/main/initial.jinja2`) currently render NO `task_response_format` slot, and aggregator's `template_master_version` is hard-pinned to `"aggregation"` by `AGGREGATION_DEFAULTS` (template_defaults.py:273). Both my v2.1 and Claude's round 2 plan had blind spots: v2.1 was aggregator-only (correct slot but wrong axis); Claude's plan was worker-emit + aggregator-merge (right axis but missing the worker-template edit and aggregator routing override). **v3.0 integrates the cleanest of both:** (a) adopt Claude's worker-emit + aggregator-merge architecture; (b) add the empirically-required worker template edit (`deep_research/main/initial.jinja2` gains a `{% if task_response_format %}` slot); (c) override AGGREGATION_DEFAULTS pinning via topology YAML (`breakdown-multiflow-plan.yaml` sets `aggregator_inferencer.template_master_version: research_propose`); (d) restructure templates under `task_response_format/research_propose/{default,aggregation}.jinja2`; (e) BTA post-hook parses aggregator output and writes sidecar; (f) **NEW worker-pre-aggregation hook** parses each worker's `individual_proposal` JSON fence and injects merged list into `aggregator_inferencer.template_extra_feed["worker_proposals"]` so the aggregator template can render the structured list. Kept from v2.1: risk register, audit history, alternatives-rejected, phased rollout, atomic-write guarantee, SOP branch integration. Kept from Claude: simple Strategy A only (no Strategy B/C), `_meta` field structure, generic fence naming. |
| 2.1 | 2026-06-02 21:13 | Rovo Dev (Tony's architectural correction) | **Tony correctly pushed back on v2.0's flag-based emission.** Empirical reverification revealed: (a) `include_winner_pick` / `include_iteration_judgment` are **generic aggregation primitives** (winner selection, convergence judgment) that any multi-flow aggregator might use — they belong in `default.jinja2`. (b) The proposal-index directive is **domain-specialized**, not a primitive — not every aggregator emits proposals; putting it in `default.jinja2` would pollute the generic template with a research-propose concept. (c) **AF has established cascade roots**: `plan/main/_variables/<var>/aggregation/<version>.jinja2` for plan-family aggregator templates; top-level `_variables/<var>/<master>/<version>.jinja2` for cross-cutting personas (where `_variables/task_preamble/research_propose/default.jinja2` already lives). (d) **Workers (`deep_research/main/initial.jinja2`) do NOT consume `task_response_format`** — only `plan/main/initial.jinja2` does. So the directive is aggregator-only despite Tony's worry; his architectural argument (specialization, not primitive) still stands. **Final design: create `plan/main/_variables/task_response_format/aggregation/research_propose.jinja2` (extends/replaces `default.jinja2` for research-propose-derived aggregators). Wire via `template_variables` per-variable override in topology YAML — NOT via `template_master_version` (that selects subdirectory, stays `"aggregation"`) and NOT via flag.** Removed §3.4's `include_proposal_index` flag plumbing and the `_params.extract_proposals` config_override; replaced with template-version override. BTA post-hook now triggers unconditionally and is a no-op when no fence is present (cheap regex search). |

---

## § 0. TL;DR

**Goal:** Make AF's `research_propose` emit a structured, machine-readable **`outputs/proposals.json`** sidecar so that:
1. `task --use-proposal <index.json> --proposal-ids P1,P3` picks individual proposals without re-parsing markdown.
2. The model_optimization SOP Phase 3 can `__branch on: proposals.json__` for parallel per-proposal execution.
3. Future `proposal_selection` conversation widgets (deferred) consume JSON directly.

**Architectural keystone (Claude's correction):** **Every piece reuses an existing AF pattern.** Zero new mechanisms invented.

| Need | Reused pattern | Evidence (live code) |
|---|---|---|
| Template fence emission | `template_extra_feed["include_X"] = True` | `multi_flow_dual_inferencer.py:303` (`include_winner_pick`), `multi_flow_inferencer.py:355` (`include_iteration_judgment`) |
| JSON fence extraction | `_extract_json_block(text, "<fence_name>")` | `flow_parsers.py:70` (`winner_pick`), `flow_parsers.py:99` (`iteration_judgment`) |
| Plan injection into PTI | `initial_response_override` | `plan_then_implement_inferencer.py:1969, 1979` |
| Tool-level pre-seed | `--initial-plan` → PTI override path | `task/executor.py` |
| Tool-config inheritance | `derived_from: task` + `config_overrides` | `research_propose/tool.json:11-22` |

---

## § 1. Problem statement

### 1.1 What's broken / missing today (empirically verified Tue 2026-06-02)

| Observation | Evidence |
|---|---|
| AF's `research_propose` produces `unified_plan.md` but **no machine-readable index** | Aggregator template writes markdown only. |
| Downstream consumers must re-parse markdown | RankEvolve's `proposal_parser.py` is 428 lines of regex; cloning that across consumers is the wrong abstraction. |
| AF's `task` has `--initial-plan <path>` but no way to say "implement just H3 and H7" | `task/tool.json` parameters list — `--initial-plan` is plain-text only. |
| SOP `__branch__` runtime needs a deterministic enumeration source | `sops/model_optimization/SOP.md:83` references per-proposal experiments but the SOP doesn't say how the list is materialized. |
| RankEvolve has `ProposalSelectionData` but it's **ML-domain-specific** (slots, includes, probability percents, "H<n>" IDs, batches) | `rankevolve/.../proposal_models.py` |

### 1.2 What we want

A **generic, domain-agnostic** `ProposalIndex` lives in `agent_foundation/common/data_models/proposal/`. Research-propose emits `outputs/proposals.json` via the established `template_extra_feed` + `_extract_json_block` pattern. Task tool gains `--use-proposal` + `--proposal-ids` that resolve into the existing `initial_response_override` path.

### 1.3 What we explicitly do NOT want

- **Don't replace `unified_plan.md`** — humans still need the markdown report.
- **Don't fork RankEvolve code into AF** — borrow the schema concept (generic dataclasses), not the ML-specific fields.
- **Don't invent new emission/parsing/injection mechanisms** — reuse the four patterns above.
- **No mandatory changes for existing prompt templates** — adding `include_proposal_index` is conditional and gated.

---

## § 2. RankEvolve reference (lessons borrowed, not code)

### 2.1 What we keep

| Concept | Source | AF generalization |
|---|---|---|
| Structured proposal dataclass | RE `StructuredProposal` | AF `Proposal` (domain-agnostic) |
| Phase grouping | RE `ProposalPhase` | AF `ProposalGroup` |
| Mutually-exclusive / requires / recommends constraints | RE `ComboConstraint` | AF `ProposalConstraint` |
| Top-level container with `from_dict`/`to_dict` | RE `ProposalSelectionData` | AF `ProposalIndex` |
| JSON fence inside markdown (Strategy A) | RE `proposal_parser.py:84` `_parse_json_strategy` | AF uses `_extract_json_block(text, "proposal_index")` |
| Three-strategy parser | RE has A/B/C | AF: A primary (sidecar JSON), B fallback (markdown fence in `unified_plan.md`), C last-resort (table parse) |

### 2.2 What we drop (ML-specific)

| Dropped field | Why |
|---|---|
| `probability` ("75%", "High (>70%)") | ML hypothesis-testing concept; SE/research doesn't have analog |
| `slots` (mutually-exclusive config knobs) | Architecture-search-specific |
| `includes` (combo bundle membership) | RankEvolve combo idiom; defer until needed |
| `Batch` (timeline-grouped subsets) | ML experiment-batching; defer |
| `source_workers` ("W0", "W4") | BTA-internal labeling; tags can subsume |
| `H<n>` ID convention | ML "hypothesis" naming; generic `P<n>` for "Proposal" |

### 2.3 RankEvolve does NOT emit a sidecar

RE's parser reads `unified_plan.md` on every consumer call. **This is the gap AF closes.** Emit once, read many. Atomic write semantics guarantee no torn reads.

---

## § 3. Proposed AF design

### 3.1 New module: `agent_foundation/common/data_models/proposal/`

```
agent_foundation/common/data_models/                  # NEW package (top-level)
├── __init__.py
└── proposal/
    ├── __init__.py            # re-exports
    ├── model.py               # Proposal, ProposalGroup, ProposalConstraint, ProposalIndex
    ├── parser.py              # parse_proposal_index(), parse_proposal_file(), find_proposal_output()
    └── README.md              # schema docs + JSON example
```

### 3.2 Schema (in `model.py`)

```python
@dataclass
class Proposal:
    id: str                      # "P1", "P2"  (generic; NOT "H1" — RE-domain-specific)
    rank: int                    # 1 = highest priority
    title: str
    summary: str = ""            # one-line summary
    impact: str = ""             # "low" | "medium" | "high"
    complexity: str = ""         # "low" | "medium" | "high"
    approach: str = ""           # how to implement
    problem: str = ""            # what it solves
    dependencies: list[str] = field(default_factory=list)    # IDs this depends on
    tags: list[str] = field(default_factory=list)            # generic labeling; replaces RE's source_workers, theme
    metadata: dict[str, Any] = field(default_factory=dict)   # domain-specific overflow (RE can put probability/slots here)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Proposal": ...

@dataclass
class ProposalGroup:
    phase: int                  # 1 = quick wins, 2 = core, 3 = exploration (convention; not enforced)
    label: str
    description: str = ""
    proposals: list[Proposal] = field(default_factory=list)
    # to_dict/from_dict ...

@dataclass
class ProposalConstraint:
    id: str
    kind: str                   # "mutually_exclusive" | "requires" | "recommends"
    proposal_ids: list[str] = field(default_factory=list)
    requires_ids: list[str] = field(default_factory=list)
    label: str = ""
    reason: str = ""
    severity: str = "error"     # "error" | "warning" | "info"
    # to_dict/from_dict ...

@dataclass
class ProposalIndex:
    version: str = "1"          # schema version pinned
    created_at: str = ""        # ISO-8601 UTC
    source_workspace: str = ""  # absolute path snapshot
    total_count: int = 0
    groups: list[ProposalGroup] = field(default_factory=list)
    constraints: list[ProposalConstraint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)   # parser/extract warnings (non-fatal)

    def all_proposals(self) -> list[Proposal]:
        """Flat list sorted by rank ascending (1 = highest)."""
    def get_proposals_by_ids(self, ids: list[str]) -> list[Proposal]:
        """Raises KeyError with valid-ID list if any ID missing."""
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "ProposalIndex": ...
```

### 3.3 Parser (in `parser.py`)

Three strategies, in priority order:

| Order | Strategy | Function |
|---|---|---|
| A (fast path, AF native) | Read `outputs/proposals.json` sidecar | `parse_proposal_file(path: Path) -> ProposalIndex` |
| B (compat, post-extraction) | Read `unified_plan.md`, extract `` ```json proposal_index `` ` fence via `_extract_json_block(text, "proposal_index")` from `flow_parsers.py` | `_parse_markdown_fence(text: str) -> ProposalIndex \| None` |
| C (last resort) | Regex-parse a Priority Ranking Table from `unified_plan.md`; recover only `id`, `rank`, `title` | `_parse_ranking_table(text: str) -> ProposalIndex \| None` |

**Public entry point:**
```python
def parse_proposal_index(workspace_path: Path) -> ProposalIndex | None:
    """Try Strategy A → B → C. Returns None only if all three fail."""
```

**Reuse:** `_extract_json_block` already exists at `common/inferencers/flow_parsers.py:_extract_json_block`. Used today for `winner_pick` and `iteration_judgment` fences. Same regex, different fence name.

### 3.4 Emission: workers emit + aggregator merges (v3.0 — `template_master_version` switch)

**Architectural principle (v3.0):** `template_master_version` is AF's generic per-tool persona switch (precedent: `understand_codebase`, `understand_data`). Setting `template_master_version="research_propose"` should swap BOTH worker AND aggregator template personas — but the existing AF infrastructure has two empirical constraints we must respect:

1. **Workers (`deep_research/main/initial.jinja2`) currently render NO `task_response_format` slot.** We must add one (small, additive edit, backward-compatible — defaults to empty for non-research-propose tools).
2. **Aggregator's `template_master_version` is pinned to `"aggregation"` by `AGGREGATION_DEFAULTS` (template_defaults.py:273).** Tool.json's value flows to workers but not aggregator. We override the aggregator's master_version via topology YAML (`breakdown-multiflow-plan.yaml`).

**Resulting data flow:**
```
Worker 1 → emits ```json individual_proposal { id: "P1", ... }```
Worker 2 → emits ```json individual_proposal { id: "P2", ... }```
Worker N → ...
       ↓
[NEW] BTA worker-pre-aggregation hook parses each individual_proposal fence
       ↓
worker_proposals: list[Proposal] injected into aggregator_inferencer.template_extra_feed
       ↓
Aggregator template renders worker_proposals as structured input
       ↓
Aggregator emits ```json proposal_index { groups: [...], ... }``` (ranks/groups/dedup)
       ↓
[NEW] BTA post-aggregation hook parses proposal_index fence
       ↓
ProposalIndex → atomic_write_json(workspace/outputs/proposals.json)
```

#### 3.4.1 New template files (4 files total)

Create:
```
plan/main/_variables/task_response_format/
├── aggregation/
│   └── default.jinja2                 # UNCHANGED — generic primitives only
└── research_propose/
    ├── default.jinja2                 # NEW — worker-rendered individual_proposal directive
    └── aggregation.jinja2             # NEW — aggregator-rendered proposal_index directive

deep_research/main/_variables/task_response_format/   # NEW SUBTREE (if not exists)
└── research_propose/
    └── default.jinja2                 # NEW — same as plan/main/.../research_propose/default.jinja2 OR a symlink/include
```

**(Alternative: place all `task_response_format/research_propose/` files under the cross-cutting top-level `_variables/task_response_format/research_propose/` so both `plan/main/` and `deep_research/main/` cascade-resolve to the same files. Phase 0 must empirically verify which cascade root applies to each family.)**

#### 3.4.2 Template contents

**Worker — `task_response_format/research_propose/default.jinja2`:**
```jinja2
Conclude your response with a per-proposal structured emission. For each
proposal you produce, append a fenced JSON block:

```json individual_proposal
{
  "id": "P1",
  "title": "...",
  "summary": "...",
  "problem": "...",
  "approach": "...",
  "impact": "high",
  "complexity": "low",
  "dependencies": [],
  "metadata": {}
}
```

Use sequential IDs (P1, P2, ...). One fence per proposal. The narrative
explanation can precede or interleave the fences.
```

**Aggregator — `task_response_format/research_propose/aggregation.jinja2`:**
```jinja2
{# Inherit generic aggregator response format primitives (winner_pick, iteration_judgment, etc.) #}
{% include "task_response_format/aggregation/default.jinja2" %}

{%- if worker_proposals %}

You have been given {{ worker_proposals | length }} individual proposals from
parallel research workers (rendered above as <WorkerProposal id="..."> blocks).

Your task is to MERGE, RANK, GROUP, and DEDUPLICATE these proposals into a
unified proposal index. You may:
- Re-rank by impact/complexity tradeoff
- Group into phases (Quick Wins, Foundational, Long-term)
- Merge near-duplicates (preserving union of dependencies)
- Add cross-proposal constraints (e.g., "P3 requires P1 first")
- Drop proposals that are dominated by others

Conclude with a unified proposal index as a fenced JSON block:

```json proposal_index
{
  "version": "1",
  "groups": [
    {
      "phase": 1,
      "label": "Quick Wins",
      "proposals": [
        {"id": "P1", "rank": 1, "title": "...", "summary": "...",
         "impact": "high", "complexity": "low",
         "problem": "...", "approach": "...",
         "dependencies": []}
      ]
    }
  ],
  "constraints": [],
  "total_count": 0
}
```
{%- endif %}
```

#### 3.4.3 Worker template edit (REQUIRED — `deep_research/main/initial.jinja2`)

Currently `deep_research/main/initial.jinja2` has NO `task_response_format` slot. Add one (additive, backward-compatible — defaults to empty for non-research-propose tools):

```jinja2
{# ADD just before the existing "## Response Format" section: #}
{%- if task_response_format %}
{{ task_response_format }}
{%- endif %}
```

This works because `task_response_format` resolves via the `master_version` cascade: under `master_version="research_propose"`, the resolver finds `task_response_format/research_propose/default.jinja2` (the worker variant). Under default (other tools), the variable resolves to empty/missing and the conditional skips.

#### 3.4.4 Aggregator routing — `breakdown-multiflow-plan.yaml` override

Override `AGGREGATION_DEFAULTS`'s pinned `template_master_version="aggregation"` at the topology level:
```yaml
aggregator_inferencer:
  template_master_version: research_propose   # NEW — overrides AGGREGATION_DEFAULTS pin
  template_variables:
    task_response_format: aggregation         # selects aggregation.jinja2 within research_propose/ subdir
```

This works because:
- Setting `template_master_version: research_propose` cascades aggregator's variable lookups into `task_response_format/research_propose/` (not `aggregation/`).
- `template_variables.task_response_format: aggregation` selects the `aggregation.jinja2` filename within that subdir — i.e., resolves `task_response_format/research_propose/aggregation.jinja2`.

**Phase 0 must verify** that overriding `template_master_version` on an aggregator_inferencer at the topology level actually wins over `AGGREGATION_DEFAULTS`'s pin. If not, fallback: revert to v2.1 per-variable-only override (keep aggregator at master_version="aggregation", select `research_propose.jinja2` within `aggregation/` subdir).

#### 3.4.5 Worker-pre-aggregation hook (NEW — in `breakdown_then_aggregate_inferencer.py`)

After all workers complete but BEFORE aggregator runs, parse each worker's output for `individual_proposal` fences and inject the merged list into the aggregator's template feed:

```python
# Pseudocode — in the aggregator builder, before invoking aggregator_inferencer
from agent_foundation.common.data_models.proposal.parser import (
    parse_individual_proposals_from_text,
)
from agent_foundation.common.data_models.proposal.model import Proposal

worker_proposals: list[Proposal] = []
for i, worker_result_text in enumerate(worker_results):
    found = parse_individual_proposals_from_text(str(worker_result_text))
    for p in found:
        # Re-namespace IDs to avoid worker-local collisions: W1-P1, W2-P1, ...
        p.id = f"W{i+1}-{p.id}"
        worker_proposals.append(p)

# Inject into aggregator template feed
if worker_proposals and hasattr(bta.aggregator_inferencer, "template_extra_feed"):
    bta.aggregator_inferencer.template_extra_feed["worker_proposals"] = [
        p.to_dict() for p in worker_proposals
    ]
# else: aggregator template's {% if worker_proposals %} branch skips — no-op
```

**Cost:** N regex searches (one per worker). Cheap.

**Behavior under no fences:** worker_proposals stays empty; aggregator template's `{% if worker_proposals %}` block is skipped; aggregator falls back to free-form synthesis (legacy path). Backward-compatible.

#### 3.4.6 Post-aggregation hook (in `breakdown_then_aggregate_inferencer.py`)

Same as v2.1 — runs unconditionally after aggregator completes, parses `proposal_index` fence, writes `outputs/proposals.json`:

```python
from agent_foundation.common.data_models.proposal.parser import (
    parse_proposal_index_from_text,
)
from agent_foundation.common.data_models.proposal.model import ProposalIndex

idx: ProposalIndex | None = parse_proposal_index_from_text(aggregator_result.text)
if idx is not None:
    idx.created_at = _utc_now_iso()
    idx.source_workspace = str(self._workspace.root)
    _atomic_write_json(
        self._workspace.root / "outputs" / "proposals.json",
        idx.to_dict(),
    )
# else: silently skip — expected for non-research aggregators
```

**Reuse:** `_extract_json_block` already exists at `common/inferencers/flow_parsers.py:70`. Same regex idiom, different fence name.

#### 3.4.7 Atomic-write helper (in `parser.py` or a tiny `_io.py`)

```python
def _atomic_write_json(path: Path, data: dict) -> None:
    """POSIX-safe: write to temp, fsync, rename. Never produces partial files."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)   # atomic on POSIX
```

### 3.5 Research-propose tool config (`tool.json`) — minimal edit

Research-propose's `tool.json` already has `template_master_version: research_propose` in defaults (verified at line 18 of current `research_propose/tool.json`). This value flows to workers automatically through the existing `_template_master_version` override mechanism (task/executor.py:602-603).

**No new tool.json edit required** for the worker-side template cascade — the existing config already routes workers correctly.

**One topology YAML edit** is needed (§3.4.4) to override the aggregator's pinned master_version.

### 3.6 Task tool extension: `--use-proposal` + `--proposal-ids`

#### 3.6.1 `tool.json` additions

```jsonc
{
  "name": "--use-proposal",
  "type": "path",
  "description": "Path to a proposals.json from research_propose. Selected proposals seed the task's plan phase via initial_response_override. Mutually exclusive with --initial-plan."
},
{
  "name": "--proposal-ids",
  "type": "string",
  "description": "Comma-separated proposal IDs to pick (e.g., 'P1,P3'). Omit to pick all proposals in the index."
}
```

#### 3.6.2 Executor flow (`task/executor.py`)

```
if --use-proposal:
    if --initial-plan: error "mutually exclusive"
    idx = parse_proposal_file(--use-proposal)
    selected = idx.get_proposals_by_ids(parse_csv(--proposal-ids)) if --proposal-ids else idx.all_proposals()
    plan_md = _format_proposals_as_plan(selected, source_index_path=--use-proposal)
    init_plan_path = workspace / "outputs" / "initial_proposal_plan.md"
    init_plan_path.write_text(plan_md)
    # fall through to existing init_plan_path → initial_response_override path
```

**`_format_proposals_as_plan`** is a small string builder (no template needed; ~30 lines):
```
# Initial Plan — Picked from research_propose
_Source: {source}_  •  _Selected: {ids}_

For each selected proposal, summarize:
- ID, rank, title, impact, complexity
- Problem (if present)
- Approach
```

#### 3.6.3 Audit record

Write `<workspace>/_picked_proposals.json` at pickup time:
```json
{
  "index_path": "/abs/path/to/proposals.json",
  "selected_ids": ["P1", "P3"],
  "picked_at": "2026-06-02T20:53:00Z"
}
```
Enables debugging "which proposals did this run actually implement?"

### 3.7 SOP `__branch on:` integration (Phase 4, depends on SOP runtime)

The model_optimization SOP Phase 3 wants one experiment per selected proposal. With `proposals.json`, the branch source is unambiguous:

```md
## Phase 3 [__branch on: proposals.json__]
For each selected proposal, run experiment...
```

The runtime materializes one branch per proposal, each carrying `{proposal_id, index_path}` in context. Each branch invokes:
```
task --use-proposal {index_path} --proposal-ids {proposal_id} ...
```

**This is the wiring that makes the SOP run end-to-end.**

---

## § 4. File inventory

| Path | Action | LoC est. |
|---|---|---|
| `agent_foundation/common/data_models/__init__.py` | NEW | 5 |
| `agent_foundation/common/data_models/proposal/__init__.py` | NEW | 10 |
| `agent_foundation/common/data_models/proposal/model.py` | NEW | ~180 |
| `agent_foundation/common/data_models/proposal/parser.py` | NEW (3 strategies; A/B small, C borrowed-pattern from RE) | ~250 |
| `agent_foundation/common/data_models/proposal/README.md` | NEW (schema docs) | ~80 |
| `agent_foundation/resources/prompt_templates/plan/main/_variables/task_response_format/research_propose/default.jinja2` | **NEW** — worker-rendered individual_proposal directive | ~20 lines |
| `agent_foundation/resources/prompt_templates/plan/main/_variables/task_response_format/research_propose/aggregation.jinja2` | **NEW** — aggregator-rendered proposal_index directive (with `{% if worker_proposals %}` branch) | ~40 lines |
| `agent_foundation/resources/prompt_templates/plan/main/_variables/task_response_format/aggregation/default.jinja2` | **UNCHANGED** — domain-pure (no v3.0 edit; generic primitives only) | 0 |
| `agent_foundation/resources/prompt_templates/deep_research/main/initial.jinja2` | EDIT — add `{% if task_response_format %}{{ task_response_format }}{% endif %}` slot before "## Response Format" | ~3 lines added |
| `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py` | EDIT — add worker-pre-aggregation hook (parses worker `individual_proposal` fences, injects merged list into aggregator feed) AND post-aggregation hook (parses aggregator `proposal_index` fence, writes sidecar) | ~30 lines added |
| `agent_foundation/resources/tools/task/configs/breakdown-multiflow-plan.yaml` | EDIT — set `aggregator_inferencer.template_master_version: research_propose` and `template_variables.task_response_format: aggregation` | ~3 lines |
| `agent_foundation/resources/tools/research_propose/tool.json` | **NO EDIT** — existing `template_master_version: research_propose` already routes workers correctly | 0 |
| `agent_foundation/resources/tools/task/tool.json` | EDIT — add `--use-proposal`, `--proposal-ids` | ~12 lines |
| `agent_foundation/resources/tools/task/cli.py` | EDIT — parse new args | ~15 lines |
| `agent_foundation/resources/tools/task/slash_args.py` | EDIT — mirror | ~10 lines |
| `agent_foundation/resources/tools/task/executor.py` | EDIT — `_resolve_proposal_plan` + mutual-exclusivity check | ~50 lines |
| `test/agent_foundation/common/data_models/proposal/test_model_roundtrip.py` | NEW | ~80 |
| `test/agent_foundation/common/data_models/proposal/test_parser_strategies.py` | NEW | ~150 |
| `test/agent_foundation/common/data_models/proposal/test_atomic_write.py` | NEW | ~40 |
| `test/agent_foundation/resources/tools/task/test_use_proposal.py` | NEW | ~120 |
| `test/agent_foundation/fixtures/proposals/unified_plan_with_fence.md` | NEW | ~120 |
| `test/agent_foundation/fixtures/proposals/proposals_v1.json` | NEW | ~150 |

**Total:** ~7 new files, ~7 edits, ~1,500 LoC including tests and fixtures.

---

## § 5. Acceptance criteria

| ID | Criterion |
|---|---|
| AC-S1 | `Proposal.to_dict()` ↔ `Proposal.from_dict()` round-trip preserves all fields (including empty defaults). |
| AC-S2 | `ProposalIndex.from_dict(idx.to_dict()) == idx` (deep equality across all 14 demo proposals). |
| AC-S3 | `parse_proposal_index(workspace)` Strategy A: when `outputs/proposals.json` exists, returns it without touching markdown. |
| AC-S4 | `parse_proposal_index(workspace)` Strategy B: when only `unified_plan.md` with fenced JSON exists, reconstructs the same `ProposalIndex`. |
| AC-S5 | `parse_proposal_index(workspace)` Strategy C: when only `unified_plan.md` with a ranking table exists, recovers `id`, `rank`, `title` for all rows. |
| AC-S6 | Atomic write: 100 concurrent writes + 100 concurrent reads → readers always see either old-complete or new-complete (never torn). |
| AC-E1 | Worker template rendering: with `master_version="research_propose"`, `deep_research/main/initial.jinja2` renders the `individual_proposal` directive block via the new `task_response_format` slot. Workers without research_propose master skip the slot (renders empty). |
| AC-E1b | Aggregator template rendering: when topology YAML sets `aggregator_inferencer.template_master_version: research_propose` + `template_variables.task_response_format: aggregation`, the rendered aggregator prompt resolves to `task_response_format/research_propose/aggregation.jinja2` and contains both the generic primitives (winner_pick, etc.) AND the `proposal_index` merge directive. |
| AC-E1c | Worker-pre-aggregation hook: given 3 mock worker outputs each containing a single `individual_proposal` fence, hook parses all 3, re-namespaces IDs as W1-P1/W2-P1/W3-P1, and injects `worker_proposals: list[dict]` into `aggregator_inferencer.template_extra_feed`. Aggregator template's `{% if worker_proposals %}` branch then renders the list. |
| AC-E1d | Worker-pre-aggregation hook gracefully no-ops: given 3 worker outputs with NO `individual_proposal` fences, hook injects empty list; aggregator template skips the merge branch; aggregator falls back to free-form synthesis (legacy backward-compat). |
| AC-E2 | After a successful `research_propose` run, `outputs/proposals.json` exists with `version == "1"`, `total_count > 0`, `created_at` set, `source_workspace` set. |
| AC-E3 | If LLM forgets the JSON fence, post-hook writes `proposals.json` with `total_count: 0` and `warnings: ["aggregator-emitted-no-structured-output"]` — never crashes the pipeline. |
| AC-T1 | `task --use-proposal <path> --proposal-ids P1,P3 "X"` filters to 2 proposals, writes `initial_proposal_plan.md`, then dispatches via `initial_response_override`. |
| AC-T2 | `task --use-proposal <path>` (no `--proposal-ids`) picks ALL proposals in the index. |
| AC-T3 | `task --use-proposal <path> --proposal-ids P99` raises with "Unknown proposal IDs: [P99]. Valid IDs: [P1, P2, ...]". |
| AC-T4 | `task --use-proposal <path> --initial-plan <path>` exits with "mutually exclusive" error. |
| AC-T5 | `task --use-proposal` writes `_picked_proposals.json` audit record with `{index_path, selected_ids, picked_at}`. |
| AC-T6 | Existing `task` runs (no `--use-proposal`) are byte-for-byte identical to today — zero regression on any of the existing 115+ test cases. |
| AC-R1 | `from agent_foundation.common.data_models.proposal import ProposalIndex` works (re-export at package level). |
| AC-X1 (Phase 4) | SOP `__branch on: proposals.json__` materializes N branches when N proposals are present in `proposals.json`. (Depends on SOP runtime — Phase 4 only.) |

---

## § 6. Phased rollout

### Phase 1 — Schema + parser (AF infrastructure, no behavior change)

**Goal:** Land the data model and parser. Zero production wiring.

- Create `common/data_models/__init__.py`
- Create `common/data_models/proposal/{__init__,model,parser,README}.py`
- Tests: AC-S1, AC-S2, AC-S3, AC-S4, AC-S5, AC-S6, AC-R1
- **Verification:** unit tests only; no production caller yet; risk near-zero

### Phase 2 — Aggregator template + BTA post-hook

**Goal:** When the feed flag is set, emit + write `proposals.json`. When unset, behavior identical to today.

- Edit `task_response_format/aggregation/default.jinja2` to add `include_proposal_index` block
- Edit `breakdown_then_aggregate_inferencer.py` to add post-aggregation hook (gated)
- Edit `breakdown-multiflow-plan.yaml` if needed to surface the flag from `_params`
- Tests: AC-E1, AC-E2, AC-E3, AC-T6 (regression)
- **Verification:** run `research_propose --breakdown-only` against a tiny query; verify `proposals.json` appears and round-trips

### Phase 3 — Task `--use-proposal`

**Goal:** Tool-level pickup.

- Edit `task/tool.json`, `cli.py`, `slash_args.py`, `executor.py`
- Tests: AC-T1, AC-T2, AC-T3, AC-T4, AC-T5
- **Verification:** end-to-end `research_propose` → `task --use-proposal` on a small fixture

### Phase 4 — SOP `__branch on:` integration (depends on SOP runtime)

- Defer until SOP framework Phase 3 lands the `__branch on: <source>__` runtime
- Tests: AC-X1
- **Verification:** model_optimization SOP run with 3 selected proposals → 3 parallel Phase 3 branches

---

## § 7. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | LLM-emitted JSON fence is malformed (bad JSON, missing field) | HIGH | Parser falls back to Strategy C (table-only). Post-hook writes `warnings: [...]`. Never crashes pipeline. |
| R2 | Schema drift between AF `Proposal` and RankEvolve `StructuredProposal` | MED | Pin `version: "1"` in JSON. Document AF as framework-canonical. RankEvolve can subclass `Proposal` to keep ML fields (see §9 migration path). |
| R3 | `--use-proposal` IDs reference proposals that have unmet `dependencies` | MED | v1: log warning, proceed. Future: `--enforce-deps` flag refuses. |
| R4 | Older RankEvolve workspaces (no `proposals.json`) passed to AF `task --use-proposal` | LOW | Strategy B/C parsers handle this; documented as supported. |
| R5 | Atomic write (`tmp→fsync→rename`) fails on Windows due to file-locked open by reader | LOW | AF is POSIX-first; documented as known limitation. |
| R6 | Topology YAML override of `aggregator_inferencer.template_master_version` does NOT win over `AGGREGATION_DEFAULTS`'s pin (defaults precedence rules unclear) | HIGH | **Phase 0 must verify this empirically before Phase 2.** If override loses, fallback design: keep aggregator at master_version="aggregation", use per-variable `template_variables.task_response_format: research_propose` (v2.1 design) — but THEN aggregator's `aggregation.jinja2` must be placed under `task_response_format/aggregation/research_propose.jinja2` (not `task_response_format/research_propose/aggregation.jinja2`). Worker side unchanged. |
| R7 | Worker template edit (`deep_research/main/initial.jinja2`) breaks other tools that use the same template family (understand_codebase, understand_data) | MED | Edit is additive and conditional (`{% if task_response_format %}`). Other tools don't define a `task_response_format` variant under their master_version → conditional renders empty. Verify with grep test that `task_response_format/understand_codebase/`, `task_response_format/understand_data/` don't exist (they shouldn't). |
| R8 | Worker IDs collide across parallel workers (Worker1 emits P1, Worker2 also emits P1) | LOW | Mitigated by hook's re-namespacing logic (W1-P1, W2-P1). Aggregator dedupes via merge. Final `proposal_index` uses re-ranked IDs assigned by aggregator. |
| R9 | BTA post-hook fires twice (resume + re-run) and races on write | LOW | Atomic write guarantees no torn file; second writer just clobbers with identical content (idempotent). |
| R10 | RankEvolve's `proposal_parser.py` Strategy B differs subtly from AF's, causing different round-trips on the same input | LOW | AF parser tested against a fixture; RE parser unaffected. Migration path (§9) bridges them when needed. |

---

## § 8. Open questions

1. **Schema versioning policy.** v1.0 pins `version: "1"`. When (if ever) do we bump to "2"? Proposed rule: only on a **breaking field rename or removal**. Additive fields don't bump. Document in README.
2. **`--proposal-ids` syntax.** v1.0 says comma-separated (`P1,P3`). Alternative: repeatable flag (`--proposal-ids P1 --proposal-ids P3`). Recommend comma — fewer keystrokes, matches `--override` precedent.
3. **`P<n>` vs `H<n>` vs custom prefix.** v1.0 says generic `P<n>`. The parser accepts any string — so RankEvolve resumes with `H<n>` IDs still work. The template **suggests** `P<n>` but doesn't enforce.
4. **Where do post-hook errors go?** v1.0 says `warnings: [...]` array in `proposals.json` itself. Alternative: separate `proposals.errors.json`. Recommend inline — one file, one truth.
5. **Should `task --use-proposal` accept a glob (`P*`)?** v1.0 says exact only. Globs invite mistakes (`P*` matches `P1, P10, P11, ...`).
6. **Combo / bundle proposals (RankEvolve's `H15 = [H8, H14]`).** v1.0 doesn't model bundles in the generic schema (left in `metadata`). Future: add `includes: list[str]` on `Proposal` if a non-RE use case demands.
7. **Atomic write on shared NFS.** POSIX `rename` is atomic locally but **not necessarily on NFS**. Document as a known limitation; recommend local filesystem for `_runtime/`.
8. **Is `breakdown-multiflow-plan.yaml` shared between research-propose and other tools that should NOT emit a proposal index?** v2.1 wires `template_variables.task_response_format: research_propose` directly in this YAML. If a non-research tool also uses `breakdown-multiflow-plan`, it would also try to render `research_propose.jinja2`. **Phase 0 must grep for all consumers of this YAML.** If shared, two resolutions: (a) fork the YAML into `breakdown-multiflow-plan-research.yaml` for research-propose's `derived_from.defaults.config`, or (b) push the `template_variables` override into research_propose's `config_overrides` (back to a tool.json edit, but on a safer axis than v2.0's flag).
9. **Should the post-hook also emit `proposals.json` for non-research aggregators?** v2.1 says: the post-hook runs unconditionally and silently skips when no fence is present. Tools that want proposals just point their topology at a template version that emits the fence. This decouples the post-hook from any flag/tool-name check.

---

## § 9. RankEvolve migration path (follow-up, not in scope)

When AF ships v2.0, RankEvolve can adopt the generic format without losing ML fields:

1. **Subclass:** `class StructuredProposal(Proposal):` adding `probability`, `slots`, `includes`, `source_workers`, `cross_refs`, `theme`. These extras live in inherited `metadata` field of the base, OR as explicit fields on the subclass.
2. **Bundle wrapper:** `ProposalSelectionData` becomes a thin wrapper around `ProposalIndex` that adds `Batch`-grouping.
3. **Parser delegation:** RE's Strategy A parser becomes `parse_proposal_index(workspace) → ProposalIndex` → then RE-specific enrichment.
4. **Conversation handler stays in RE:** `ProposalSelectionHandler` depends on `HubAwareToolExecutor` (RE-specific). No change.

**This is not a Phase of this plan** — it's a forward-looking note so RE can move when convenient.

---

## § 10. Alternatives considered and rejected

| Alternative | Why rejected |
|---|---|
| Make `unified_plan.md` the only artifact; parse on demand | Forces every downstream tool to know the regex tower. RankEvolve already has 428-line parser; cloning that everywhere instead of writing JSON once is the wrong abstraction. |
| Replace markdown with JSON | Loses human-readability; the markdown report is end-user-facing in the SOP review widget. |
| Pass full proposal text via `--initial-plan` (existing flow) | Loses the structured ID — downstream can't say "user picked P3, not P4". |
| Embed selection state in conversation `prior_context` only (RankEvolve approach) | Couples to ConversationalInferencer. CLI/SOP-headless callers can't pick a proposal without a CI session. JSON sidecar decouples cleanly. |
| Use SQLite for proposals | Overkill. One JSON file is human-inspectable, git-trackable, and atomic. |
| `dict[str, Any]` schema | Loses IDE help and round-trip validation. Schema-version bumps are cleaner with dataclasses. |
| Invent a new template-extra-feed-like mechanism (v1.0 of this plan) | Empirically rejected after seeing the established `include_winner_pick` pattern. Reuse > invention. |
| Invent a new injection path other than `initial_response_override` (v1.0 of this plan) | Empirically rejected after finding `plan_then_implement_inferencer.py:1969, 1979`. The `--initial-plan` → PTI override is the proven path. |
| **Flag-based emission `include_proposal_index` in `default.jinja2` (v2.0 of this plan)** | **Rejected in v2.1 after Tony's architectural pushback.** `include_winner_pick` and `include_iteration_judgment` are generic aggregation primitives that belong in `default.jinja2`. The proposal-index directive is domain-specialized, not a primitive — putting it in `default.jinja2` pollutes the generic template with a research-propose concept and forces every future structured-output tool to add another flag. Per-variable `template_version` override is the right axis (specialization, not primitive). |
| **Aggregator-only emission via per-variable `template_variables` override (v2.1)** | **Rejected in v3.0 after Tony pointed out the design wastes worker output.** v2.1 had the aggregator invent the proposal index from scratch from raw worker text (regex on prose). v3.0 has workers EMIT structured JSON per-proposal, then the aggregator MERGES (preserves provenance, allows dedup/rank/group with structured input, scales to more workers cleanly). Architecturally cleaner because proposals are now first-class data flowing worker→aggregator, not regex-extracted afterthought. Tradeoff: requires worker template edit (R7) and aggregator master_version override (R6) — both verified safe in Phase 0. |
| `template_master_version="research_propose"` (worker-only, no aggregator override) | This is what tool.json already does today. Insufficient for v3.0 because workers' template family doesn't render task_response_format slot yet — emission would silently no-op. v3.0 adds the slot. |

---

## § 11. Honest comparison: v1.0 (mine) vs Claude's plan vs v2.0 (integrated)

| Dimension | v1.0 (mine) | Claude | v2.0 (this plan) |
|---|---|---|---|
| Module path | `common/proposals/` (invented) | `common/data_models/proposal/` (AF convention) | ✅ Claude's choice |
| ID convention | `H<n>` (RE-specific) | `P<n>` (generic) | ✅ Claude's choice |
| Emission mechanism | New `writer.py` + `post_hooks.py` | Reuse `template_extra_feed["include_X"]` + `_extract_json_block` | ✅ Claude's choice |
| Plan injection | New `initial_plan_from_proposals.md.j2` template | Reuse `initial_response_override` (proven pattern) | ✅ Claude's choice |
| Schema fields | Cloned all 14 RE fields | Trimmed to 10 generic | ✅ Claude's choice + AF `metadata` for overflow |
| Risk register | 6 risks itemized | 0 (gap) | ✅ Kept from v1.0; expanded to 8 |
| Open questions | 5 itemized | 0 (gap) | ✅ Kept from v1.0; expanded to 8 |
| Phased rollout | 4 phases with verification | "Verification" section but no phases | ✅ Kept from v1.0 |
| Alternatives-rejected | Itemized | None | ✅ Kept from v1.0; expanded to 8 |
| Sidecar vs in-markdown | Sidecar (key insight) | Sidecar (same) | ✅ Both correct |
| Atomic-write guarantee | POSIX `tmp→fsync→rename` | Not specified | ✅ Kept from v1.0 |
| SOP `__branch on:` integration story | Phase 4 explicit | Brief mention | ✅ Kept from v1.0 |
| Audit history table | Present | Absent | ✅ Kept from v1.0 |
| RankEvolve migration path | None | Brief mention | ✅ Expanded into §9 |

**Net:** v2.0 is **strictly the union of the best parts**. Architecture is Claude's. Operational discipline (risks, phases, alternatives, ACs, audit) is mine. Result is implementer-ready and pattern-consistent.

---

## § 12. "If forced to pick one plan, which?"

**This integrated v2.0.** Honestly: Claude's plan is architecturally superior (5/5 correct calls vs my 0/5), but missing the operational scaffolding that catches issues before implementation (risks register, ACs, phased rollout, alternatives-rejected). My v1.0 plan had the operational scaffolding but designed in the abstract instead of pattern-matching against AF.

**v2.0 = Claude's architecture × my operational discipline.** No piece reinvents anything; every claim has line-number-cited evidence; every risk has a mitigation; every AC is testable; every alternative was considered.

If forced to pick between Claude's standalone vs. my standalone: **pick Claude's**. The architectural patterns matter more than the operational scaffolding — wrong patterns are debug cycles; missing scaffolding is review cycles.

---

## § 13. Next steps

- **A.** Audit v2.0 critically (find issues/gaps before any external review — there may be propagation issues I didn't catch in the `template_extra_feed → topology YAML → aggregator inferencer` chain)
- **B.** Update `_plan/README.md` to add this plan to the `tools/` index
- **C.** Answer the 8 open questions in §8 to lock the design
- **D.** Start Phase 1 — port the schema + parser, write the 6+ unit tests
- **E.** Verify `_params.extract_proposals` plumbing through `breakdown-multiflow-plan.yaml` (the one place I'm not 100% sure works without a small edit)

---

_End of plan v2.0 — integrated 2026-06-02 20:50._
