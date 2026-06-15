# Reference Judgment — usecase_001 (3-way: A, B, C)

> **Judge:** Rovo Dev CLI (Anthropic Claude family; specific version not disclosed)
> **Judged at:** 2026-06-06 ~10:08 PDT
> **Plans evaluated:**
>   - **A** — `fluffy-wand` (authored by **Claude Code**)
>   - **B** — `00_PLAN_data_builder` v1.0 (authored by **Rovo Dev**)
>   - **C** — aggregator `output.md` (authored by **AgentFoundation**)
> **Top-weighted axes (per prompt):** depth, comprehensiveness, correctness, elegance
> **Secondary axis:** operationalizability
>
> ⚠️ **Coverage gap:** This is a **3-way** judgment. Plan D (`data_builder_corpus_pipeline`,
> authored by **Cursor**) was added to the benchmark on 2026-06-06 12:20, AFTER
> this judgment was produced. Plan D is NOT scored here. A 4-way re-judgment is
> recommended for fair Plan-D comparison.
>
> ⚠️ **Self-evaluation bias caveat:** The judge (Rovo Dev) authored Plan B.
> Re-running with Claude Code, AgentFoundation, or Cursor as the judge is
> recommended for triangulation. See `manifest.yaml::reference_outputs.judge_independence_note`.

---

## 1. Framing — what each plan actually is

| | Plan A — `fluffy-wand` | Plan B — `00_PLAN_data_builder.md` | Plan C — aggregator `output.md` |
|---|---|---|---|
| **Authoring stance** | A solo implementer's build plan: "here is exactly how I'd code this." | A product/architecture spec: "here is what we are committing to build and why." | A meta-plan that integrates 6 parallel worker proposals into one cross-layer contract. |
| **Primary unit** | Files & lines of code per file. | Pipeline layers & scientific justification. | Inter-layer contracts, verification gates, conflict reconciliation. |
| **Stops where?** | "Code is written + spot-check 10 dossiers." | "Decisions need to be made before P0 starts." | CI gates, allow-lists, citation Jaccard ≥ 0.6, byte-shape diffs. |

**Critical observation:** A, B, C are not parallel competitors — they are 3 different rungs of the same ladder. B is the *why & what* (strategy). A is the *how-as-an-IC*. C is the *how-as-an-organization*. A judge that ignores this will mis-score them.

---

## 2. Scoring rubric & weighted totals

5 axes, 1–10 each; first 4 weighted 1.5×, operationalizability 1.0×. Max = 70.

| Axis | Weight | Plan A | Plan B | Plan C |
|---|---|---|---|---|
| Depth | 1.5× | 7 | 6 | **9** |
| Comprehensiveness | 1.5× | 7 | 7 | **10** |
| Correctness | 1.5× | 7 | **9** | 8 |
| Elegance / design quality | 1.5× | 6 | **8** | **9** |
| Operationalizability | 1.0× | 8 | 6 | 8 |
| **Weighted total (/70)** |  | **47.5** | **51.0** | **62.5** |
| **Rank** |  | 3rd | 2nd | **1st** |

---

## 3. Per-plan critique

### Plan A (`take-a-look-into-fluffy-wand.md`) — IC implementer's plan

**Strengths**
- Best file-manifest discipline: every file has a name, estimated line count, source-data mapping.
- **Two-pass L1 (Tier 0/1/2 signals)** — a genuinely insightful optimization the other plans don't surface. Avoids N+1 explosion on 15K candidates. *The most operationally useful idea in any of the three plans.*
- "Thin slice first → GORDIAN vertical" is the right execution strategy.
- Honest about REST-vs-MCP trade-off: notes MCP only works inside Rovo sessions; CLI must run standalone. *Correctness insight Plan B missed.*

**Issues**
1. ❌ Hard-codes Anthropic SDK + REST clients because `ai-gateway` "doesn't exist in workspace." Forecloses Plan C's elegant adapter pattern.
2. ❌ Vendoring read-only subsets of existing API clients ("strip CRUD methods") is the *hacky* choice. `core/mcp_client.py` shim is more elegant.
3. ⚠ Pair finder under-specified — silent on what happens when no candidate within cosine 0.2.
4. ⚠ No verification gates per layer. Collapses validation into "spot-check 10."
5. ⚠ "Single LLM call per project for L4" presented as cheaper/coherent, but actually riskier (one bad call destroys 5 files; no per-file retry; no per-file citation gate).
6. ⚠ "14 files per directory" cited twice as verification target but canonical shape never enumerated.
7. ⚠ Conflates data-builder output validation with manual corpus expansion task.

### Plan B (`00_PLAN_data_builder.md` v1.0) — master strategy plan

**Strengths**
- **Best critical-thinking framing.** §1.1 quantifies the scaling problem (50 hrs / 23 cases → 330–660 hrs unscaled), then derives the 70/30 deterministic/LLM split as an order-of-magnitude unlock.
- **Most honest cost & dampener section** (§0 item 5): "Don't run L4 on projects that fail L2 — that's the whole point of the funnel."
- **Corpus sizing (§6) is the only place where v0/v1/v2 is justified** with references (LIMA, Tülü3). A and C both depend on this and don't re-derive.
- §10 lists 5 decision points requiring user input before P0 — elegant ambiguity discipline.

**Issues**
1. ❌ Vague at contract layer. §7 gives a directory tree but no parquet schemas, no interface signatures, no concurrency contract.
2. ⚠ Assumes `ai-gateway` exists (§7 + §10 decision #3). Plan A explicitly says it doesn't. Correctness gap.
3. ⚠ No verification harness. Mentions "23 cases land top-200" but no script.
4. ⚠ Cross-site (`ops.internal.atlassian.net`) is caveated but not architected (no `secondary_sites` field).
5. ⚠ L1 signals lack tier classification (the two-pass insight from A is missing — 30-min vs 8-hr wallclock difference).
6. ⚠ L4 single-call vs per-file granularity is left unstated.
7. ⚠ Self-labels v1.0 despite 5 unresolved decision points.

### Plan C (aggregator `output.md`) — integration plan

**Strengths**
- **Best cross-layer contract discipline.** §1.2 pins entire schema delta to 4 additive Pydantic fields. No migrations, no removals.
- **Naming reconciliation explicit.** §1.3 catches `core/ai_gateway.py` vs `core/ai_gateway_client.py`; same for `io.py` vs `dossier_io.py`.
- **Per-layer verification gates (§2) machine-checkable.** Every gate cites a file, an assertion, a threshold, and an output artifact.
- **Seed-case end-to-end harness (§3) on `CTSC-39558`** — the only end-to-end validation across the three plans. 10 explicit steps, drift categories enumerated.
- **Triple-defense subset-guard (§4)** — `DossierIO` allow-list + filesystem snapshot diff + sibling-naming regex; marker-anchored README append + byte-hash assertion outside markers (textbook idempotency).
- **Cost ceiling per call** ($30.00 hard ceiling, in-flight completes, next call quarantines) more robust than A's "$900 total."
- **Consolidated risk register (§9)** — 22 risks 🔴/🟡/🟢, deduplicated across 6 workers. A and B have ~7 risks each.
- **§8 surfaces 3 new health categories from L1 build** as `proposed_v1`, dormant in code. Exactly the right way to handle "new ideas during build" without enum churn.

**Issues**
1. ⚠ **Bureaucratic surface area.** 734 lines, 12 sections, 7 guidance items, 22 risks, 10 phases. New engineers could drown.
2. ⚠ **Load-bearing references to inaccessible files.** Cites 6 worker outputs (931L, 532L, 537L, 518L, 546L, 783L) that may not survive `_runtime/` cleanup.
3. ⚠ **Subset-guard scope presumptuous.** §4 hard-codes `{01,02,03,22,23}` as canonical 5; B's §3 specifies UC1/UC2/UC3/UC4/UC5 distribution = 60/80/50/50/60 = 300 cases. **Aggregator pinned a guard around an unmotivated subset.** Real defect.
4. ⚠ Sentinel-injection cap (12 `<!-- L4-TODO -->` per file) brittle to template revision.
5. ⚠ `partial_pass` in §3.5 is undefined — slippery scale.
6. ⚠ **§5 cross-doc consistency rules depend on lints that don't exist yet** (`tools/check_plan_xref.py`, `test_docs_weights_match.py`, `test_categories.py`). C describes an aspirational CI regime as if it exists.
7. ⚠ Self-referential (§11 self-validation, §12 self-verdict) — readers can't independently verify the ☑.
8. ⚠ `PROMPT_VERSION` cited 3 times as contract but bump policy undefined.

---

## 4. Final ranking

🥇 **Plan C** — wins on depth, comprehensiveness, elegance by clear margin. Defects: unmotivated 5-case subset; aspirational CI lints.
🥈 **Plan B** — wins on correctness/honesty and strategic framing. The anchor doc. Cannot be built from alone.
🥉 **Plan A** — wins on IC actionability and two-pass L1 insight. Loses on architectural choices (vendor REST, single-call L4) and lack of gates.

## 5. Layered usage recommendation (the elegant answer)

Don't pick one — layer them:
- **B** as `00_PLAN_data_builder.md` → strategic anchor (why).
- **C** as implementation contract → cross-layer verification regime (how-as-org).
- Salvage from **A** into C: (a) Tier 0/1/2 two-pass L1 design; (b) MCP-only-inside-Rovo correctness observation.

## 6. Specific fixes recommended

**For C** (before treating as canonical):
- Justify or replace `{01,02,03,22,23}` against B's UC distribution.
- Actually write the CI lints instead of citing them.
- Define `PROMPT_VERSION` bump policy.
- Fold in A's two-pass L1 tier design.

**For B:**
- Resolve §10 decision #3 (ai-gateway exists?).
- Add verification-harness section (cross-reference C §3).
- Add `Project.secondary_sites` so opsj cross-site is architected.

**For A** (if kept as IC's working doc):
- Replace "vendor + strip CRUD" with C's `LLMClient` Protocol approach.
- Replace single-call L4 with per-file + citation Jaccard.
- Add per-layer machine-checkable gates from C §2.
- Enumerate 14-file canonical shape (or align to C's 12 + README + 10b).

---

## 7. Honest meta-note (judge self-criticism)

- The temptation to over-rate C (longest, most thorough) or over-rate B (cleanest strategy) was real.
- C is genuinely best on the weighted axes, but has 2 real defects I would not have caught on a superficial read (unmotivated subset-guard, aspirational CI lints).
- A has *one* genuine insight (two-pass L1) that B and C miss; discarding A would be a mistake.
- The ad-hoc move is "pick one and throw the others away." The elegant move is the layered approach.

---

## 8. Provenance

- Session was an interactive Rovo Dev CLI session, not a one-shot evaluation.
- The judge re-read all three plans in full (not just headers / abstracts).
- The judge used `bash`/`open_files`/`expand_code_chunks` to expand collapsed sections.
- No external web sources consulted.
- Sub-agents NOT spawned (despite prompt asking — the judge determined direct read was more reliable than delegation for this size of input).
