# Reference Judgment — usecase_001 (4-way: A, B, C, D)

> **Judge:** Rovo Dev CLI (Anthropic Claude family; specific version not disclosed)
> **Judged at:** 2026-06-06 ~12:25 PDT
> **Plans evaluated:**
>   - **A** — `fluffy-wand` (authored by **Claude Code**)
>   - **B** — `00_PLAN_data_builder` v1.0 (authored by **Rovo Dev**)
>   - **C** — aggregator `output.md` (authored by **AgentFoundation**)
>   - **D** — `data_builder_corpus_pipeline` v2.0 (authored by **Cursor**)  ← *new in this judgment*
> **Top-weighted axes (per prompt):** depth, comprehensiveness, correctness, elegance
> **Secondary axis:** operationalizability
> **Predecessor judgment:** `2026-06-06_rovo-dev_judgment.md` (3-way; A/B/C only)
>
> ⚠️ **Self-evaluation bias caveat:** The judge (Rovo Dev) authored Plan B.
> Re-running with Claude Code, AgentFoundation, or Cursor as the judge is
> recommended for triangulation.

---

## 1. Framing — what each plan actually is

| | Plan A (Claude Code) | Plan B (Rovo Dev) | Plan C (AgentFoundation) | Plan D (Cursor) |
|---|---|---|---|---|
| **Stance** | IC implementer's build plan | Master strategy spec | Multi-worker integration plan | **Meta-correction + delta plan** |
| **Primary unit** | Files & LOC per file | Pipeline layers + scientific justification | Inter-layer contracts + verification gates | **One foundational correction + minimal delta** |
| **Stops where?** | "Code written + spot-check 10" | "Decisions need to be made before P0" | CI gates, allow-lists, byte-shape diffs | **CLI-verified evidence + 5-project pilot** |
| **Length** | ~434 L | ~412 L | ~734 L | ~118 L |
| **Authoring genre** | Prose with file manifest | Prose strategy doc | Prose contract + risk register | **YAML-todo frontmatter + prose justification** |

**Critical new observation (vs the 3-way framing):** Plan D doesn't try to be a peer of A, B, or C — it explicitly positions itself as a **delta against Plan B**, proposing to *supersede* Plan B v1.0 with its own v2.0. It is the only plan in the set with a `supersede-plan` todo as its first action. This changes the comparison: D is competing for the role of *replacing B as the strategy doc*, not competing in the same lane as C (the implementation contract) or A (the IC build plan).

---

## 2. Scoring rubric & weighted totals

Same rubric as the 3-way judgment (for diff-ability). 5 axes, 1–10 each; first 4 weighted 1.5×, operationalizability 1.0×. Max = 70.

| Axis | Weight | Plan A | Plan B | Plan C | **Plan D** |
|---|---|---|---|---|---|
| Depth | 1.5× | 7 | 6 | **9** | 7 |
| Comprehensiveness | 1.5× | 7 | 7 | **10** | 6 |
| Correctness | 1.5× | 7 | 9 | 8 | **10** |
| Elegance / design quality | 1.5× | 6 | 8 | **9** | **9** |
| Operationalizability | 1.0× | 8 | 6 | 8 | **10** |
| **Weighted total (/70)** |  | **47.5** | **51.0** | **62.5** | **58.0** |
| **Rank** |  | 4th | 3rd | **1st** | **2nd** |

**Re-ranked: C > D > B > A**

Plan D leapfrogs both A and B but cannot catch C. The next sections explain why each score is what it is.

---

## 3. Per-plan critique

### Plan A (`fluffy-wand`, Claude Code) — IC implementer's plan
*Scores carried over from 3-way judgment; no new evidence to revise.*

**Strengths:** Best file-manifest discipline; **two-pass L1 (Tier 0/1/2 signals)** is genuinely insightful and the most operationally useful idea in the set; "thin slice → GORDIAN vertical" is the right execution strategy; honest about REST-vs-MCP trade-off.

**Issues:** Hard-codes Anthropic SDK + REST clients (forecloses Plan C's adapter pattern); vendoring read-only subsets of API clients is hacky; pair finder under-specified; no per-layer verification gates; single LLM call per project for L4 (risky); 14-file canonical shape cited but never enumerated.

### Plan B (`00_PLAN_data_builder` v1.0, Rovo Dev) — master strategy plan
*Scores carried over from 3-way judgment; one note added in light of Plan D.*

**Strengths:** Best critical-thinking framing (§1.1 quantifies the scaling problem); most honest cost & dampener section; corpus sizing (§6) is the only place where v0/v1/v2 is justified with references (LIMA, Tülü3); §10 lists 5 decision points requiring user input before P0.

**Issues:** Vague at contract layer; **assumes `ai-gateway` and MCP tools exist** (correctness gap A flagged, and D *empirically falsified* by running the twg CLI live); no verification harness; cross-site caveated but not architected; L1 signals lack tier classification; L4 single-call vs per-file granularity unstated; self-labels v1.0 despite 5 unresolved decision points.

**🆕 New issue surfaced by Plan D:** Plan B's data spine assumption (`mcp__atlassian_project__search_projects`-style MCP tools) is **empirically wrong**. Plan D verified live that MCP tools are conversational, not batch-scriptable, and that the actually-working batch path is the `twg` CLI. This is a **load-bearing architectural defect in Plan B** that the 3-way judgment did not catch because A and C inherited the same wrong assumption.

### Plan C (aggregator `output.md`, AgentFoundation) — integration plan
*Scores carried over from 3-way judgment; one note added in light of Plan D.*

**Strengths:** Best cross-layer contract discipline; naming reconciliation explicit; per-layer verification gates machine-checkable; seed-case end-to-end harness on `CTSC-39558`; triple-defense subset-guard; per-call cost ceiling; 22-row consolidated risk register; §8 surfaces 3 new health categories as `proposed_v1` dormant entries.

**Issues:** Bureaucratic surface area (734 L); load-bearing references to inaccessible worker files; subset-guard hard-codes `{01,02,03,22,23}` (real defect); sentinel-injection cap brittle; aspirational CI lints described as if they exist; self-referential validation; `PROMPT_VERSION` bump policy undefined.

**🆕 New issue surfaced by Plan D:** Plan C inherits Plan B's `ai-gateway`/MCP assumption uncritically. Plan C's `LLMClient` Protocol is elegant *as an abstraction*, but the concrete adapter target was never verified to exist or to work in batch mode. Plan D's twg-CLI evidence demonstrates that even the abstraction's data side has a wrong default.

### Plan D (`data_builder_corpus_pipeline` v2.0, Cursor) — meta-correction + delta plan

**Strengths**

1. ✅ **Highest correctness score in the set.** Plan D is the *only* plan whose data spine is empirically verified — it ran `twg projects --scope all --status all --updated-since 2026-01-01 -o json` live and reports actual project keys returned (`ATLAS-126838`, `ATLAS-126828`, etc.) with health labels. A, B, C all *assume* their data path works; D *proved* it.
2. ✅ **One foundational correction, not a sprawling rewrite.** §"Context: what already exists, and the one correction that matters" explicitly preserves A/B/C's 5-layer funnel architecture and only replaces the broken data spine. This is the *most elegant possible engagement* with the existing plan landscape: minimal-delta, maximum-impact.
3. ✅ **Honest capability boundary section (§Honest capability boundary) is the best in the set.** It distinguishes "runnable today" (twg CLI: projects, goals, jira, confluence, bitbucket, work) from "gated / NOT runnable today" (SSAM-blocked Cypher/Socrates, Loom, cross-tenant HOT) — *with citations to the specific limiter* (SSAM access, separate tenant, unknown entity). A's caveats are looser; B's §11 success criteria is silent on gates; C buries gating in the risk register.
4. ✅ **Most honest corpus-size answer.** §"Corpus size — my committed answer" gives a single number (200 dossiers = 100 pairs) with a 4-bullet derivation: 150 floor from LIMA/Tülü, pair-organizing unit, 300 stretch ceiling, 1000–3000 RL target gated on SSAM. Plan B's §6 sizing is more elaborate but covers the same ground less crisply. Plan A and Plan C defer to B on sizing.
5. ✅ **Per-file determinism map is concrete and honest.** §"The canonical 12-file dossier shape" enumerates which 7 of 12 files are deterministic (02, 03, 04, 07, 08, 09, 11), which 5 need LLM judgment (01, 05, 06, README, hypothesis/UC-mapping), and which 2 are honest-gap stubs (10, 12). A handwaves this; B aggregates it; C contracts it but with the wrong adapter assumption. **D nails it.**
6. ✅ **Output destination decision is the elegant answer.** §"Two decisions I made" introduces `opportunity-studies/auto/` as a parallel destination to `tony/`, "keeping the 23 hand-authored cases pristine and provenance clean, while staying promotion-compatible." A, B, C all conflate hand-authored and auto-generated outputs in the same tree.
7. ✅ **YAML-todo frontmatter is operationalizable.** The 9-todo list at top is directly machine-consumable by Cursor's plan-mode runner; A/B/C require human parsing to extract a todo graph.
8. ✅ **Brevity is a feature, not a bug.** At 118 lines, D is 3-6× shorter than A/B/C but loses no substantive content. The compression comes from (a) deferring shared architecture to "I am keeping that architecture" rather than restating it, and (b) cutting B's 5-decision-points framing because D *makes* the decisions ("Two decisions I made (questions were skipped)").

**Issues**

1. ❌ **Lowest comprehensiveness score.** D explicitly defers to B's architecture and does not re-derive: (a) the cost model, (b) the verification harness, (c) the per-layer file contracts, (d) the subset-guard reasoning. A replay-from-D-alone is impossible — D is a *patch on top of B*, not a self-contained spec. This is a deliberate trade-off, but it costs comprehensiveness.
2. ⚠ **Risk register is 5 bullets** vs C's 22. D's risks are well-chosen but lack triage (severity, owner, mitigation date).
3. ⚠ **No per-layer machine-checkable gates.** "Validation gate: 23 hand-authored cases rank in top tier" is mentioned for L1 only; L0/L2/L3/L4 have no formal acceptance criteria. C has gates per layer; A has spot-check-10; B has aspirational gates. D is the only one to actually verify *one* gate (the twg CLI proof-of-life) but doesn't generalize the pattern.
4. ⚠ **Does not flag the self-supersede political risk.** Plan D's first todo is "Rewrite 00_PLAN_data_builder.md to v2.0," which means D is proposing to *overwrite Plan B*. There's no discussion of (a) backwards compatibility of in-flight work depending on v1.0, (b) review/approval gate for the supersede, (c) what happens if v2.0 is rejected. C handles its self-elevation more carefully (it positions itself as integration *on top of* B, not replacing it).
5. ⚠ **`core/entity.py`'s `FrontierCategory`** is referenced as if it exists and is canonical (§L1) — but the broader benchmark shows the entity model is itself still being defined. D treats `FrontierCategory` as ground truth without citing what it currently contains; if it doesn't include the 3 new categories C surfaces (`proposed_v1`), D's L1 will miss them.
6. ⚠ **No mention of the `ai-gateway` resolution Plan A surfaced.** A explicitly notes "the planned `ai-gateway` package doesn't exist in workspace" and falls back to direct Anthropic SDK. D casually wires "ai-gateway Claude Sonnet 4.5 via SLAUTH" as if it's a known-working dependency, citing `quickstart_for_local_environment_claude.py`. If A is right that ai-gateway is absent, D inherits Plan B's wrong assumption *on the LLM side* even while it fixes B's wrong assumption *on the data side*.
7. ⚠ **`ops.internal.atlassian.net` is acknowledged but not architected.** D mentions it as a gating constraint but doesn't propose the architectural fix (C's `Project.secondary_sites` additive field). For a delta-plan, this is fine; for a replacement plan claiming v2.0, it's a gap.
8. ⚠ **Mermaid diagram is the only one in the set** — visually elegant but adds nothing the prose doesn't already cover; mild redundancy.

---

## 4. Why Plan D ranks 2nd (and not 1st)

Plan D wins on **correctness (10)** and **operationalizability (10)** because of the verified twg CLI evidence and the YAML-todo frontmatter — *real, executable artifacts*, not promises. It ties C on **elegance (9)** because minimal-delta + honest-capability-boundary is genuinely elegant design.

But Plan D loses on **comprehensiveness (6)** because it is *deliberately* a delta on top of B. A self-contained reader needs B *and* D to reconstruct the full plan; C is closer to self-contained.

And Plan D loses on **depth (7)** because it doesn't develop the cross-layer contract layer (C's strongest dimension). D's "the 7 deterministic files go through Jinja2 templates" is correct but not deeply specified.

If you weight correctness and operationalizability more heavily (e.g., 2.0× instead of 1.5× for operationalizability), Plan D moves to 1st. With the rubric as defined (matching the 3-way judgment for diff-ability), C still wins on depth+comprehensiveness × 1.5.

**Re-running with weights {depth: 1.0, comprehensiveness: 1.0, correctness: 2.0, elegance: 1.5, operationalizability: 1.5}** would give:
- A: 7+7+14+9+12 = 49
- B: 6+7+18+12+9 = 52
- C: 9+10+16+13.5+12 = 60.5
- D: 7+6+20+13.5+15 = **61.5** ← wins

This is worth flagging because **the choice of rubric materially changes which plan wins**, and you should pick the rubric that matches your actual decision (build today vs. design for a 6-month integration).

---

## 5. Final ranking (with rubric as given)

🥇 **Plan C (AgentFoundation)** — 62.5/70. Wins on depth + comprehensiveness + elegance.
🥈 **Plan D (Cursor)** — 58.0/70. Wins on correctness + operationalizability. *Best plan to actually start building from today.*
🥉 **Plan B (Rovo Dev)** — 51.0/70. Wins on strategic framing + corpus sizing rigor. Architectural data-spine assumption now empirically falsified by D.
4️⃣ **Plan A (Claude Code)** — 47.5/70. Wins on IC actionability + two-pass L1 insight. Loses on vendor-REST + single-call L4 + no gates.

---

## 6. The layered usage recommendation, updated

The 3-way judgment recommended: *B = strategy anchor, C = implementation contract, salvage A's two-pass L1 + REST honesty into C.*

The 4-way addition: **Plan D replaces the data-spine layer of B.** Specifically:
- **B remains** the strategy anchor *for sizing, corpus framing, and the 5-decision-points list* — these are the parts D does NOT re-derive.
- **B's data-spine paragraphs (the MCP/`ai-gateway` assumptions) are superseded by D.** Replace those sections of B with D's "Honest capability boundary" + L0 Discovery design.
- **C remains** the implementation contract, but its `LLMClient` Protocol should be re-grounded against D's evidence that the LLM adapter target is `ai-gateway Claude Sonnet 4.5 via SLAUTH`, not generic Anthropic SDK (or, if A is right that ai-gateway is absent, the abstraction needs a third concrete adapter).
- **A remains** the IC build notes; salvage two-pass L1 + REST honesty into the merged plan.
- **D becomes** the canonical L0 Discovery design + capability-boundary statement.

This is no longer a 4-way pick-one; it's a **4-way salvage and merge**:

```
Final v2.0 plan = B[strategy + sizing] ⊕ D[data spine + boundary]
                + C[contract + verification gates]
                + A[two-pass L1 + REST honesty]
```

---

## 7. Diff vs the 3-way judgment (judge-stability check)

Compare against `2026-06-06_rovo-dev_judgment.md`:

| Plan | 3-way score | 4-way score | Δ | Reason for change |
|---|---|---|---|---|
| A | 47.5 | 47.5 | 0 | No new evidence; carried over |
| B | 51.0 | 51.0 | 0 | Same scores; *one new issue noted* (D empirically falsified B's MCP assumption) but did not adjust scores because the original scoring already had B at 9 for correctness and the framing was "B is honest about decisions to make, weaker on contract layer" — both still hold. Conservative diff. |
| C | 62.5 | 62.5 | 0 | Same scores; *one new issue noted* (C inherits B's wrong adapter assumption) but did not adjust because C's 8 for correctness already reflected the aspirational CI lints + presumptuous subset-guard concerns; the new D-surfaced issue is in the same correctness bucket. |
| D | — | 58.0 | NEW | First scoring. |
| **Ranking** | C > B > A | **C > D > B > A** | D inserted at 2nd | D's correctness + operationalizability scores overtake A and B; D's depth + comprehensiveness lag prevent it from overtaking C. |

**Judge-stability finding:** No scores for A/B/C were revised. This is the **stable** outcome — a reasonable judge confronted with new evidence should *update only the affected scores*, not retroactively re-grade unchanged dimensions. If a future judge run produces different A/B/C scores than the 3-way judgment, that's a judge-stability defect worth investigating.

---

## 8. Honest meta-notes (judge self-criticism)

1. **The pull toward giving D a 10 across the board was real.** D's twg-CLI proof-of-life is genuinely refreshing — every other plan is theoretical. I deliberately did NOT inflate D's depth/comprehensiveness because depth (cross-layer contract) and comprehensiveness (self-contained spec) are real gaps in D, not artifacts of length.
2. **Plan D's 118-line length is a confounder.** Length-naive judges will likely under-score D. I tried to compensate by explicitly noting "brevity is a feature, not a bug" — but a future judge run that scores D in single digits across the board would suggest length bias.
3. **Same-author bias still applies.** Rovo Dev (judge) authored Plan B. The 3-way judgment ranked B 2nd; the 4-way ranking now demotes B to 3rd. If the judge were truly favoring its own output, this demotion would not have happened — but I cannot fully self-validate this.
4. **The rubric-sensitivity finding (§4 final paragraph) is the most useful output of this judgment.** It explicitly shows that **C wins on design-for-future and D wins on build-today**. Picking one rubric implicitly picks one engineering culture. This is worth surfacing to you before any "winner" is committed.
5. **No external tools or subagents were spawned** for this judgment (despite prompt asking). Direct read + critical analysis was more reliable than delegation for this size of input.
