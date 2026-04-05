# RankEvolve Public Demo Video Recording Guidelines

**Target Duration:** ~7-8 minutes total (full "Evolve" loop with 2 complete iterations)
**Demo Name:** RankEvolve Public Demo
**Focus:** Deep research workflow demonstrating iterative evolution from SYNAPSE v1 → v2

> **Note:** Wukong has been removed from this demo as it is not publicly available. The demo focuses on generative-recommenders (HSTU architecture).

---

## Demo Flow Overview (18 Steps)

### Iteration 1 (Steps 0-8)

| Step | Section | Duration | Cumulative |
|------|---------|----------|------------|
| 0 | Welcome & Plan Selection | 0:25 | 0:25 |
| 1 | Codebase Investigation | 0:35 | 1:00 |
| 2 | Research Planning (Iteration 1) | 0:40 | 1:40 |
| 3 | Deep Research Execution (Iteration 1) | 0:45 | 2:25 |
| 4 | Proposal Generation (Iteration 1) | 0:35 | 3:00 |
| 5 | Unified Architecture Synthesis (Iteration 1) | 0:30 | 3:30 |
| 6 | Implementation (Iteration 1) | 0:30 | 4:00 |
| 7 | Experiments (Iteration 1) | 0:30 | 4:30 |
| 8 | Analysis (Iteration 1) | 0:25 | 4:55 |

### Evolve Transition & Iteration 2 (Steps 9-16)

| Step | Section | Duration | Cumulative |
|------|---------|----------|------------|
| 9 | Evolve: Start Iteration 2 | 0:20 | 5:15 |
| 10 | Research Planning (Iteration 2) | 0:20 | 5:35 |
| 11 | Deep Research (Iteration 2) | 0:25 | 6:00 |
| 12 | Proposal Generation (Iteration 2) | 0:20 | 6:20 |
| 13 | Unified Architecture Synthesis (Iteration 2) | 0:20 | 6:40 |
| 14 | Implementation (Iteration 2) | 0:20 | 7:00 |
| 15 | Experiments (Iteration 2) | 0:25 | 7:25 |
| 16 | Analysis (Iteration 2) | 0:20 | 7:45 |

### Completion (Step 17)

| Step | Section | Duration | Cumulative |
|------|---------|----------|------------|
| 17 | Final Summary & Completion | 0:25 | 8:10 |

> **Note:** Total ~8:10. Iteration 2 (Steps 9-16) can be sped up 1.5× in post-production. The key narrative: Iteration 1 identifies fixed τ=24h as hurting temporal items (-2.5%) → Iteration 2 fixes it with multi-timescale FLUID → +3.0% swing on temporal items.

---

## Step-by-Step Script

### Step 0: Welcome & Plan Selection (0:25)

**What happens on screen:**
- RankEvolve welcome message appears
- 8-step research cycle displayed (Codebase Investigation → Research Planning → Deep Research → Proposal Generation → Merge & Synthesize → Implementation → Experiments → Analysis)
- "Execute the Plan" and "Adjust Plan" buttons shown
- Codebase path input field with default "generative-recommenders"
- Research mode selector (Paradigm-Shifting, Systematic, Targeted, Exploratory)

**Speaker talking points:**
- "Welcome to RankEvolve - an AI research assistant for discovering paradigm-shifting innovations"
- "Notice the research cycle: we start with codebase investigation, plan research queries, execute deep research, generate proposals, and iterate"
- "The key innovation is the Evolve methodology - we iterate based on experimental findings"

**Action:** Click "✅ Execute the Plan"

---

### Step 1: Codebase Investigation (0:35)

**What happens on screen:**
- Two parallel progress sections appear:
  1. 🔍 Generative-Recommenders Investigation
  2. 📝 Codebase Documentation (Sphinx)
- Messages stream showing file analysis activities
- Summary of findings appears with key limitations

**Speaker talking points:**
- "RankEvolve first analyzes the generative-recommenders codebase"
- "It's investigating HSTU (Hierarchical Sequential Transduction Unit) with O(N²) self-attention"
- "Notice the Sphinx documentation being generated automatically"
- "Key limitations identified: computational O(N²), representational (static embeddings), temporal (discrete positions)"

**Action:** Click "✅ Continue to Research Planning"

---

### Step 2: Research Planning (Iteration 1) (0:40)

**What happens on screen:**
- Progress header explains 5 reasoning patterns
- Research planning progress section streams messages
- 4 ranked queries appear with editable checkboxes
- Additional queries available to add

**Speaker talking points:**
- "ITERATION 1 - Step 2/8: Research planning using 5 reasoning patterns"
- "Barrier-First Thinking, Cross-Domain Transfer, Mathematical Equivalence, Assumption Challenging, Scalability Inversion"
- "Q1: Linear-time sequence modeling via Mamba/SSM"
- "Q2: Fundamental barriers analysis"
- "Q3: Contextual and polysemous representations"
- "Q4: Temporal and interaction design assumptions - note this becomes critical in Iteration 2"

**Action:** Click "✅ Execute Deep Research"

---

### Step 3: Deep Research Execution (Iteration 1) (0:45)

**What happens on screen:**
- 5 parallel research streams execute:
  - 🔍 General Deep Research
  - 🔍 Q1: Linear-Time Sequence Modeling
  - 🔍 Q2: Fundamental Barriers Analysis
  - 🔍 Q3: Contextual Representations
  - 🔍 Q4: Temporal & Interaction Design
- Each stream shows detailed progress messages
- Research result files appear when complete

**Speaker talking points:**
- "5 parallel deep research streams scanning RecSys '24, KDD '24, ICML '24, NeurIPS '24"
- "Q1 discovers Mamba-2's State Space Duality proof - Linear Attention equals SSM"
- "Q2 quantifies the industrial optimization ceiling at 0.02-0.15% NE"
- "Q4 designs FLUID with **fixed τ=24h** - remember this, it's our Iteration 2 target"

**Action:** Click one research result to view, then click "✅ Generate Proposals"

---

### Step 3 (Optional Branch): Deep Dive on a Topic

**What happens on screen:**
- If user clicks "🔍 Deep dive on a topic"
- Additional research section appears (e.g., Cross-Sequence Interaction 2024-2025)
- Updated research findings added to Q4

**Speaker talking points:**
- "Users can request deep dives on specific topics"
- "This adds 2024-2025 research findings: Orthogonal Alignment Thesis, SSM Renaissance, Dual Representation Learning"

**Action:** Click "✅ Continue to Generate Proposals"

---

### Step 4: Proposal Generation (Iteration 1) (0:35)

**What happens on screen:**
- 4 themed proposal sections appear:
  - 💡 Q1: SSD-FLUID Backbone Proposal
  - 💡 Q2: Three Validity Walls Framework Proposal
  - 💡 Q3: PRISM Hypernetwork Proposal
  - 💡 Q4: FLUID & Cross-Sequence Interaction Proposal
- Progress shows proposal synthesis

**Speaker talking points:**
- "Each research stream generates a themed proposal"
- "Q1 designs the SSD-FLUID backbone: O(N) training, O(1) streaming inference"
- "Q2 formalizes the Three Validity Walls framework"
- "Q3 proposes PRISM hypernetwork for user-conditioned embeddings"
- "Q4 designs FLUID with fixed τ=24h (baseline) plus Multi-Token cross-sequence interaction"

**Action:** Click "✅ Synthesize Unified Architecture"

---

### Step 5: Unified Architecture Synthesis (Iteration 1) (0:30)

**What happens on screen:**
- Synthesis progress section streams messages
- SYNAPSE v1 architecture diagram/table appears
- Three pillars: SSD-FLUID, PRISM, FLUID

**Speaker talking points:**
- "Proposals synthesized into SYNAPSE v1 architecture"
- "SSD-FLUID: O(N) training, O(1) inference - targeting 5-15× throughput"
- "PRISM: User-conditioned polysemous embeddings - targeting +2-5% cold-start CTR"
- "FLUID: Closed-form temporal decay with **fixed τ=24h** - targeting +2-4% re-engagement"
- "Multi-Token: Cross-sequence interaction via Orthogonal Alignment"

**Action:** Click "✅ Implement SYNAPSE v1"

---

### Step 6: Implementation (Iteration 1) (0:30)

**What happens on screen:**
- Code generation progress for modules:
  - 🔧 Code Generation
  - 📋 Module Integration
- Implementation files appear (ssd_fluid_backbone.py, prism_hypernetwork.py, fluid_temporal_layer.py, multi_token_interaction.py)

**Speaker talking points:**
- "Implementation generates four core modules"
- "SSD-FLUID backbone with dual-mode processing (parallel training, recurrent inference)"
- "PRISM hypernetwork with shared generator (300× memory reduction)"
- "Note the FLUID layer uses **fixed τ=24 hours** for ALL content types"
- "This will be our improvement target in Iteration 2"

**Action:** Click "✅ Run Experiments"

---

### Step 7: Experiments (Iteration 1) (0:30)

**What happens on screen:**
- Experiment sections:
  - ✅ Existing Experiment Results Available
  - 🧪 Experiment Setup (if enabled)
  - 📊 Training Progress (if enabled)
  - 📈 Metrics Collection (if enabled)
- Results table showing SYNAPSE v1 vs HSTU baseline

**Speaker talking points:**
- "Experiments on MovieLens-25M and Amazon-Books"
- "Full system results: NDCG -9.3% (vs HSTU+sampling: -5.2%), Throughput 2×, Cold-start +2.6%"
- "**FLUID re-engagement: -1%** - below baseline!"
- "Critical finding: **Temporal items: -2.5%** - fixed τ=24h HURTS fast-decay content"

**Action:** Click "✅ Analyze Results"

---

### Step 8: Analysis (Iteration 1) (0:25)

**What happens on screen:**
- Analysis sections:
  - 📊 Performance Comparison
  - 🔍 Ablation Analysis
  - 💡 Insight Extraction
- Root cause identification highlighted

**Speaker talking points:**
- "Analysis reveals the root cause!"
- "SSD-FLUID: 4× throughput but -6% NDCG (O(N) approximation is lossy)"
- "PRISM: +2.6% cold-start but -6% warm NDCG, +30% latency"
- "**FLUID: -2.5% on temporal items** - τ=24h is fundamentally wrong for news/trending"
- "Multi-Token: +0.5% NDCG but +25% latency (exceeds budget)"
- "**Recommendation**: Multi-timescale FLUID + Enhanced Multi-Token v2 for Iteration 2"

**Action:** Click "🔄 Start Iteration 2"

---

### Step 9: Evolve: Start Iteration 2 (0:20)

**What happens on screen:**
- Iteration 1 summary displayed
- Iteration 2 focus identified (Dual Focus)
- "🔄 Iteration 2" indicator appears

**Speaker talking points:**
- "This is the Evolve transition - we loop back based on experiment insights"
- "Iteration 1 diagnosed: fixed τ=24h HURTS temporal items (-2.5%)"
- "Iteration 2 has dual focus:"
- "  Focus 1: Multi-Timescale FLUID (fix τ mismatch)"
- "  Focus 2: Enhanced Multi-Token v2 (reduce latency from +25% to +12%)"

**Action:** Click "✅ Continue to Research Planning (Iteration 2)"

---

### Step 10: Research Planning (Iteration 2) (0:20)

**What happens on screen:**
- 4 targeted research queries focused on temporal modeling:
  - Q1: Multi-Timescale Temporal Decay Architectures
  - Q2: Cross-Sequence Temporal Interactions (SSM Renaissance)
  - Q3: Stable Training for Learnable Timescale Parameters
  - Q4: Enhanced Multi-Token v2 Latency Optimization
- "🔄 Iteration 2" indicator visible

**Speaker talking points:**
- "ITERATION 2 - targeted research based on Iteration 1 diagnosis"
- "Q1: Multi-timescale architectures (TiM4Rec, MS-RNN)"
- "Q2: SSM Renaissance + Orthogonal Alignment for cross-sequence"
- "Q3: Stable training for learnable τ parameters"
- "Q4: GQA + sparse attention to reduce Multi-Token latency"

**Action:** Click "✅ Execute Deep Research (Iteration 2)"

---

### Step 11: Deep Research (Iteration 2) (0:25)

**What happens on screen:**
- 4 temporal-focused research streams:
  - 🔍 Q1: Advanced Temporal Decay Research
  - 🔍 Q2: Cross-Sequence Temporal Interactions
  - 🔍 Q3: Stable Training for Learnable Timescale
  - 🔍 Q4: Enhanced Multi-Token v2 Optimization
- Research findings appear

**Speaker talking points:**
- "Temporal-focused deep research"
- "Q1: TiM4Rec shows 2-3× improvement; 3-tier system (fast ~3h, medium ~24h, slow ~168h)"
- "Q2: Mamba-2 proves Linear Attention ≡ SSM"
- "Q3: Use log(τ) parameterization for stability, temperature annealing"
- "Q4: GQA with K=4 provides 2-4× speedup, top-k sparse attention"

**Action:** Click "✅ Generate Proposals (Iteration 2)"

---

### Step 12: Proposal Generation (Iteration 2) (0:20)

**What happens on screen:**
- 2 proposal sections:
  - 💡 Temporal Refinement Proposal
  - 🔗 Cross-Sequence Interaction Proposal

**Speaker talking points:**
- "Temporal Refinement: Multi-timescale FLUID v2 with learnable τ per category"
- "  τ_fast ≈ 3h (news, trending), τ_medium ≈ 24h (default), τ_slow ≈ 168h (evergreen)"
- "Cross-Sequence: Enhanced Multi-Token v2 with GQA + sparse attention"
- "  Expected: +0.6% NDCG with only +12% latency (down from +25%)"

**Action:** Click "✅ Synthesize Unified Architecture"

---

### Step 13: Unified Architecture Synthesis (Iteration 2) (0:20)

**What happens on screen:**
- SYNAPSE v2 architecture synthesis
- Key change: Multi-Timescale FLUID v2 formula shown
- Training stabilization techniques listed

**Speaker talking points:**
- "SYNAPSE v2 integrates FLUID v2 + Multi-Token v2"
- "Key change: τ(x) is now a weighted sum of learnable timescales"
- "Training stability: log-space parameterization, temperature annealing, separation regularization"

**Action:** Click "✅ Implement SYNAPSE v2"

---

### Step 14: Implementation (Iteration 2) (0:20)

**What happens on screen:**
- Code generation for enhanced components:
  - advanced_fluid_decay.py
  - multi_timescale_layer.py
  - enhanced_multi_token.py
  - temporal_attention.py

**Speaker talking points:**
- "Enhanced temporal implementation"
- "AdvancedFLUIDDecay: 3 learnable timescales with log_tau for stability"
- "TimescalePredictor MLP routes items to appropriate timescales"
- "EnhancedMultiToken: GQA K=4, sparse top-k=4 attention"

**Action:** Click "✅ Run Experiments (Iteration 2)"

---

### Step 15: Experiments (Iteration 2) (0:25)

**What happens on screen:**
- Experiment results comparison: v1 vs v2
- Learned timescales displayed (2.8h, 26.3h, 158.2h)
- **+0.5% temporal items** highlighted (was -2.5%)

**Speaker talking points:**
- "SYNAPSE v2 experiments - watch the timescales learn!"
- "Learned values: τ_fast=2.8h, τ_medium=26.3h, τ_slow=158.2h"
- "These match domain expectations perfectly"
- "**Temporal items: -2.5% → +0.5%** - a +3.0% swing!"
- "Multi-Token latency: +25% → +12% (within budget)"

**Action:** Click "✅ Analyze Results (Iteration 2)"

---

### Step 16: Analysis (Iteration 2) (0:20)

**What happens on screen:**
- Iteration comparison table (v1 vs v2)
- Improvement attribution breakdown
- Evolution analysis validation

**Speaker talking points:**
- "Evolution success! Both diagnosed issues fixed"
- "Temporal items: -2.5% → +0.5% (+3.0% swing - exactly where diagnosis pointed)"
- "Multi-Token latency: +25% → +12% (within budget)"
- "NDCG recovery: -9.3% → -5.2% (+4.1% quality recovery, matching HSTU+sampling baseline)"
- "Throughput maintained at 2.3×"

**Action:** Click "✅ Complete Workflow" or "🔄 Start Iteration 3"

---

### Step 17: Final Summary & Completion (0:25)

**What happens on screen:**
- Final summary with evolution trajectory
- Iteration 1 → Iteration 2 comparison table
- Key takeaways highlighted

**Speaker talking points:**
- "RankEvolve workflow complete!"
- "Iteration 1 built full SYNAPSE and diagnosed fixed τ limitation"
- "Iteration 2 deep-dived on temporal → achieved +3.0% swing on diagnosed area"
- "Key lesson: Diagnose before iterating, then validate improvement concentrates where expected"
- "This demonstrates how focused iteration based on experiment insights leads to meaningful improvements"
- "Thank you for watching the RankEvolve demo!"

---

## Recording Tips

### Pacing
- Let animations complete naturally - don't rush through progress sections
- Pause briefly (1-2s) when new sections appear to let viewers read
- **Speed up Iteration 2 (Steps 9-16) to 1.5× in post-production** - viewers understand the pattern
- Emphasize the Evolve transition (Step 9) - this is the key differentiator

### Key Narrative Beats

1. **Step 0**: "Research cycle with Evolve methodology"
2. **Step 2**: "5 reasoning patterns for query generation"
3. **Step 5**: "SYNAPSE v1 with fixed τ=24h - remember this"
4. **Step 7**: "SYNAPSE v1: -9.3% NDCG, 2× throughput - quality vs efficiency trade-off"
5. **Step 8**: "Root cause: τ=24h wrong for news/trending, -2.5% on temporal items"
6. **Step 9**: "Evolve transition - loop back based on insights"
7. **Step 15**: "Timescales learned! 2.8h, 26.3h, 158.2h - matches expectations"
8. **Step 16**: "+4.1% quality recovery to match HSTU+sampling baseline"

### Screen Setup
- Resolution: 1920×1080 recommended
- Browser zoom: 100% for optimal text readability
- Close unnecessary browser tabs
- Dark mode for visual appeal

### Narration Style
- Conversational but professional
- Focus on the "why" not just the "what"
- Emphasize the diagnostic → fix → validate pattern
- Stress that non-focus areas stay unchanged (realistic for ML research)

---

## Post-Production Notes

- Add chapter markers at each step transition
- **Speed up Steps 9-16 (Iteration 2) to 1.5× with visual indicator**
- Add subtle zoom on key metrics when mentioned
- Include intro/outro slides with RankEvolve branding
- Consider adding visual "Evolve loop" animation at Step 9
- Add side-by-side comparison at Step 17 showing v1 vs v2 metrics

---

## Quick Reference Card

```
ITERATION 1 (Steps 0-8)
[0:00-0:25] Welcome & Plan Selection → Click "Execute Plan"
[0:25-1:00] Codebase Investigation → Click "Continue"
[1:00-1:40] Research Planning (Iter 1) → Click "Execute Research"
[1:40-2:25] Deep Research (Iter 1) → Click "Generate Proposals"
[2:25-3:00] Proposal Generation → Click "Synthesize"
[3:00-3:30] Architecture Synthesis → Click "Implement"
[3:30-4:00] Implementation (Iter 1) → Click "Run Experiments"
[4:00-4:30] Experiments (Iter 1) → Click "Analyze"
[4:30-4:55] Analysis (Iter 1) → Click "Start Iteration 2"

EVOLVE TRANSITION & ITERATION 2 (Steps 9-16)
[4:55-5:15] Evolve: Start Iteration 2 → Click "Continue"
[5:15-5:35] Research Planning (Iter 2)
[5:35-6:00] Deep Research (Iter 2)
[6:00-6:20] Proposal Generation (Iter 2)
[6:20-6:40] Architecture Synthesis (Iter 2)
[6:40-7:00] Implementation (Iter 2)
[7:00-7:25] Experiments (Iter 2)
[7:25-7:45] Analysis (Iter 2) → Click "Complete"

COMPLETION
[7:45-8:10] Final Summary
```

---

## Evolution Story Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION NARRATIVE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ITERATION 1: Full SYNAPSE Architecture (Steps 0-8)                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Built complete SYNAPSE v1: SSD-FLUID + PRISM + FLUID + Multi-Token      │
│  • SSD-FLUID: 2× throughput but -6% NDCG (trade-off)                       │
│  • PRISM: +2.6% cold-start (meets target)                                  │
│  • FLUID: -1% re-engagement ❌ (below target)                               │
│  • Multi-Token: +0.5% NDCG but +25% latency ⚠️                              │
│  • Overall: -9.3% NDCG vs -5.2% for HSTU+sampling baseline                 │
│                                                                             │
│  🔍 DIAGNOSIS: Fixed τ=24h HURTS temporal items (-2.5% vs +1.8% others)    │
│                                                                             │
│                              ↓ EVOLVE (Step 9)                              │
│                                                                             │
│  ITERATION 2: Targeted Fix (Steps 9-16)                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Dual Focus: Multi-timescale FLUID + Enhanced Multi-Token v2             │
│  • Research: TiM4Rec, 3-tier timescale, GQA optimization                   │
│  • Solution: Learned τ per category (fast/medium/slow)                      │
│                                                                             │
│  📊 RESULTS (Step 17):                                                      │
│  • NDCG: -9.3% → -5.2% (+4.1% quality recovery to match sampling baseline) │
│  • Re-engagement: -1% → +2% (+3% improvement)                              │
│  • Temporal items: -2.5% → +0.5% (+3.0% swing where fix applied!)          │
│  • Multi-Token latency: +25% → +12% (within budget)                        │
│                                                                             │
│  ✅ VALIDATION: Improvement concentrated exactly where diagnosis pointed    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Realistic Results Note

This demo presents **honest, realistic ML research results**:

| Metric | Traditional Demo | RankEvolve Demo |
|--------|------------------|-----------------|
| Claims | "+50× throughput, +18% CTR" | "2× throughput with -9.3% NDCG trade-off (vs -5.2% for simple sampling)" |
| Presentation | All wins, no trade-offs | Trade-offs explicitly shown |
| Iteration narrative | N/A | "Diagnosed problem → Fixed → Validated" |

**Why this matters:**
- Builds trust through honest assessment
- Shows realistic ML research dynamics
- Demonstrates the value of iterative diagnosis

---

## Extended 10-Minute Version

For a more detailed demo with full explanations:

| Step | Section | Duration | Cumulative |
|------|---------|----------|------------|
| 0 | Welcome & Plan Selection | 0:35 | 0:35 |
| 1 | Codebase Investigation | 0:50 | 1:25 |
| 2 | Research Planning (Iter 1) | 0:55 | 2:20 |
| 3 | Deep Research (Iter 1) | 1:00 | 3:20 |
| 4 | Proposal Generation (Iter 1) | 0:50 | 4:10 |
| 5 | Architecture Synthesis (Iter 1) | 0:45 | 4:55 |
| 6 | Implementation (Iter 1) | 0:45 | 5:40 |
| 7 | Experiments (Iter 1) | 0:45 | 6:25 |
| 8 | Analysis (Iter 1) | 0:40 | 7:05 |
| 9 | Evolve Transition | 0:30 | 7:35 |
| 10 | Research Planning (Iter 2) | 0:30 | 8:05 |
| 11 | Deep Research (Iter 2) | 0:35 | 8:40 |
| 12 | Proposal Generation (Iter 2) | 0:30 | 9:10 |
| 13 | Architecture Synthesis (Iter 2) | 0:25 | 9:35 |
| 14 | Implementation (Iter 2) | 0:30 | 10:05 |
| 15 | Experiments (Iter 2) | 0:35 | 10:40 |
| 16 | Analysis (Iter 2) | 0:30 | 11:10 |
| 17 | Final Summary | 0:35 | 11:45 |

**Tips for 10-minute version:**
- Let all progress animations play at 1× speed
- Show more file content when clicking research results
- Expand on the diagnostic insights in Steps 8 and 16
- Include Q&A pauses or callouts for key innovations
