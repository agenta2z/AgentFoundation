# Model Improvement Hypothesis Report for model_roo_v0.py

**Date:** December 11, 2025 **Target Model:** `model_roo_v0.py`
(MainFeedMTMLROO) **Primary Goal:** Improve NE (Normalized Entropy) **Secondary
Goal:** Reduce model size to improve QPS with minimal NE regression

---

## Executive Summary

This report presents a comprehensive analysis of the `model_roo_v0.py`
architecture and proposes hypotheses for improving model performance (NE) and
efficiency (QPS). Based on analysis of the model architecture, internal Meta
research on recommendation systems, and code documentation on module
implementations, we identify **11 actionable hypotheses** spanning architecture
changes, module optimization, and efficiency improvements.

Key findings from the analysis:

- The current CGCModule/PLE implementation uses an **asymmetric gating design**
  where only 2 of 4 task groups benefit from shared experts
- The model uses a **single-layer PLE** despite research showing multi-layer
  progressive extraction improves task specialization
- **HSTU integration** with longer sequence lengths and more layers shows
  significant NE gains (Slimper V2: +0.405% NE)
- **Search-based Interest Modeling (SIM)** provides additive gains (+0.1225% NE)
  on top of HSTU improvements
- Hard-coded configurations (loss weights, gate dimensions) present optimization
  opportunities

---

## Phase 1: Context Analysis

### Model Architecture Overview

The `MainFeedMTMLROO` model implements a sophisticated multi-task learning
architecture with the following key components:

1. **Sparse/Dense Processing (RO/NRO Split)**
   - Request-Only (RO): User features, shared across candidates
   - Non-Request-Only (NRO): Candidate-specific features

2. **HSTU Module (HstuCintModule)**
   - Sequential user behavior modeling
   - DataFM foundation model features
   - Transformer encoder with PMA pooling

3. **Interaction Architecture**
   - Deep Compressed Factorization Networks (DCFN)
   - DeepFM for embedding interactions
   - Multi-Head Talking Attention (MHTA) with 8 heads

4. **Content Intelligence (CI) Module**
   - SlimDHEN for content-specific embeddings
   - 128-dim output integrated with main pathway

5. **CGCModule/PLE Layer**
   - 4 task groups, 1 expert per group
   - **Asymmetric gating**: only groups 2,3 access shared experts
   - Single PLE layer (no progressive stacking)

6. **Task Architecture**
   - 65+ tasks across prediction, auxiliary, and bias correction
   - Hard-coded loss weights for specific tasks

### Key Research Insights

From internal Meta research (`internal_research_metamate.md`):

| Research Area       | Key Finding                                                           | Potential Application       |
| ------------------- | --------------------------------------------------------------------- | --------------------------- |
| **HSTU Scaling**    | 200-300% capacity savings, 2.7x inference speedup                     | Optimize HSTU configuration |
| **Slimper V2**      | +0.405% NE with deeper model (3→6 layers), longer sequence (512→2048) | Sequence/depth scaling      |
| **SIM Integration** | +0.1225% NE with search-based interest modeling                       | Add SIM module              |
| **Slimper + SIM**   | +0.575% NE (additive gains, no cannibalization)                       | Combined approach           |
| **Semantic IDs**    | 682x compression, cold-start improvements                             | Content representation      |
| **LLaTTE**          | 2.55x higher scaling efficiency                                       | Architecture modernization  |
| **DeepSeek MoE**    | Auxiliary-loss-free load balancing, fine-grained experts              | PLE optimization            |

### Identified Opportunities

1. **CGCModule Asymmetric Design (Lines 505-519)**
   - Gates 0,1 have trivial softmax (single expert)
   - Potential negative transfer protection OR missed optimization

2. **Single-Layer PLE**
   - Original PLE paper shows multi-layer improves task specialization
   - Current: single CGCModule layer

3. **Hard-Coded Loss Weights (Lines 955-970)**
   - Fixed weights: `linear_vpvd=10.3`, `share=1.5`, `comment_pairwise=10.0`
   - No learnable scaling for most tasks

4. **HSTU Configuration**
   - Current: Fixed sequence length, 2 encoder layers
   - Research shows significant gains from scaling

5. **MoE Design**
   - Current: 1 expert per group
   - Research: Fine-grained experts + always-active shared experts

---

## Phase 2: Hypothesis Formulation

### Hypothesis 1: Symmetric CGCModule Gating

**Category:** Model Module Refactor

**Current State:** The CGCModule implements asymmetric gating where only task
groups 2 and 3 can access the shared expert. Groups 0 and 1 have trivial gating
(single expert = 100% weight).

```python
# Lines 505-519: Asymmetric gate output dimensions
out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)
```

**Hypothesis:** Making all gates symmetric (all 4 groups access shared experts)
will improve NE by enabling knowledge sharing across all task groups.

**Supporting Evidence:**

- Original PLE paper designs symmetric expert access for all gates
- DeepSeek research shows shared experts effectively capture common patterns
- Task groups 0,1 may benefit from shared representations currently unavailable
  to them

**Contradicting Evidence:**

- The asymmetric design may have been intentionally chosen to protect
  task-specific learning in groups 0,1
- Could have been tuned in previous experiments (hard-coded nature suggests
  historical optimization)
- Risk of negative transfer if groups 0,1 tasks conflict with groups 2,3

**Success Probability:** Medium (40-70%) **Potential Impact:** Medium
(+0.05-0.15% NE)

**Justification:** The undocumented nature of the asymmetric design suggests it
may not have been thoroughly validated. However, it could also represent
institutional knowledge from past experiments.

---

### Hypothesis 2: Multi-Layer Progressive PLE

**Category:** Model Module Change

**Current State:** Single CGCModule layer processes all task groups.

**Hypothesis:** Stacking 2-3 CGC layers with progressive refinement will improve
task-specific learning and NE.

**Supporting Evidence:**

- Original PLE paper: "Progressive" refers to stacked CGC layers for incremental
  specialization
- Research shows later layers can specialize more for individual tasks
- Each layer builds on learned representations from previous layers

**Contradicting Evidence:**

- Increased latency from additional layers
- More parameters = more memory and compute
- Diminishing returns possible if current single layer is well-tuned

**Success Probability:** Medium (50-65%) **Potential Impact:** Medium-High
(+0.1-0.3% NE)

**Justification:** Multi-layer PLE is a proven technique in the literature. The
current implementation's single-layer design may be leaving performance on the
table.

---

### Hypothesis 3: Increase Experts per Task Group

**Category:** Model Module Change

**Current State:** `num_expert_in_group = 1` (single expert per task group)

**Hypothesis:** Increasing to 2-4 experts per task group with proper gating will
improve model capacity and NE.

**Supporting Evidence:**

- DeepSeek: Fine-grained segmentation into 256 experts enables
  hyper-specialization
- More experts = more capacity for learning diverse patterns
- Current design limits flexibility of representation learning

**Contradicting Evidence:**

- More experts increase compute and memory requirements
- Risk of expert collapse without proper load balancing
- May require auxiliary loss or bias-based balancing

**Success Probability:** Medium (45-60%) **Potential Impact:** Medium (+0.1-0.2%
NE)

**Justification:** The MoE literature strongly supports multiple experts for
improved representation learning, but requires careful implementation.

---

### Hypothesis 4: HSTU Sequence Length Scaling (512 → 2048)

**Category:** Model Ad-hoc Change

**Current State:** HSTU processes sequences with `max_seq_len = 460` (512 -
50 - 2)

**Hypothesis:** Increasing HSTU sequence length to 2048 (Slimper V2 style) will
capture longer user history and improve NE.

**Supporting Evidence:**

- Slimper V2: +0.405% NE with 512→2048 sequence scaling
- Research: "Longer sequence, richer features... key to performance gain"
- User behavior patterns extend beyond 512 interactions

**Contradicting Evidence:**

- 4x longer sequences = significant compute increase
- Training QPS regression (-5.99% for Slimper V2)
- Memory requirements grow quadratically with sequence length for attention

**Success Probability:** High (70-85%) **Potential Impact:** High (+0.3-0.5% NE)

**Justification:** This is a well-validated approach from Slimper V2 with clear
NE gains documented.

---

### Hypothesis 5: HSTU Depth Scaling (2 → 6 Layers)

**Category:** Model Ad-hoc Change

**Current State:** HSTU uses 2 encoder layers

**Hypothesis:** Increasing HSTU encoder depth to 6 layers (Slimper V2 style)
will improve sequential pattern learning and NE.

**Supporting Evidence:**

- Slimper V2: Deeper model (3→6 layers) contributed to +0.405% NE
- Deeper transformers learn more complex temporal dependencies
- Combined with sequence scaling, provides compound gains

**Contradicting Evidence:**

- More layers = more latency and compute
- Diminishing returns beyond certain depth
- Need activation checkpointing for memory efficiency

**Success Probability:** High (65-80%) **Potential Impact:** Medium-High
(+0.2-0.4% NE)

**Justification:** Depth scaling is a proven technique with documented benefits
in the Slimper V2 research.

---

### Hypothesis 6: Add Search-based Interest Modeling (SIM)

**Category:** Additional Feature

**Current State:** No SIM module; relies solely on HSTU for sequential modeling

**Hypothesis:** Adding SIM module for long-range user history matching will
provide additive NE gains on top of HSTU.

**Supporting Evidence:**

- SIM alone: +0.1225% NE
- Slimper + SIM: +0.575% NE (additive, no cannibalization)
- SIM handles 22,000 max sequence length vs HSTU's 512
- Two-stage (GSU + ESU) efficiently handles very long histories

**Contradicting Evidence:**

- Significant implementation complexity
- Training QPS impact: -6% for SIM alone, -16.29% combined
- GPU memory increase: +20.78% for combined approach

**Success Probability:** High (75-85%) **Potential Impact:** High (+0.1-0.15%
NE, additive to HSTU gains)

**Justification:** SIM has been validated to provide additive gains with no
cannibalization. The research clearly shows complementary benefits.

---

### Hypothesis 7: Learnable Loss Weights for All Tasks

**Category:** Model Ad-hoc Change

**Current State:** Hard-coded loss weights in `MainFeedMTMLLoss`:

```python
if task_name in {"linear_vpvd", "kd_linear_vpvd", "kd_bce_linear_vpvd"}:
    task_weight = 10.3
elif task_name in {"share"}:
    task_weight = 1.5
elif task_name in {"comment_pairwise"}:
    task_weight = 10.0
```

**Hypothesis:** Replacing hard-coded weights with learnable loss scaling (using
`LearnableLossScale`) for all tasks will enable automatic task balancing and
improve overall NE.

**Supporting Evidence:**

- `LearnableLossScale` module already exists and is used for some tasks
- Dynamic weighting can adapt to task difficulty during training
- Removes manual tuning burden and potential suboptimality

**Contradicting Evidence:**

- Hard-coded weights may represent optimized values from extensive tuning
- Learnable weights could be unstable early in training
- Risk of some tasks dominating others

**Success Probability:** Medium (40-55%) **Potential Impact:** Low-Medium
(+0.05-0.1% NE)

**Justification:** The existing infrastructure for learnable loss scaling
suggests it's a validated approach, but current hard-coded values likely
represent tuned baselines.

---

### Hypothesis 8: Reduce MHTA Heads for QPS Improvement

**Category:** Model Module Refactor (Efficiency)

**Current State:** MHTA uses 8 heads with 1024-dim each, plus gating and
residual modules

**Hypothesis:** Reducing MHTA heads from 8 to 4-6 will improve QPS with minimal
NE regression.

**Supporting Evidence:**

- MHTA is computationally expensive (8 DCN heads + gating)
- Research shows attention can be efficiently reduced with proper design
- Inference QPS directly benefits from fewer attention computations

**Contradicting Evidence:**

- MHTA is a core interaction module; reduction may hurt expressiveness
- Heterogeneous feature interactions are critical per prior knowledge
- May need compensating capacity elsewhere

**Success Probability:** Medium (45-60%) **Potential Impact:** Medium QPS
improvement (10-20%), possible NE regression (-0.05 to -0.1%)

**Justification:** This is a reasonable efficiency vs. performance trade-off
exploration.

---

### Hypothesis 9: Remove or Simplify Content Intelligence Module

**Category:** Model Module Refactor (Efficiency)

**Current State:** CI module processes content embeddings through SlimDHEN
(128-dim output)

**Hypothesis:** Ablating CI module will determine its contribution; if minimal,
removing it improves QPS.

**Supporting Evidence:**

- CI module adds parameters and compute
- SlimDHEN is a separate processing pathway
- Content features may be redundant with main pathway features

**Contradicting Evidence:**

- CI was likely added for specific value (IFM Mini, KGID embeddings)
- May hurt content-specific task performance
- 128-dim is relatively small overhead

**Success Probability:** Low-Medium (30-45%) **Potential Impact:** Medium QPS
improvement if removed, unknown NE impact

**Justification:** This is an ablation study to understand module contribution
before deciding on removal.

---

### Hypothesis 10: DeepSeek-Style Auxiliary-Loss-Free Load Balancing

**Category:** Model Module Refactor

**Current State:** CGCModule/PLE has no explicit load balancing (tasks are
pre-assigned to experts)

**Hypothesis:** Implementing bias-based routing (DeepSeek style) with dynamic
expert utilization balancing will improve expert specialization without gradient
conflicts.

**Supporting Evidence:**

- DeepSeek V3: Bias-based routing maintains load balance without hurting primary
  objective
- Current design has no mechanism for dynamic expert utilization
- Can enable more fine-grained expert pools

**Contradicting Evidence:**

- PLE's task-specific expert design makes load balancing less critical than LLM
  MoE
- Implementation complexity for marginal gains
- Different problem domain (task-based vs token-based routing)

**Success Probability:** Low (25-40%) **Potential Impact:** Low-Medium
(+0.02-0.08% NE)

**Justification:** The technique is proven in LLM domain but may not translate
directly to recommendation MTL.

---

### Hypothesis 11: Semantic ID Integration for Cold-Start Improvement

**Category:** Additional Feature

**Current State:** Uses traditional embedding tables for content representation

**Hypothesis:** Adding semantic ID representations (RQ-VAE or RQ-KMeans) will
improve cold-start performance and overall NE through better content
generalization.

**Supporting Evidence:**

- Research: 682x compression while maintaining reconstruction quality
- Production wins: IG Reels +0.11% teen TS, FB Reels +0.77% messenger DAU
- Unified vocabularies enable cross-content understanding

**Contradicting Evidence:**

- Significant infrastructure changes required
- May require OVIS platform integration
- Cold-start may not be the primary bottleneck for NE

**Success Probability:** Medium (40-55%) **Potential Impact:** Medium (+0.1-0.2%
NE for cold-start scenarios)

**Justification:** Semantic IDs are a proven technology but require significant
implementation effort.

---

## Phase 3: Hypothesis Ranking & Summary

### Final Ranked Hypothesis List

| Rank   | Hypothesis                       | Category              | Success Prob.       | Impact       | Priority Justification                         |
| ------ | -------------------------------- | --------------------- | ------------------- | ------------ | ---------------------------------------------- |
| **1**  | H4: HSTU Sequence Length Scaling | Model Ad-hoc Change   | High (70-85%)       | High         | Well-validated with +0.405% NE from Slimper V2 |
| **2**  | H6: Add SIM Module               | Additional Feature    | High (75-85%)       | High         | Additive +0.1225% NE, no cannibalization       |
| **3**  | H5: HSTU Depth Scaling           | Model Ad-hoc Change   | High (65-80%)       | Medium-High  | Compound gains with H4, proven approach        |
| **4**  | H2: Multi-Layer Progressive PLE  | Model Module Change   | Medium (50-65%)     | Medium-High  | Core PLE design improvement                    |
| **5**  | H1: Symmetric CGCModule Gating   | Model Module Refactor | Medium (40-70%)     | Medium       | Low-risk design improvement                    |
| **6**  | H3: Increase Experts per Group   | Model Module Change   | Medium (45-60%)     | Medium       | MoE best practice                              |
| **7**  | H7: Learnable Loss Weights       | Model Ad-hoc Change   | Medium (40-55%)     | Low-Medium   | Infrastructure exists                          |
| **8**  | H8: Reduce MHTA Heads (QPS)      | Model Module Refactor | Medium (45-60%)     | Medium (QPS) | Secondary goal: efficiency                     |
| **9**  | H11: Semantic ID Integration     | Additional Feature    | Medium (40-55%)     | Medium       | Long-term improvement                          |
| **10** | H9: CI Module Ablation (QPS)     | Model Module Refactor | Low-Medium (30-45%) | Medium (QPS) | Ablation study for efficiency                  |
| **11** | H10: Bias-Based Load Balancing   | Model Module Refactor | Low (25-40%)        | Low-Medium   | Domain transfer uncertainty                    |

### Recommended Implementation Strategy

**Phase A: High-Impact NE Improvements (Immediate)**

1. Implement H4 + H5 together (HSTU scaling) - Expected: +0.4-0.5% NE
2. Implement H6 (SIM) - Expected: +0.1-0.15% NE additive

**Phase B: Architecture Optimization (Short-term)** 3. Implement H2 (Multi-layer
PLE) 4. Implement H1 (Symmetric gating) 5. Experiment with H3 (More experts)

**Phase C: Efficiency Improvements (Parallel track)** 6. Run H9 ablation to
understand CI contribution 7. Implement H8 if QPS is critical and CI ablation
shows limited impact

**Phase D: Advanced Features (Long-term)** 8. H11: Semantic IDs (requires
infrastructure work) 9. H10: Advanced load balancing (if moving to finer-grained
MoE)

---

## Appendix: Key Code Locations

| Component                  | File                     | Lines           |
| -------------------------- | ------------------------ | --------------- |
| CGCModule Asymmetric Gates | `model_roo_v0.py`        | 505-519         |
| PLE Layer Creation         | `model_roo_v0.py`        | 520-524         |
| Hard-Coded Loss Weights    | `model_roo_v0.py`        | 955-970         |
| HSTU Module                | `pytorch_modules_roo.py` | HstuCintModule  |
| MHTA Configuration         | `model_roo_v0.py`        | 181-288         |
| CI Module (SlimDHEN)       | `model_roo_v0.py`        | 327-352         |
| Task Architecture          | `pytorch_modules_roo.py` | TaskArchROOWrap |

---

## References

1. Internal Meta Research: `internal_research_metamate.md`
2. Code Documentation: `coscience/context/code_docs/models/`
3. PLE Paper: Tang et al., "Progressive Layered Extraction (PLE)" RecSys 2020
4. DeepSeek-V3 Technical Report (2024)
5. Slimper + SIM Proposal Documentation
