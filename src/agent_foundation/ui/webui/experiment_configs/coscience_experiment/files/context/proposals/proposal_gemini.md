# Hypothesis Report: CFR Model Performance & Efficiency Improvements

## Executive Summary

This report presents a comprehensive analysis of potential improvements for
`model_roo_v0.py` based on:

1. Current architecture analysis (5.1B parameters, 65+ tasks, ROO dual-path
   architecture)
2. External research synthesis (HSTU, Wukong, OneRec, DFGR, DeepSeek MoE
   innovations)
3. Code documentation review (PLE/CGCModule, HSTU DataFM, architecture patterns)
4. Prior knowledge (heterogeneous feature interactions, deeper models, efficient
   memory handling)

**Goals:**

- **Primary:** Improve NE (Normalized Entropy) with reasonable model size
  increase
- **Secondary:** Reduce model size for QPS improvement while maintaining NE

---

## Phase 1: Context Analysis & Opportunity Identification

### 1.1 Current Architecture Strengths

- **ROO dual-path (RO/NRO)**: Efficient pairwise ranking with feature separation
- **MHTA (8 heads, 1 stack)**: Multi-Head Talking Attention with DCN-based heads
- **HSTU with DataFM**: Sequential user behavior modeling (512 seq, 192 dim)
- **PLE/CGCModule**: Multi-task learning with 4 expert groups
- **Content Intelligence**: SlimDHEN for content embeddings

### 1.2 Identified Gaps & Opportunities

1. **Asymmetric PLE design**: Gates 0,1 have trivial gating (only 1 expert
   accessible)
2. **Single CGC layer**: No progressive layering as in original PLE paper
3. **Fixed MHTA stacking**: Only 1 stack, potential for deeper interaction
4. **Hard-coded loss weights**: Task weights are fixed, not adaptive
5. **Standard attention in HSTU**: Quadratic complexity, not exploiting HSTU
   pointwise aggregation
6. **No shared expert load balancing**: Unlike DeepSeek's bias-based approach

### 1.3 Research Techniques Applicable to Target

| Research     | Technique                               | Applicability                               |
| ------------ | --------------------------------------- | ------------------------------------------- |
| Meta HSTU    | Pointwise aggregation                   | Already partially integrated, can optimize  |
| Wukong SFMs  | Stacked Factorization Machines          | High - scaling law for feature interactions |
| DFGR         | Dual-flow (item/action separation)      | High - reduces sequence length              |
| DeepSeek MoE | Fine-grained experts + shared isolation | Medium - applicable to PLE                  |
| OneRec       | DPO alignment                           | Low - requires infrastructure change        |
| PLUM         | Semantic IDs                            | Low - major architectural change            |

---

## Phase 2: Hypothesis Formulation

### Hypothesis 1: Symmetric PLE Gate Design

**Category:** Model Module Refactor

**Statement:** Convert the asymmetric CGCModule gate design to symmetric design
where all 4 task groups have access to both their task-specific expert AND the
shared expert.

**Current Implementation (Lines 505-519 in model_roo_v0.py):**

```python
out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)  # Gates 0,1: 1, Gates 2,3: 2
```

**Proposed Change:**

```python
out_features=num_expert_in_group + len(shared_experts)  # All gates: 2
```

**Supporting Evidence:**

- Original PLE paper shows symmetric design improves task correlation learning
- Gates 0,1 currently have NO gating benefit (trivial softmax over 1 expert =
  1.0)
- Code documentation explicitly flags this as "potential optimization
  opportunity"
- DeepSeek's shared expert isolation shows shared experts capture cross-task
  patterns effectively

**Contradicting Evidence:**

- Asymmetric design may have been intentional after tuning (hard to verify
  without experiment history)
- May increase gradient interference if tasks in groups 0,1 conflict with shared
  knowledge

**Success Probability:** Medium (50%) **Impact:** Low-Medium (NE: +0.01-0.05%)
**Model Size Change:** Minimal (~0.01% increase from gate parameters)

---

### Hypothesis 2: Progressive Layered PLE (Multi-Layer CGC)

**Category:** Model Module Change

**Statement:** Stack 2-3 CGCModule layers to enable progressive task
specialization, where early layers capture shared patterns and later layers
specialize for individual tasks.

**Current Implementation:** Single CGCModule layer **Proposed Change:**

```python
num_ple_layers = 2  # or 3
self.ple_layers = nn.ModuleList([CGCModule(...) for _ in range(num_ple_layers)])
```

**Supporting Evidence:**

- Original PLE paper "Progressive Layered Extraction" shows multi-layer design
  core to architecture
- Code documentation states: "potential optimization opportunity: adding more
  layers could improve task performance"
- Progressive information flow: Layer 1 (shared) → Layer N (task-specific)
- Aligns with user prior knowledge: "Deeper model... key to performance gain"

**Contradicting Evidence:**

- Increases model size and latency proportionally
- May require careful gradient scaling to prevent vanishing/exploding gradients
- Additional training cost

**Success Probability:** Medium-High (60%) **Impact:** Medium (NE: +0.05-0.15%)
**Model Size Change:** +25-50% in PLE module parameters (~20M additional)

---

### Hypothesis 3: Increase Expert Count per Task Group

**Category:** Model Module Change

**Statement:** Increase `num_expert_in_group` from 1 to 2-3 to provide richer
task-specific representations and true MoE gating within each task group.

**Current Implementation:**

```python
num_expert_in_group = 1  # Single expert per task group
```

**Proposed Change:**

```python
num_expert_in_group = 2  # Or 3
```

**Supporting Evidence:**

- Current design has trivial intra-group gating (only 1 expert to route to)
- MoE literature (DeepSeek, Mixtral) shows multiple experts improve
  capacity/specialization
- Code documentation explicitly recommends this as "Optimization Recommendation
  #1"
- More experts allow different "views" of task-specific patterns

**Contradicting Evidence:**

- Risk of expert collapse without load balancing (though PLE has implicit
  task-based balancing)
- Increased memory footprint proportional to expert count
- May require careful initialization to avoid expert redundancy

**Success Probability:** Medium (55%) **Impact:** Medium (NE: +0.03-0.10%)
**Model Size Change:** +15-30% in PLE module parameters

---

### Hypothesis 4: MHTA Stack Depth Increase

**Category:** Model Module Change

**Statement:** Increase MHTA `num_stacks` from 1 to 2 to enable deeper feature
interaction refinement through residual gating.

**Current Implementation:**

```python
mhta_num_stack = 1  # Single stack
```

**Proposed Change:**

```python
mhta_num_stack = 2  # Two stacks for progressive refinement
```

**Supporting Evidence:**

- Wukong research shows stacked feature interactions capture higher-order
  dependencies (2^L order at layer L)
- Prior knowledge states "heterogeneous feature interactions are critical"
- MHTA already has infrastructure for multi-stack (gating_module,
  short_cut_modules, res_modules all support multiple stacks)
- Each additional stack adds residual connection + gating for refined
  interactions

**Contradicting Evidence:**

- Increases latency (more sequential computation)
- Potential gradient flow issues with deeper stacks
- May not be beneficial if first stack already captures key interactions

**Success Probability:** Medium (50%) **Impact:** Medium (NE: +0.02-0.08%)
**Model Size Change:** +20% in MHTA module parameters

---

### Hypothesis 5: Dual-Flow Sequence Encoding (DFGR-inspired)

**Category:** Model Module Refactor

**Statement:** Refactor HSTU to use dual-flow architecture separating item
embeddings and action embeddings into parallel streams with cross-attention
gates, inspired by Meituan's DFGR.

**Current Implementation:** HSTU processes interleaved item-action sequences (2N
sequence length)

**Proposed Change:**

- Item Flow: [Item_1, Item_2, ...] processed separately
- Action Flow: [Action_1, Action_2, ...] processed separately
- Cross-attention gates for information exchange

**Supporting Evidence:**

- DFGR reports 2x faster training, 4x faster inference vs standard HSTU
- Sequence length reduction from 2N to N significantly reduces attention
  complexity
- Research shows this matches HSTU accuracy while improving efficiency
- Directly addresses QPS improvement goal

**Contradicting Evidence:**

- Requires significant refactoring of HSTU module
- Cross-attention gates add implementation complexity
- May lose some item-action causal relationships that interleaving captures

**Success Probability:** Medium (45%) **Impact:** High for QPS (2-4x inference
speedup), Neutral for NE **Model Size Change:** Minimal (restructuring, not
adding parameters)

---

### Hypothesis 6: Learnable Loss Scale Expansion

**Category:** Model Ad-hoc Change

**Statement:** Expand `learnable_loss_reg_scale` to all major tasks instead of
the current limited implementation, allowing the model to adaptively balance
task importance during training.

**Current Implementation:** Only specific tasks have `LearnableLossScale`

```python
if one_task_config.learnable_loss_reg_scale is not None:
    self.learnable_loss_scale_module_dict[task_name] = LearnableLossScale(...)
```

**Proposed Change:** Enable learnable loss scaling for all prediction tasks with
uncertainty weighting.

**Supporting Evidence:**

- Current hard-coded weights (e.g., `task_weight = 10.3` for vpvd, `1.0` for
  most tasks) may be suboptimal
- Uncertainty-weighted losses (Kendall & Gal) have shown consistent improvements
  in MTL
- Amazon PEMBOT research shows adaptive task weighting critical for
  Pareto-optimal solutions
- Allows model to self-balance conflicting task gradients

**Contradicting Evidence:**

- Hard-coded weights may have been carefully tuned for this specific model
- Learnable weights may destabilize training initially
- Adds overhead for loss scale computation

**Success Probability:** Medium (50%) **Impact:** Low-Medium (NE: +0.01-0.05%)
**Model Size Change:** Negligible

---

### Hypothesis 7: Wukong-Style Stacked Factorization Machines

**Category:** Model Module Change

**Statement:** Replace or augment DeepCrossNet (DCN) heads in MHTA with Stacked
Factorization Machines (SFMs) to better capture high-order feature interactions
with provable scaling properties.

**Current Implementation:**

```python
head_modules = nn.ModuleList([
    DeepCrossNet(input_dim=..., low_rank_dim=768, num_layers=2, ...)
    for _ in range(mhta_num_heads)
])
```

**Proposed Change:**

```python
head_modules = nn.ModuleList([
    StackedFactorizationMachine(num_layers=3, ...)  # 2^3 = 8th order interactions
    for _ in range(mhta_num_heads)
])
```

**Supporting Evidence:**

- Wukong research demonstrates SFMs exhibit scaling laws (performance improves
  predictably with compute)
- DCN-based approaches saturate at certain complexity levels
- Meta's own research shows SFMs outperform MLPs/DCN at higher compute budgets
- Aligns with prior knowledge on "heterogeneous feature interactions"

**Contradicting Evidence:**

- Requires new module implementation (SFM not in current codebase)
- May require extensive hyperparameter tuning
- Increased compute per head

**Success Probability:** Medium-High (55%) **Impact:** Medium-High (NE:
+0.05-0.15%) **Model Size Change:** +10-20% in MHTA heads

---

### Hypothesis 8: HSTU Sequence Length Extension

**Category:** Feature Engineering Change

**Statement:** Extend HSTU `max_seq_len` from 460 (512-50-2) to 1024-2048 to
capture longer-term user preferences, enabled by efficient attention mechanisms.

**Current Implementation:**

```python
self.max_seq_len = 512 - 50 - 2  # 460 tokens
```

**Proposed Change:**

```python
self.max_seq_len = 1024 - 50 - 2  # Or 2048 - 50 - 2
```

**Supporting Evidence:**

- Meta's HSTU paper shows speedups at 8192 sequence length vs FlashAttention
- Research states "long user sequences typical in industrial settings (thousands
  of historical clicks)"
- Prior knowledge: "longer sequence... key to performance gain"
- ByteDance LongRetriever shows ultra-long sequences improve retrieval quality

**Contradicting Evidence:**

- Requires more memory during training
- May increase latency without HSTU optimization
- Older interactions may have diminishing relevance

**Success Probability:** Medium (50%) **Impact:** Medium (NE: +0.03-0.10%)
**Model Size Change:** Embedding table expansion, ~5-10% increase

---

### Hypothesis 9: DCN Low-Rank Dimension Optimization

**Category:** Model Ad-hoc Change (QPS-focused)

**Statement:** Reduce DCN `low_rank_dim` from 768 to 256-512 in MHTA head
modules to improve QPS with minimal NE regression.

**Current Implementation:**

```python
DeepCrossNet(low_rank_dim=768, ...)
```

**Proposed Change:**

```python
DeepCrossNet(low_rank_dim=384, ...)  # 50% reduction
```

**Supporting Evidence:**

- Low-rank factorization often has diminishing returns at high ranks
- 768 is relatively large for cross-network; literature typically uses 256-512
- Directly reduces computation per DCN layer
- Targets secondary goal: QPS improvement with minimal NE impact

**Contradicting Evidence:**

- May lose capacity for complex feature interactions
- Current value may have been optimized for this model

**Success Probability:** High (65%) **Impact:** Low for NE (-0.01-0%), Medium
for QPS (+10-20%) **Model Size Change:** -15-25% in DCN parameters

---

### Hypothesis 10: Expert Bottleneck Architecture

**Category:** Model Module Refactor (QPS-focused)

**Statement:** Apply bottleneck design to PLE experts: down-project input →
process in smaller dimension → up-project output.

**Current Implementation:**

```python
nn.Sequential(
    nn.Linear(ple_input_dim, expert_layer_size),  # 1200 → 256
    SwishLayerNorm([expert_layer_size]),
    nn.Linear(expert_layer_size, expert_layer_size),  # 256 → 256
    SwishLayerNorm([expert_layer_size]),
)
```

**Proposed Change:**

```python
nn.Sequential(
    nn.Linear(ple_input_dim, 64),  # Down-project: 1200 → 64
    SwishLayerNorm([64]),
    nn.Linear(64, 128),  # Process: 64 → 128
    SwishLayerNorm([128]),
    nn.Linear(128, expert_layer_size),  # Up-project: 128 → 256
    SwishLayerNorm([expert_layer_size]),
)
```

**Supporting Evidence:**

- Bottleneck architectures (ResNet, EfficientNet) proven effective for parameter
  efficiency
- Code documentation recommends "Deeper Experts with Bottleneck" as optimization
- Reduces intermediate computation while maintaining output dimension
- DeepSeek uses fine-grained experts with smaller dimensions

**Contradicting Evidence:**

- May lose representational capacity in bottleneck
- Adds depth which could affect gradient flow

**Success Probability:** Medium (45%) **Impact:** Medium for QPS (+15-25%), Low
for NE (-0.01-0%) **Model Size Change:** -20-30% in expert parameters

---

### Hypothesis 11: Auxiliary Loss Weight Tuning for Dense/Interaction Arch

**Category:** Model Ad-hoc Change

**Statement:** Adjust the hard-coded auxiliary loss divisors (100 for interarch,
300 for densearch) to learnable or empirically optimized values.

**Current Implementation (Lines 729-733):**

```python
if key == "interarch_aux_loss_outputs":
    los /= 100
if key == "densearch_aux_loss_outputs":
    los /= 300
```

**Proposed Change:** Make divisors learnable or run hyperparameter search to
find optimal values.

**Supporting Evidence:**

- Hard-coded values suggest manual tuning without systematic optimization
- Auxiliary losses contribute to representation learning; improper scaling may
  hurt
- Similar to Hypothesis 6 but for auxiliary (not main) task losses

**Contradicting Evidence:**

- Values may have been carefully calibrated
- Changing may destabilize training

**Success Probability:** Low-Medium (40%) **Impact:** Low (NE: +0.01-0.03%)
**Model Size Change:** None

---

### Hypothesis 12: Content Intelligence Module Enhancement

**Category:** Model Module Change

**Statement:** Expand CI module from single SlimDHEN layer to 2-layer
architecture with attention mechanism between layers.

**Current Implementation:**

```python
ci_arch = SlimDHEN(layers=[SlimDHENLayer(...)])  # Single layer
```

**Proposed Change:**

```python
ci_arch = SlimDHEN(
    layers=[
        SlimDHENLayer(..., attention_fm=SimpleAttentionFM(...)),  # Layer 1 with attention
        SlimDHENLayer(...)  # Layer 2
    ]
)
```

**Supporting Evidence:**

- Content features (IFM Mini, KGID) are rich semantic signals deserving deeper
  processing
- Current CI module has `attention_fm=None` - potentially underutilized
- Prior knowledge: "richer features... key to performance gain"

**Contradicting Evidence:**

- CI module contributes 128-dim to final representation; deeper processing may
  have limited impact
- Increased latency for content processing

**Success Probability:** Low-Medium (40%) **Impact:** Low (NE: +0.01-0.04%)
**Model Size Change:** +5-10% in CI module parameters

---

### Hypothesis 13: Residual Connections in CGCModule

**Category:** Model Module Refactor

**Statement:** Add residual connections from PLE input to output to improve
gradient flow and allow experts to learn residual transformations.

**Current Implementation:** No skip connection

```python
output = weighted_sum(expert_outputs)
```

**Proposed Change:**

```python
output = weighted_sum(expert_outputs) + residual_projection(input)
```

**Supporting Evidence:**

- Code documentation lists "Add Residual Connections" as optimization
  recommendation
- Residual learning is standard in deep networks for gradient flow
- Allows experts to focus on residual refinement rather than full transformation

**Contradicting Evidence:**

- May reduce expert specialization (experts can "do nothing" and rely on
  residual)
- Requires careful initialization

**Success Probability:** Medium (50%) **Impact:** Low (NE: +0.01-0.03%) **Model
Size Change:** +1-2% (projection layer)

---

## Phase 3: Hypothesis Ranking

### Priority 1 - High Impact, Actionable (NE Focus)

| Rank | Hypothesis                | Category       | Success Prob | NE Impact   | Model Size  |
| ---- | ------------------------- | -------------- | ------------ | ----------- | ----------- |
| 1    | H7: Wukong SFMs           | Module Change  | 55%          | +0.05-0.15% | +10-20%     |
| 2    | H2: Progressive PLE       | Module Change  | 60%          | +0.05-0.15% | +25-50% PLE |
| 3    | H8: HSTU Seq Extension    | Feature Change | 50%          | +0.03-0.10% | +5-10%      |
| 4    | H3: Expert Count Increase | Module Change  | 55%          | +0.03-0.10% | +15-30% PLE |

### Priority 2 - Moderate Impact, Low Risk

| Rank | Hypothesis                    | Category        | Success Prob | NE Impact   | Model Size |
| ---- | ----------------------------- | --------------- | ------------ | ----------- | ---------- |
| 5    | H1: Symmetric PLE Gates       | Module Refactor | 50%          | +0.01-0.05% | Minimal    |
| 6    | H4: MHTA Stack Increase       | Module Change   | 50%          | +0.02-0.08% | +20% MHTA  |
| 7    | H6: Learnable Loss Scale      | Ad-hoc Change   | 50%          | +0.01-0.05% | Negligible |
| 8    | H13: CGC Residual Connections | Module Refactor | 50%          | +0.01-0.03% | +1-2%      |

### Priority 3 - QPS Focus (Secondary Goal)

| Rank | Hypothesis                 | Category        | Success Prob | QPS Impact | NE Impact |
| ---- | -------------------------- | --------------- | ------------ | ---------- | --------- |
| 9    | H5: Dual-Flow HSTU         | Module Refactor | 45%          | +2-4x      | Neutral   |
| 10   | H9: DCN Low-Rank Reduction | Ad-hoc Change   | 65%          | +10-20%    | -0.01-0%  |
| 11   | H10: Expert Bottleneck     | Module Refactor | 45%          | +15-25%    | -0.01-0%  |

### Priority 4 - Exploratory

| Rank | Hypothesis                 | Category      | Success Prob | NE Impact   | Model Size |
| ---- | -------------------------- | ------------- | ------------ | ----------- | ---------- |
| 12   | H11: Aux Loss Tuning       | Ad-hoc Change | 40%          | +0.01-0.03% | None       |
| 13   | H12: CI Module Enhancement | Module Change | 40%          | +0.01-0.04% | +5-10% CI  |

---

## Recommended Implementation Order

### Phase A: Quick Wins (Low Risk, Fast Validation)

1. **H1: Symmetric PLE Gates** - Simple code change, validates gate access
   hypothesis
2. **H9: DCN Low-Rank Reduction** - QPS win with minimal NE risk
3. **H6: Learnable Loss Scale** - Enable for top tasks, monitor stability

### Phase B: NE Improvements (Primary Goal)

4. **H2: Progressive PLE** - Strong theoretical backing from original paper
5. **H3: Expert Count Increase** - Proven MoE approach
6. **H4: MHTA Stack Increase** - Leverage existing infrastructure

### Phase C: Architectural Changes (Higher Effort)

7. **H7: Wukong SFMs** - Requires new module implementation
8. **H8: HSTU Seq Extension** - Memory/training considerations
9. **H5: Dual-Flow HSTU** - Significant refactoring for QPS gains

---

## Implementation Notes

The hypothesis report will be saved to:
`/Users/zgchen/fbsource/fbcode/minimal_viable_ai/models/main_feed_mtml/coscience/hypotheses_and_plans/hypothesis_report.md`

Each hypothesis includes sufficient detail for direct implementation in
`model_roo_v0.py` with specific:

- Line number references for current implementation
- Code snippets showing proposed changes
- Expected impact ranges based on literature and prior experiments

### Key Configuration Parameters

| Parameter             | Current Value  | Location                 |
| --------------------- | -------------- | ------------------------ |
| `num_expert_in_group` | 1              | CGCModule initialization |
| `mhta_num_stack`      | 1              | MHTA configuration       |
| `mhta_num_heads`      | 8              | MHTA configuration       |
| `low_rank_dim`        | 768            | DeepCrossNet heads       |
| `max_seq_len`         | 460 (512-50-2) | HSTU DataFM              |
| `expert_layer_size`   | 256            | PLE experts              |
| `ple_input_dim`       | 1200           | PLE input                |

### Testing Recommendations

1. **A/B Testing Protocol:**
   - Run each hypothesis as separate experiment
   - Minimum 7-day test period for statistical significance
   - Monitor both NE and QPS metrics

2. **Ablation Strategy:**
   - For composite hypotheses (e.g., H2 + H3), test individually first
   - Validate that gains are additive before combining

3. **Rollback Criteria:**
   - NE regression > 0.05%: Immediate rollback
   - QPS degradation > 15%: Review and optimize
   - Training instability (loss spikes): Reduce learning rate or revert

### Risk Mitigation

| Hypothesis          | Primary Risk              | Mitigation                                      |
| ------------------- | ------------------------- | ----------------------------------------------- |
| H2: Progressive PLE | Training instability      | Gradient clipping, careful initialization       |
| H3: Expert Count    | Expert collapse           | Monitor expert utilization, add load balancing  |
| H5: Dual-Flow HSTU  | NE regression             | Extensive offline validation before online test |
| H7: Wukong SFMs     | Implementation complexity | Start with single head replacement, validate    |
| H8: Seq Extension   | Memory overflow           | Gradient checkpointing, batch size reduction    |

### Success Metrics

**Primary (NE Improvement):**

- Target: +0.05% NE improvement
- Stretch Goal: +0.10% NE improvement
- Minimum Acceptable: +0.02% NE improvement

**Secondary (QPS Improvement):**

- Target: +15% QPS improvement with NE-neutral
- Stretch Goal: +25% QPS improvement
- Maximum Acceptable NE Regression: -0.01%

---

## Conclusion

This report identifies 13 actionable hypotheses for improving the CFR model's
performance (NE) and efficiency (QPS). The hypotheses span multiple categories:

- **4 Module Changes** (H2, H3, H4, H7, H8, H12): Architectural modifications
  for capacity increase
- **4 Module Refactors** (H1, H5, H10, H13): Structural optimizations without
  major redesign
- **4 Ad-hoc Changes** (H6, H9, H11): Quick wins through
  hyperparameter/configuration tuning

**Key Takeaways:**

1. **Wukong SFMs (H7)** and **Progressive PLE (H2)** offer the highest potential
   NE gains but require more implementation effort.

2. **DCN Low-Rank Reduction (H9)** and **Expert Bottleneck (H10)** provide the
   most direct path to QPS improvements with minimal NE risk.

3. **Symmetric PLE Gates (H1)** is the lowest-hanging fruit for immediate
   validation of the asymmetric design hypothesis.

4. The current architecture has several hard-coded parameters (loss weights,
   gate configurations) that may benefit from learnable alternatives.

5. Aligning with prior knowledge, deeper interactions (H2, H4), longer sequences
   (H8), and richer feature processing (H7) are likely to yield performance
   gains.

**Recommended Next Steps:**

1. Implement and validate H1 (Symmetric PLE Gates) as a quick diagnostic
2. Parallel implementation of H9 (DCN reduction) for QPS baseline
3. Prioritize H2 or H7 based on team capacity for major NE improvement
   initiative
4. Consider H5 (Dual-Flow HSTU) as a longer-term efficiency project

---

_Report generated: December 11, 2025_ _Target model: model_roo_v0.py_ _Analysis
framework: Co-Scientist hypothesis generation protocol_
