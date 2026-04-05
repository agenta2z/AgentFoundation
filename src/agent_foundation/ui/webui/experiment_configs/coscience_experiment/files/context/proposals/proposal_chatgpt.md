# Model ROO V0 Improvement Hypotheses Report

**Date**: December 11, 2025 **Target Model**: `model_roo_v0.py` **Goals**:

- **Primary**: Improve NE (Normalized Entropy) with reasonable model size
  increase
- **Secondary**: Reduce model size to improve QPS while maintaining model
  performance (marginal NE regression acceptable)

---

## Executive Summary

After comprehensive analysis of the `model_roo_v0.py` architecture, external
research literature, and code documentation, this report proposes **10
prioritized hypotheses** for model improvement. The hypotheses span multiple
categories including PLE/MoE modifications, feature interaction enhancements,
architecture refinements, and efficiency optimizations.

**Key Insights from Analysis**:

1. The current PLE (CGCModule) implementation uses an **asymmetric gating
   design** where only task groups 2 and 3 access shared experts—groups 0 and 1
   have no effective gating.
2. The model has **single-layer PLE** without progressive layering.
3. Research shows **multi-embedding architectures** can address embedding
   collapse and significantly improve capacity.
4. The **MHTA (Multi-Head Talking Attention)** module has 8 heads with DCN, but
   uses split input patterns that could be optimized.
5. **Hardcoded task weights** in loss function suggest prior tuning but may not
   be optimal.

---

## Phase 1: Context Review & Opportunity Identification

### 1.1 Target Model Architecture Summary

The `model_roo_v0.py` implements a complex multi-task ranking model with:

| Component                | Configuration                                    | Notes                                          |
| ------------------------ | ------------------------------------------------ | ---------------------------------------------- |
| **ROO Architecture**     | Dual-path (RO/NRO)                               | Request-Only vs Non-Request-Only feature split |
| **MHTA**                 | 8 heads, 1024 dim each                           | Multi-Head Talking Attention with DeepCrossNet |
| **PLE/CGC**              | 4 groups, 1 expert/group, 1 shared expert        | Asymmetric design - groups 0,1 have no gating  |
| **HSTU**                 | 512 seq length, 192 dim, 2 layers                | Sequential user behavior modeling              |
| **Content Intelligence** | SlimDHEN with 128 output dim                     | Separate CI feature processing                 |
| **Tasks**                | 65+ tasks (23 prediction, 13 auxiliary, 29 bias) | BCE, MSE, KDLM, Pairwise loss                  |

### 1.2 External Research Key Findings

From the external research survey (ICML 2024, KDD 2024, RecSys 2024):

1. **Multi-Embedding Architecture** (Guo et al., ICML 2024):
   - Addresses "embedding collapse" where embeddings become low-rank
   - Multiple embedding tables per feature increase diversity
   - Demonstrated consistent gains across FM, MLPs, DCNv2

2. **Heterogeneous MoE with Multi-Embedding** (Tencent KDD 2024):
   - +3.9% GMV lift in production
   - Each expert only interacts with features from one embedding table
   - Temporal Interest Module for time-aware sequences

3. **Progressive Layered Extraction (PLE)** (RecSys 2020):
   - Original design is symmetric (all gates access task + shared experts)
   - Progressive layering (multiple stacked CGC layers) improves task
     specialization
   - Current model uses single layer, asymmetric design

4. **Auxiliary-Loss-Free Load Balancing** (DeepSeek-V3):
   - Bias-based routing avoids gradient conflict
   - Shared expert isolation for common patterns

5. **CALRec Contrastive Alignment** (RecSys 2024):
   - +37% Recall@1 through two-stage training
   - Contrastive objectives improve representation learning

### 1.3 Code Analysis Findings

**Identified Opportunities**:

1. **Asymmetric PLE Design** (Lines 505-519):

   ```python
   out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)
   ```

   Groups 0,1 have single-output gates with no routing—effectively bypassing MoE
   benefits.

2. **Single PLE Layer**: No progressive layering despite PLE literature showing
   benefits.

3. **Fixed Expert Count**: Only 1 expert per task group, limiting capacity.

4. **Hardcoded Loss Weights** (Lines 955-970):

   ```python
   if task_name in {"linear_vpvd", "kd_linear_vpvd", "kd_bce_linear_vpvd"}:
       task_weight = 10.3
   elif task_name in {"share"}:
       task_weight = 1.5
   ```

   These fixed values suggest prior optimization but may not be globally
   optimal.

5. **MHTA Split Pattern** (Lines 278-288):

   ```python
   split_sizes=[1024 * 3, 1024 * 3, 1024, 1024, (num_embs * k), deep_fm_output_dim // 2, deep_fm_output_dim // 2]
   input_indices=[[0, 2, 4, 5], [1, 3, 4, 6]]
   ```

   Fixed split pattern may not be optimal for all tasks.

6. **Content Intelligence Module Potential**: CI architecture uses SlimDHEN but
   could benefit from deeper integration with main pathway.

---

## Phase 2: Hypothesis Formulation

### Hypothesis 1: Symmetric PLE Gating Design

**Category**: Model Module Refactor (PLE)

**Hypothesis**: Convert the asymmetric PLE gating (where groups 0,1 have no
shared expert access) to symmetric design (all groups access their
task-specific + shared experts).

**Current State**:

- Groups 0,1: Gate output dim = 1, no shared expert access (trivial softmax)
- Groups 2,3: Gate output dim = 2, access both task + shared experts

**Proposed Change**:

```python
# Change from:
out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)
# To:
out_features=num_expert_in_group + len(shared_experts)  # For ALL groups
```

**Supporting Evidence**:

- Original PLE paper (RecSys 2020) shows symmetric design improves cross-task
  learning
- Groups 0,1 currently miss MoE capacity benefits entirely
- Documentation notes this as "undocumented design choice" suggesting it may be
  experimental

**Contradicting Evidence**:

- Current asymmetric design may have been intentionally tuned for specific task
  groups
- Could increase compute for groups 0,1 without proportional benefit

**Predicted Success**: **Medium (50%)** **Potential Impact**: **Medium** (NE
improvement 0.05-0.2%)

---

### Hypothesis 2: Multi-Layer Progressive PLE

**Category**: Model Module Change (PLE)

**Hypothesis**: Stack 2-3 CGC layers progressively, where each layer receives
outputs from previous layer, enabling gradual task specialization.

**Current State**: Single CGC layer **Proposed Change**:

```python
class ProgressivePLE(nn.Module):
    def __init__(self, num_layers=2, ...):
        self.layers = nn.ModuleList([
            CGCModule(...) for _ in range(num_layers)
        ])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
```

**Supporting Evidence**:

- PLE paper explicitly designs for progressive layering showing improved
  task-specific representations
- LLM MoE research shows stacking experts deepens specialization
- Documentation identifies this as "potential optimization opportunity"

**Contradicting Evidence**:

- Adds significant model parameters (~2-3M per layer)
- Increases training time and latency
- May cause gradient vanishing in deeper experts

**Predicted Success**: **Medium (45%)** **Potential Impact**: **High** (NE
improvement 0.1-0.3%, with model size trade-off)

---

### Hypothesis 3: Increase Experts per Task Group

**Category**: Model Module Change (PLE)

**Hypothesis**: Increase from 1 expert per task group to 2-3 experts, enabling
finer-grained specialization within each task domain.

**Current State**: `num_expert_in_group = 1` **Proposed Change**:
`num_expert_in_group = 2` or `3`

**Supporting Evidence**:

- DeepSeek-V3 uses 256 routed experts showing fine-grained experts improve
  specialization
- With only 1 expert, gating becomes trivial (100% weight)
- More experts allow learning different "views" of task-specific patterns

**Contradicting Evidence**:

- Linear parameter increase
- May cause expert collapse if not balanced
- Small dataset per expert could hurt convergence

**Predicted Success**: **Medium (55%)** **Potential Impact**: **Medium** (NE
improvement 0.05-0.15%)

---

### Hypothesis 4: Multi-Embedding Architecture for Heterogeneous Feature Interaction

**Category**: Feature Engineering / Architecture Change

**Hypothesis**: Implement multi-embedding tables per feature (similar to Tencent
KDD 2024), where each embedding set captures different interaction patterns and
feeds into separate expert pathways.

**Current State**: Single embedding table per feature **Proposed Change**:

```python
# Create M embedding sets per feature
embedding_sets = [
    EmbeddingTable(feature, dim=embedding_dim)
    for _ in range(M)  # M=2 or 3
]
# Each set feeds into different expert/interaction module
```

**Supporting Evidence**:

- ICML 2024 paper shows multi-embedding alleviates embedding collapse
- Tencent KDD 2024 reports +3.9% GMV with this approach
- User's prior knowledge states "heterogeneous feature interactions are
  critical"

**Contradicting Evidence**:

- Significant memory increase (M× embedding tables)
- Requires careful design of which embeddings feed which experts
- May not generalize well if features are already diverse

**Predicted Success**: **Medium (40%)** **Potential Impact**: **High** (NE
improvement 0.2-0.5%, major parameter increase)

---

### Hypothesis 5: Learnable Loss Weights via Uncertainty Modeling

**Category**: Model Ad-hoc Change (Loss Function)

**Hypothesis**: Replace hardcoded task weights with learnable uncertainty-based
weights that adapt during training.

**Current State**:

```python
if task_name in {"linear_vpvd", "kd_linear_vpvd"}:
    task_weight = 10.3
elif task_name in {"comment_pairwise"}:
    task_weight = 10.0
# ... more hardcoded weights
```

**Proposed Change**: Implement homoscedastic uncertainty weighting:

```python
class LearnableTaskWeights(nn.Module):
    def __init__(self, num_tasks):
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        weighted_loss = precision * losses + self.log_vars
        return weighted_loss.sum()
```

**Supporting Evidence**:

- Multi-task learning literature shows uncertainty weighting improves
  convergence
- The model already has `LearnableLossScale` for some tasks—inconsistent
  application
- Hardcoded weights suggest manual tuning; learnable weights automate this

**Contradicting Evidence**:

- Hardcoded weights may represent optimal values from extensive tuning
- Learnable weights could destabilize training initially
- Some tasks may have external constraints on relative importance

**Predicted Success**: **Medium (50%)** **Potential Impact**: **Medium** (NE
improvement 0.05-0.15%)

---

### Hypothesis 6: MHTA Head Pruning for QPS Improvement

**Category**: Model Module Refactor (MHTA) — **QPS Focus**

**Hypothesis**: Reduce MHTA heads from 8 to 4-6 based on attention importance
analysis, trading marginal NE for significant latency reduction.

**Current State**: 8 heads × 1024 dim = 8192 total dim **Proposed Change**:
Analyze head importance via gradient-based pruning, reduce to 4-6 heads

**Supporting Evidence**:

- Transformer pruning research shows many attention heads are redundant
- 8 parallel DCN heads is computationally expensive
- QPS secondary goal explicitly allows marginal NE regression

**Contradicting Evidence**:

- Each head may capture unique interaction patterns
- ROO architecture relies on diverse head outputs for MHTA
- Pruning wrong heads could cause significant NE regression

**Predicted Success**: **High (65%)** **Potential Impact**: **High for QPS**
(10-15% latency reduction, 0-0.1% NE regression)

---

### Hypothesis 7: Expert Bottleneck Architecture

**Category**: Model Module Refactor (PLE)

**Hypothesis**: Add bottleneck structure to experts (down-project → process →
up-project) to reduce parameters while maintaining capacity.

**Current State**:

```python
nn.Sequential(
    nn.Linear(ple_input_dim, 256),  # Direct projection
    SwishLayerNorm([256]),
    nn.Linear(256, 256),
    SwishLayerNorm([256]),
)
```

**Proposed Change**:

```python
nn.Sequential(
    nn.Linear(ple_input_dim, 64),   # Bottleneck down
    SwishLayerNorm([64]),
    nn.Linear(64, 128),              # Process
    SwishLayerNorm([128]),
    nn.Linear(128, 256),             # Up-project
    SwishLayerNorm([256]),
)
```

**Supporting Evidence**:

- Bottleneck architectures (ResNet, MobileNet) proven efficient
- Reduces parameters while preserving representational capacity
- Can maintain NE while reducing model size

**Contradicting Evidence**:

- Information bottleneck may lose critical task-specific signals
- Requires careful tuning of bottleneck dimension
- May increase depth-related training challenges

**Predicted Success**: **Medium (55%)** **Potential Impact**: **Medium for QPS**
(5-10% parameter reduction, neutral NE)

---

### Hypothesis 8: Content Intelligence Deep Integration

**Category**: Feature Transformation Refactor

**Hypothesis**: Integrate Content Intelligence (CI) module output earlier in the
pipeline (before MHTA instead of after) and add cross-attention between CI and
main pathway.

**Current State**: CI output concatenated after MHTA:

```python
# CI processed separately, concatenated with MHTA output
combined = torch.cat([mhta_out, ci_out], dim=-1)
```

**Proposed Change**:

```python
# CI as additional input to MHTA
ci_aware_interaction = CrossAttention(main_features, ci_features)
mhta_input = torch.cat([interaction_out, ci_aware_interaction], dim=-1)
```

**Supporting Evidence**:

- User's prior knowledge emphasizes heterogeneous feature interactions
- CI features (IFM, KGID) contain rich semantic information
- Earlier integration allows MHTA to learn CI-aware patterns

**Contradicting Evidence**:

- May increase complexity without proportional benefit
- CI module already contributes to final representation
- Cross-attention adds significant compute

**Predicted Success**: **Medium (45%)** **Potential Impact**: **Medium** (NE
improvement 0.05-0.15%)

---

### Hypothesis 9: SEDOT Module Efficiency Optimization

**Category**: Model Module Refactor — **QPS Focus**

**Hypothesis**: Optimize the SEDOT (FeedTransformerSpecialInteraction) module by
reducing `n_repeat` from 4 to 2 and `num_output_tokens` from 32 to 16.

**Current State**:

```python
FeedTransformerSpecialInteraction(
    num_input_tokens=144,
    num_output_tokens=32,
    n_repeat=4,
    num_seed_tokens=32,
    # ...
)
```

**Proposed Change**:

```python
FeedTransformerSpecialInteraction(
    num_input_tokens=144,
    num_output_tokens=16,  # Reduced
    n_repeat=2,            # Reduced
    num_seed_tokens=16,    # Reduced
    # ...
)
```

**Supporting Evidence**:

- SEDOT is computationally intensive with 4 transformer repeats
- Output tokens may have redundancy
- QPS goal allows marginal NE trade-off

**Contradicting Evidence**:

- SEDOT captures important special interactions
- Reducing capacity may lose critical patterns
- Output dimension changes cascade through model

**Predicted Success**: **High (60%)** **Potential Impact**: **Medium for QPS**
(5-8% latency reduction, 0-0.05% NE regression)

---

### Hypothesis 10: Auxiliary Loss Scaling Optimization

**Category**: Model Ad-hoc Change (Loss)

**Hypothesis**: Optimize the auxiliary loss scaling factors
(`interarch_aux_loss_outputs / 100`, `densearch_aux_loss_outputs / 300`) through
grid search or gradient-based tuning.

**Current State**:

```python
if key == "interarch_aux_loss_outputs":
    los /= 100
if key == "densearch_aux_loss_outputs":
    los /= 300
```

**Proposed Change**: Make these learnable or conduct systematic ablation:

```python
self.aux_loss_scales = nn.ParameterDict({
    "interarch_aux_loss": nn.Parameter(torch.tensor(0.01)),
    "densearch_aux_loss": nn.Parameter(torch.tensor(0.0033)),
})
```

**Supporting Evidence**:

- Fixed values (100, 300) suggest arbitrary tuning
- Auxiliary losses help but sub-optimal scaling can hurt main task
- Similar to Hypothesis 5, learnable parameters automate tuning

**Contradicting Evidence**:

- Current values may be carefully tuned for stability
- Learnable auxiliary scales could dominate or be suppressed
- Risk of training instability

**Predicted Success**: **High (60%)** **Potential Impact**: **Low-Medium** (NE
improvement 0.02-0.08%)

---

## Phase 3: Hypothesis Ranking & Summary

### Final Ranked Hypothesis List

| Rank   | Hypothesis                     | Category                   | Success Prob. | Impact     | Priority Score |
| ------ | ------------------------------ | -------------------------- | ------------- | ---------- | -------------- |
| **1**  | H6: MHTA Head Pruning          | Model Module Refactor      | High (65%)    | High QPS   | ⭐⭐⭐⭐⭐     |
| **2**  | H3: Increase Experts per Group | Model Module Change        | Medium (55%)  | Medium NE  | ⭐⭐⭐⭐       |
| **3**  | H1: Symmetric PLE Gating       | Model Module Refactor      | Medium (50%)  | Medium NE  | ⭐⭐⭐⭐       |
| **4**  | H9: SEDOT Efficiency           | Model Module Refactor      | High (60%)    | Medium QPS | ⭐⭐⭐⭐       |
| **5**  | H10: Auxiliary Loss Scaling    | Model Ad-hoc Change        | High (60%)    | Low-Med NE | ⭐⭐⭐         |
| **6**  | H7: Expert Bottleneck          | Model Module Refactor      | Medium (55%)  | Medium QPS | ⭐⭐⭐         |
| **7**  | H5: Learnable Loss Weights     | Model Ad-hoc Change        | Medium (50%)  | Medium NE  | ⭐⭐⭐         |
| **8**  | H2: Multi-Layer PLE            | Model Module Change        | Medium (45%)  | High NE    | ⭐⭐⭐         |
| **9**  | H8: CI Deep Integration        | Feature Transform Refactor | Medium (45%)  | Medium NE  | ⭐⭐           |
| **10** | H4: Multi-Embedding            | Architecture Change        | Medium (40%)  | High NE    | ⭐⭐           |

### Recommended Implementation Order

**Phase A: Low-Risk, High-Confidence (QPS Focus)**

1. **H6**: MHTA Head Pruning — Analyze head importance, prune incrementally
2. **H9**: SEDOT Efficiency — Reduce n_repeat and output tokens
3. **H7**: Expert Bottleneck — Add bottleneck to existing experts

**Phase B: Medium-Risk, NE Improvement** 4. **H1**: Symmetric PLE — Enable
shared expert access for groups 0,1 5. **H3**: Increase Experts — Add 1 more
expert per task group 6. **H10**: Auxiliary Loss Scaling — Conduct grid search
on scaling factors

**Phase C: Higher-Risk, Potential High Reward** 7. **H5**: Learnable Loss
Weights — Implement uncertainty weighting 8. **H2**: Multi-Layer PLE — Add
second CGC layer with careful monitoring 9. **H8**: CI Deep Integration — Add
cross-attention module 10. **H4**: Multi-Embedding — Requires significant
engineering, highest risk/reward

---

## Implementation Considerations

### For NE Improvement Hypotheses:

- Start with ablation studies to verify baseline contributions
- Monitor validation NE closely for early stopping
- Consider combining H1 + H3 for synergistic effects

### For QPS Improvement Hypotheses:

- Measure latency on representative hardware before/after
- Verify NE regression is within acceptable bounds (<0.1%)
- Consider H6 + H9 together for compounding latency reduction

### General Notes:

- All hypotheses should be tested with A/B experiments before production
- Parameter count changes should be documented for serving cost estimation
- Consider gradual rollout with traffic splitting

---

## References

1. Guo et al., "On the Embedding Collapse When Scaling Up Recommendation Models"
   (ICML 2024)
2. Pan et al., "Ads Recommendation in a Collapsed and Entangled World"
   (KDD 2024)
3. Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
   Learning Model" (RecSys 2020)
4. DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
5. Li et al., "CALRec: Contrastive Alignment of Generative LLMs for Sequential
   Recommendation" (RecSys 2024)
6. CFR Architecture Documentation (`coscience/context/code_docs/models/cfr/`)
7. CGCModule Documentation
   (`coscience/context/code_docs/models/moe/cgcmodule_roo.rst`)
