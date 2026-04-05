# Model ROO v0 Improvement Hypotheses Report

## Executive Summary

This report presents a comprehensive analysis of the CFR `model_roo_v0.py` architecture and proposes **10 ranked hypotheses** for improving NE (Normalized Entropy) and/or QPS. The hypotheses range from incremental architectural refinements to significant module refactoring, all grounded in:
1. External research findings (HSTU, GradCraft, MultiBalance, DCNv3, DeepSeek MoE patterns)
2. Code pattern analysis from the target model
3. Prior knowledge about heterogeneous feature interactions being critical to model performance

---

## Phase 1: Context Analysis Summary

### Current Model Architecture Overview

The `model_roo_v0.py` implements a **5.1B parameter** multi-task ranking model with:

- **RO/NRO Feature Split**: Request-Only (user context) vs Non-Request-Only (item features)
- **HSTU Integration**: Transformer-based sequential user modeling (512 seq len, 192 dim)
- **Multi-Head Talking Attention (MHTA)**: 8 heads × 1024 dim with DCN-based head modules
- **PLE/CGC Module**: 4 task groups with **asymmetric gating** (only groups 2,3 access shared experts)
- **Content Intelligence Module**: SlimDHEN for content-specific embeddings
- **65+ Tasks**: BCE, MSE, KDLM, and pairwise ranking losses

### Key Identified Patterns & Potential Opportunities

| Pattern | Location | Observation |
|---------|----------|-------------|
| **Asymmetric PLE Gates** | Lines 505-519 | Gates 0,1 have trivial softmax (only 1 expert) - no real routing |
| **Hard-coded Task Weights** | Lines 955-970 | Fixed weights like `linear_vpvd=10.3`, `share=1.5`, `comment_pairwise=10.0` |
| **Single-layer PLE** | Lines 520-524 | Only 1 CGC layer; progressive layering not used |
| **Fixed MHTA Split Sizes** | Lines 278-287 | Static `split_sizes` and `input_indices` configurations |
| **HSTU Sequence Length** | hstu_datafm.rst | Max 460 tokens; research shows 10K+ is beneficial |
| **Aux Loss Division** | Lines 729-732 | Hardcoded `/100` and `/300` for aux losses |

---

## Phase 2: Hypothesis Formulation

### Hypothesis 1: Symmetric PLE Gating
**Category**: Model Module Refactor

**Description**: Convert the current asymmetric CGC/PLE gating to symmetric design where ALL task groups (0-3) can access both task-specific AND shared experts.

**Current State**:
- Gates 0,1 output dimension = 1 (can only access their own task expert)
- Gates 2,3 output dimension = 2 (can access task expert + shared expert)
- This means task groups 0,1 have **NO actual gating/routing mechanism**

**Proposed Change**:
```python
# Current (asymmetric):
out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)

# Proposed (symmetric):
out_features=num_expert_in_group + num_shared_experts  # For ALL groups
```

**Supporting Evidence**:
1. Original PLE paper uses symmetric design with all gates accessing shared experts
2. Code documentation explicitly notes: "Gates 0 and 1 have effectively NO gating!"
3. DeepSeek-V3 research shows shared experts capture common patterns while specialized experts hyper-specialize

**Contradicting Evidence**:
1. The asymmetric design may have been intentional optimization from prior experiments
2. Adding more routing may increase inference latency slightly

**Success Probability**: **High (75%)**
**Potential Impact**: **Medium** - Expected NE improvement of 0.1-0.3% based on PLE paper

---

### Hypothesis 2: Multi-Layer Progressive PLE
**Category**: Model Module Addition

**Description**: Stack 2-3 CGC layers to implement true "Progressive Layered Extraction" instead of single-layer.

**Current State**: Only 1 CGC layer is used

**Proposed Change**:
```python
# Add progressive layers
ple_layers = nn.ModuleList([
    CGCModule(task_specific_experts_layer1, gates_layer1, shared_expert_modules_layer1),
    CGCModule(task_specific_experts_layer2, gates_layer2, shared_expert_modules_layer2),
])
```

**Supporting Evidence**:
1. PLE paper shows progressive layering enables gradual task specialization
2. Each layer refines representations from previous layer
3. Research shows deeper models with proper architecture capture more complex interactions

**Contradicting Evidence**:
1. Increases model size and latency
2. May require careful initialization to avoid training instability
3. Diminishing returns after 2-3 layers per original paper

**Success Probability**: **Medium (55%)**
**Potential Impact**: **Medium-High** - Expected NE improvement of 0.2-0.5% with ~15% model size increase

---

### Hypothesis 3: Learnable Task Weights via MultiBalance
**Category**: Model Ad-hoc Change

**Description**: Replace hard-coded task weights with learnable gradient-balanced weights following Meta's MultiBalance approach.

**Current State** (Lines 955-970):
```python
if task_name in {"linear_vpvd", "kd_linear_vpvd", "kd_bce_linear_vpvd"}:
    task_weight = 10.3  # Hard-coded
elif task_name in {"share"}:
    task_weight = 1.5   # Hard-coded
```

**Proposed Change**: Implement representation-level gradient balancing with moving-average gradient computation.

**Supporting Evidence**:
1. **MultiBalance paper (Meta, 2024)**: "0.738% Normalized Entropy improvement with neutral training cost"
2. Balances shared representation gradients rather than parameter gradients
3. **Zero QPS degradation** - no inference cost increase

**Contradicting Evidence**:
1. Training complexity increases
2. Requires careful tuning of moving-average parameters
3. Current hard-coded weights may be optimized from extensive experimentation

**Success Probability**: **High (70%)**
**Potential Impact**: **High** - Expected NE improvement of 0.3-0.7% with no QPS impact

---

### Hypothesis 4: Increase MHTA Heads with Smaller Dimensions
**Category**: Model Module Refactor

**Description**: Change MHTA from 8×1024 to 16×512 (same total dim) for more diverse feature interactions.

**Current State**:
```python
mhta_head_dim = 1024
mhta_num_heads = 8
# Total: 8192 dimensions
```

**Proposed Change**:
```python
mhta_head_dim = 512
mhta_num_heads = 16
# Total: 8192 dimensions (same)
```

**Supporting Evidence**:
1. More heads capture more diverse interaction patterns
2. Research on attention mechanisms shows more heads with smaller dimensions often outperform fewer large heads
3. Prior knowledge: "heterogeneous feature interactions are critical" - more heads = more interaction types

**Contradicting Evidence**:
1. Smaller head dimension may reduce per-head expressiveness
2. The 8-head configuration may be optimized for the DCN-based head modules
3. May require adjusting split_sizes configuration

**Success Probability**: **Medium (50%)**
**Potential Impact**: **Low-Medium** - Expected NE improvement of 0.05-0.2% with neutral model size

---

### Hypothesis 5: DCNv3-Style Explicit Supervision for Head Modules
**Category**: Model Module Addition

**Description**: Add Tri-BCE loss (DCNv3) to provide direct supervision to different MHTA head sub-networks.

**Current State**: MHTA heads only receive gradient through final task losses

**Proposed Change**: Add intermediate auxiliary losses to each DCN head module:
```python
# Add per-head supervision losses
for i, head_output in enumerate(head_outputs):
    head_aux_loss = tri_bce_loss(head_output, labels)
    total_loss += head_aux_loss * head_weight
```

**Supporting Evidence**:
1. **DCNv3 paper**: "state-of-the-art using only explicit feature interactions"
2. Improves interpretability by encouraging head specialization
3. Can eliminate need for implicit DNN components

**Contradicting Evidence**:
1. Increases training complexity
2. May cause gradient conflict with main task losses
3. Requires careful loss weighting to avoid dominating main losses

**Success Probability**: **Medium (45%)**
**Potential Impact**: **Medium** - Expected NE improvement of 0.1-0.3%

---

### Hypothesis 6: Reduce Model Size via Expert Pruning
**Category**: Model Module Change (QPS Focus)

**Description**: Reduce PLE experts from 4 task-specific + 1 shared to 2 task-specific + 1 shared, consolidating similar task groups.

**Current State**:
- 4 task groups × 1 expert = 4 task-specific experts
- 1 shared expert
- Total: 5 experts

**Proposed Change**:
- Merge similar task groups (e.g., engagement vs quality signals)
- 2 task groups × 1 expert = 2 task-specific experts + 1 shared
- Total: 3 experts (~40% reduction in PLE parameters)

**Supporting Evidence**:
1. Netflix research shows "multi-task consolidation benefits" - combining related tasks improves performance
2. Fewer experts = faster inference (QPS improvement)
3. Task groups 0,1 currently have no real gating anyway

**Contradicting Evidence**:
1. May lose task-specific representations
2. Could increase negative transfer between dissimilar tasks
3. Requires careful grouping based on task correlations

**Success Probability**: **Medium (50%)**
**Potential Impact**: **Medium for QPS** - Expected 10-20% QPS improvement with <0.1% NE regression

---

### Hypothesis 7: Extend HSTU Sequence Length with Chunked Attention
**Category**: Model Module Refactor

**Description**: Increase HSTU sequence length from 460 to 2000+ using chunked attention mechanisms.

**Current State**:
```python
self.max_seq_len = 512 - 50 - 2  # = 460 tokens
```

**Proposed Change**: Implement STCA-style spatial-temporal chunk attention:
```python
self.max_seq_len = 2048
self.chunk_size = 256
self.use_chunked_attention = True
```

**Supporting Evidence**:
1. **ByteDance STCA**: 10,000-length sequences on Douyin at production latency
2. **TWIN V2**: 100,000+ historical behaviors at 400M+ DAU scale
3. Prior knowledge: "longer sequence, richer features are key to performance gain"
4. User behavior has temporal patterns - more history = better predictions

**Contradicting Evidence**:
1. Significant engineering complexity
2. Increases memory usage during training
3. May require distributed attention implementation
4. Current 460 length may already capture most relevant behavior

**Success Probability**: **Medium (50%)**
**Potential Impact**: **High** - Expected NE improvement of 0.3-0.8% (based on industry reports)

---

### Hypothesis 8: Add Auxiliary Loss-Free Load Balancing (DeepSeek-style)
**Category**: Model Ad-hoc Change

**Description**: For the PLE gating, implement bias-based routing instead of current approach to prevent expert collapse without auxiliary loss conflicts.

**Current State**: PLE uses standard softmax gating

**Proposed Change**:
```python
# Add dynamic bias to gating
s_i = affinity(input, expert_i) + bias_i
selected = softmax(s)

# Bias update (not gradient-based):
if expert_i is overloaded:
    bias_i -= gamma
elif expert_i is underloaded:
    bias_i += gamma
```

**Supporting Evidence**:
1. **DeepSeek-V3**: "auxiliary-loss-free load balancing" improves training stability
2. Avoids gradient conflict between quality loss and balance loss
3. Acts like PID controller for expert utilization

**Contradicting Evidence**:
1. PLE experts are task-exclusive, so load balancing is less critical
2. Adds training complexity
3. Current model may not suffer from expert collapse

**Success Probability**: **Low (35%)**
**Potential Impact**: **Low** - Expected improvement marginal, mainly training stability

---

### Hypothesis 9: Replace DeepFM with Gated DCN (GDCN)
**Category**: Model Module Replacement

**Description**: Replace the DeepFM combiner module with GDCN (Gated Deep Cross Network) for better feature interaction.

**Current State** (Lines 395-399):
```python
cat_embedding_module=DeepFM(
    deep_fm_arch=model_config.deep_fm_arch,
    input_dim=model_config.embedding_dim * k,
    output_dim=deep_fm_output_dim,  # 256
)
```

**Proposed Change**:
```python
cat_embedding_module=GatedDCN(
    input_dim=model_config.embedding_dim * k,
    output_dim=deep_fm_output_dim,
    num_layers=4,
    low_rank_dim=256,
    use_information_gates=True,  # Key GDCN feature
)
```

**Supporting Evidence**:
1. **GDCN research**: Information gates dynamically filter feature interactions
2. Delays performance plateau to depth 8-10 vs DCN-V2's early saturation
3. Field-level dimension optimization can reduce parameters ~75% with comparable accuracy

**Contradicting Evidence**:
1. DeepFM may be sufficient for current use case
2. Additional gating adds inference latency
3. Requires retuning of downstream components

**Success Probability**: **Medium (55%)**
**Potential Impact**: **Medium** - Expected NE improvement of 0.1-0.3%

---

### Hypothesis 10: Learnable Aux Loss Scaling
**Category**: Model Ad-hoc Change

**Description**: Replace hard-coded auxiliary loss division with learnable scaling parameters.

**Current State** (Lines 729-732):
```python
if key == "interarch_aux_loss_outputs":
    los /= 100  # Hard-coded
if key == "densearch_aux_loss_outputs":
    los /= 300  # Hard-coded
```

**Proposed Change**:
```python
self.aux_loss_scales = nn.ParameterDict({
    "interarch_aux_loss": nn.Parameter(torch.tensor(0.01)),
    "densearch_aux_loss": nn.Parameter(torch.tensor(0.003)),
})
# With optional regularization to prevent collapse
```

**Supporting Evidence**:
1. Model already has `LearnableLossScale` module (Line 930) for some tasks
2. Hard-coded values unlikely to be optimal across training dynamics
3. Learnable scales can adapt to dataset/distribution changes

**Contradicting Evidence**:
1. May cause training instability if not properly regularized
2. Current values may be extensively tuned
3. Auxiliary losses meant to be small - learnable may over-amplify

**Success Probability**: **Medium (50%)**
**Potential Impact**: **Low-Medium** - Expected NE improvement of 0.05-0.15%

---

## Phase 3: Ranked Hypothesis List

| Rank | Hypothesis | Category | Success Prob | NE Impact | QPS Impact | Priority Score |
|------|-----------|----------|--------------|-----------|------------|----------------|
| **1** | **Learnable Task Weights via MultiBalance** | Model Ad-hoc Change | High (70%) | High (+0.3-0.7%) | Neutral | ⭐⭐⭐⭐⭐ |
| **2** | **Symmetric PLE Gating** | Model Module Refactor | High (75%) | Medium (+0.1-0.3%) | Neutral | ⭐⭐⭐⭐⭐ |
| **3** | **Extend HSTU Sequence Length** | Model Module Refactor | Medium (50%) | High (+0.3-0.8%) | -5% | ⭐⭐⭐⭐ |
| **4** | **Multi-Layer Progressive PLE** | Model Module Addition | Medium (55%) | Medium-High (+0.2-0.5%) | -10% | ⭐⭐⭐⭐ |
| **5** | **Replace DeepFM with GDCN** | Model Module Replacement | Medium (55%) | Medium (+0.1-0.3%) | -5% | ⭐⭐⭐ |
| **6** | **Reduce Model Size via Expert Pruning** | Model Module Change | Medium (50%) | Low (<-0.1%) | +15-20% | ⭐⭐⭐ |
| **7** | **DCNv3-Style Head Supervision** | Model Module Addition | Medium (45%) | Medium (+0.1-0.3%) | Neutral | ⭐⭐⭐ |
| **8** | **Increase MHTA Heads (16×512)** | Model Module Refactor | Medium (50%) | Low-Medium (+0.05-0.2%) | Neutral | ⭐⭐ |
| **9** | **Learnable Aux Loss Scaling** | Model Ad-hoc Change | Medium (50%) | Low-Medium (+0.05-0.15%) | Neutral | ⭐⭐ |
| **10** | **Aux Loss-Free Load Balancing** | Model Ad-hoc Change | Low (35%) | Low | Neutral | ⭐ |

---

## Implementation Recommendations

### Tier 1 (High Priority - Implement First)
1. **Hypothesis 1 (Symmetric PLE)** + **Hypothesis 3 (MultiBalance)** can be combined
   - Both target multi-task optimization with complementary approaches
   - Low risk, high evidence support

### Tier 2 (Medium Priority - After Tier 1 Validation)
2. **Hypothesis 4 (Progressive PLE)** - if NE gains needed
3. **Hypothesis 6 (Expert Pruning)** - if QPS gains needed
4. **Hypothesis 7 (Extended HSTU)** - requires more engineering effort

### Tier 3 (Lower Priority - Experimental)
5. Remaining hypotheses for incremental gains

---

## References

1. Meta MultiBalance Paper (2024) - Multi-objective gradient balancing
2. PLE Paper (RecSys 2020) - Progressive Layered Extraction
3. DeepSeek-V3 Technical Report - Auxiliary-loss-free load balancing
4. DCNv3 Paper - Tri-BCE loss and explicit feature interactions
5. ByteDance STCA - Long sequence processing
6. GradCraft Paper (KDD 2024) - Global gradient direction alignment
