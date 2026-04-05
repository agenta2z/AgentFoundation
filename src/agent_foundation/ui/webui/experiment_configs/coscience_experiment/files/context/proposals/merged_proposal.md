# Consolidated Model ROO V0 Improvement Hypotheses Report

**Date**: December 15, 2025 **Target Model**: `model_roo_v0.py`
(MainFeedMTMLROO) **Goals**:

- **Primary**: Improve NE (Normalized Entropy) with reasonable model size
  increase
- **Secondary**: Reduce model size to improve QPS while maintaining model
  performance (marginal NE regression acceptable)

---

## Executive Summary

This consolidated report aggregates hypotheses from multiple independent
analyses and integrates **10 innovative new proposals** from recent idea
generation efforts. After removing familiar HSTU/Wukong proposals, merging
related ideas, and removing MultiBalance (found neutral), we present **31
distinct hypotheses** organized into **Tier 1 (High Priority)**, **Tier 2
(Medium Priority)**, and **Tier 3 (Exploratory)**.

**Key Consolidated Findings**:

1. **Zero-Cost Training Enhancement** (H1 Cross-Task Soft Labels) offers NE
   gains with **zero inference impact** - combines semantic relations + pairwise
   distillation
2. **Task-Adaptive Capacity** (H2 PRISM-T) is the **only proposal targeting
   TaskArchROOWrap** - backed by real P50 production results (-0.08% NE)
3. **HSTU as Control Signal** (H3 NEXUS, H5 AURORA, H6 TSWSP) - novel approaches
   transforming HSTU from passive feature to active controller
4. **HSTU Depth Scaling** (H7) - proven approach for NE improvement (2→6 layers)
5. **Cross-Stream Architecture** (H9 CASCADE) leverages existing RO/NRO
   infrastructure for selective capacity allocation

**Note**: MultiBalance/Learnable Loss Weights was **removed** (found neutral in
testing). SIM was **removed** (already proven/implemented).

**New Ideas Integration**:

- **Source Collection 1219251556003744**: CASCADE, SYNAPSE (NEXUS, QUANTUM),
  AURORA, PRISM-T, SPECTRUM
- **Source Collection 1394659704917730**: PEIN-DCN, TSWSP, TASLE

**Prior Knowledge Alignment**:

- Heterogeneous feature interactions → Focus on interaction layer improvements
  (MHTA, CASCADE, PEIN)
- Deeper model, longer sequence → HSTU depth scaling, Progressive PLE
- Efficient memory/embedding handling → Expert bottleneck, QUANTUM compression

---

## Final Ranked Hypothesis Summary Table

| Rank | ID  | Hypothesis                                          | Source ID                                   | Category   | Goal | Success Prob         | Impact         | Priority   |
| ---- | --- | --------------------------------------------------- | ------------------------------------------- | ---------- | ---- | -------------------- | -------------- | ---------- |
| 1    | H1  | Cross-Task Soft Label Enrichment (TASLE+CATALYST)   | 1394659704917730_P63 + 1219251556003744_P37 | Loss       | NE   | High (60%)           | +0.03-0.10% NE | ⭐⭐⭐⭐⭐ |
| 2    | H2  | PRISM-T Task Complexity-Aware Scaling               | 1219251556003744_P50                        | TaskArch   | NE   | Medium-High (55%)    | +0.05-0.15% NE | ⭐⭐⭐⭐⭐ |
| 3    | H3  | SYNAPSE-NEXUS HSTU-Guided Expert Routing            | 1219251556003744_P37                        | PLE        | NE   | Medium (50%)         | +0.05-0.15% NE | ⭐⭐⭐⭐⭐ |
| 4    | H4  | Symmetric PLE Gating Design                         | (original)                                  | PLE        | NE   | Medium-High (50-75%) | +0.05-0.3% NE  | ⭐⭐⭐⭐⭐ |
| 5    | H5  | AURORA Temporal Prediction Calibration              | 1219251556003744_P37_AURORA                 | Post-pred  | NE   | Medium (50%)         | +0.03-0.08% NE | ⭐⭐⭐⭐   |
| 6    | H6  | TSWSP Temporal Saliency Pooling                     | 1394659704917730_P19                        | HSTU PMA   | NE   | Medium (50%)         | +0.02-0.08% NE | ⭐⭐⭐⭐   |
| 7    | H7  | HSTU Depth Scaling (2→6 Layers)                     | (original)                                  | HSTU       | NE   | High (65-80%)        | +0.2-0.4% NE   | ⭐⭐⭐⭐   |
| 8    | H8  | Multi-Layer Progressive PLE                         | (original)                                  | PLE        | NE   | Medium (50-65%)      | +0.1-0.3% NE   | ⭐⭐⭐⭐   |
| 9    | H9  | CASCADE Asymmetric RO/NRO Depth                     | 1219251556003744_P24                        | MHTA       | NE   | Medium (45%)         | +0.05-0.10% NE | ⭐⭐⭐⭐   |
| 10   | H10 | PEIN-DCN Enhanced Experts                           | 1394659704917730_P16                        | PLE        | NE   | Medium (45%)         | +0.02-0.06% NE | ⭐⭐⭐     |
| 11   | H11 | MHTA Head Reduction (8→4-6)                         | (original)                                  | MHTA       | QPS  | Medium-High (50-65%) | +10-20% QPS    | ⭐⭐⭐     |
| 12   | H12 | Increase Experts per Task Group (1→2-3)             | (original)                                  | PLE        | NE   | Medium (45-55%)      | +0.05-0.15% NE | ⭐⭐⭐     |
| 13   | H13 | Expert Bottleneck Architecture                      | (original)                                  | PLE        | QPS  | Medium (45-55%)      | +15-25% QPS    | ⭐⭐⭐     |
| 14   | H14 | SEDOT Module Efficiency Optimization                | (original)                                  | Module     | QPS  | Medium (45-60%)      | +5-10% QPS     | ⭐⭐⭐     |
| 15   | H15 | DCN Gating Enhancement (GDCN-style)                 | (original)                                  | MHTA       | NE   | Medium (50-55%)      | +0.1-0.3% NE   | ⭐⭐⭐     |
| 16   | H16 | MHTA Reconfiguration (16×512 vs 8×1024)             | (original)                                  | MHTA       | NE   | Medium (50%)         | +0.05-0.2% NE  | ⭐⭐⭐     |
| 17   | H17 | SPECTRUM-HARVEST Feature Importance                 | 1219251556003744_P66                        | Features   | NE   | Medium (40%)         | +0.03-0.08% NE | ⭐⭐       |
| 18   | H18 | SPECTRUM-FUSION Pathway Integration                 | 1219251556003744_P66                        | Pre-PLE    | NE   | Low-Medium (40%)     | +0.02-0.06% NE | ⭐⭐       |
| 19   | H19 | Semantic ID Integration                             | (original)                                  | Features   | NE   | Medium-High (55-70%) | +0.1-0.2% NE   | ⭐⭐       |
| 20   | H20 | DCN Low-Rank Dimension Reduction                    | (original)                                  | MHTA       | QPS  | High (65%)           | +10-20% QPS    | ⭐⭐       |
| 21   | H21 | Multi-Embedding Architecture                        | (original)                                  | Arch       | NE   | Medium (40-45%)      | +0.2-0.5% NE   | ⭐⭐       |
| 22   | H22 | SYNAPSE-QUANTUM Codebook Compression                | 1219251556003744_P37                        | Embeddings | NE   | Medium (40%)         | +0.05-0.12% NE | ⭐⭐       |
| 23   | H23 | Content Intelligence Deep Integration               | (original)                                  | Features   | NE   | Medium (40-45%)      | +0.05-0.15% NE | ⭐⭐       |
| 24   | H24 | DCNv3-Style Explicit Head Supervision               | (original)                                  | MHTA       | NE   | Medium (45%)         | +0.1-0.3% NE   | ⭐⭐       |
| 25   | H25 | Remove/Simplify Content Intelligence Module         | (original)                                  | Module     | QPS  | Low-Medium (30-45%)  | +10-15% QPS    | ⭐         |
| 26   | H26 | Dual-Flow Sequence Encoding (DFGR-inspired)         | (original)                                  | HSTU       | QPS  | Medium (45%)         | +2-4x QPS      | ⭐         |
| 27   | H27 | Auxiliary Loss-Free Load Balancing (DeepSeek-style) | (original)                                  | PLE        | NE   | Low (25-40%)         | +0.02-0.08% NE | ⭐         |
| 28   | H28 | Task Group Reorganization Based on Gradient         | (original)                                  | PLE        | NE   | Medium (45%)         | +0.05-0.1% NE  | ⭐         |
| 29   | H29 | Learnable Auxiliary Loss Scaling                    | (original)                                  | Loss       | NE   | Low (30-40%)         | +0.02-0.05% NE | ⭐         |
| 30   | H30 | Replace Tanh with GELU in DCFN                      | (original)                                  | DCFN       | NE   | Low (30%)            | +0.01-0.03% NE | ⭐         |

---

## Tier 1: High Priority Hypotheses (Implement First)

### H1: Cross-Task Soft Label Enrichment (TASLE+CATALYST Merged)

**Source IDs**: `1394659704917730_P63` + `1219251556003744_P37`

**Category**: Loss Function | **Goal**: NE Improvement

**Description**: Unified cross-task soft label distillation combining two
complementary modes:

1. **TASLE Mode**: Anchor tasks (like, comment, share) provide soft supervision
   for semantically related tasks
2. **CATALYST Mode**: Task pairs exchange knowledge bidirectionally
   (pointwise↔pairwise)

**Key Innovation**:

- **ZERO inference cost** (training-only loss)
- Two complementary distillation modes in one module
- Temperature-scaled soft targets for smoother gradients
- Easy rollback via `tasle_weight` and `catalyst_weight` hyperparameters

**Task Correlations**:

```python
# TASLE Mode: Anchor → Semantically Related
SEMANTIC_CORRELATIONS = {
    "like": ["love_reaction", "haha_reaction", "wow_reaction", "support_reaction"],
    "comment": ["comment_surface_click", "comment_tap", "comment_vpvd"],
    "share": ["video_share"],
    "hide": ["hide_v2", "dislike", "report"],
}

# CATALYST Mode: Pointwise ↔ Pairwise Pairs
PAIRWISE_CORRELATIONS = [
    ("like", "like_pairwise"),
    ("comment", "comment_pairwise"),
    ("share", "share_pairwise"),
    ("linear_vpvd", "kd_linear_vpvd"),
    ("photo_click", "photo_click_pairwise"),
]
```

**Implementation Sketch**:

```python
class CrossTaskSoftLabelEnrichment(nn.Module):
    def __init__(self, temperature=2.0, tasle_weight=0.05, catalyst_weight=0.05):
        self.temperature = temperature
        self.tasle_weight = tasle_weight
        self.catalyst_weight = catalyst_weight

    def forward(self, task_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TASLE: anchor → related semantic distillation
        tasle_loss = self.compute_semantic_distillation(task_logits)
        # CATALYST: pointwise ↔ pairwise bidirectional distillation
        catalyst_loss = self.compute_pairwise_distillation(task_logits)
        return self.tasle_weight * tasle_loss + self.catalyst_weight * catalyst_loss
```

**Supporting Evidence**:

- TASLE inspired by P63 Post-click FM soft labels: ~0.15% NE gain observed
- CATALYST exploits natural task pair relationships already in model
- Zero inference cost - purely training enhancement
- Similar soft label approaches proven in knowledge distillation literature

**Contradicting Evidence**:

- May overlap with existing KD tasks (kd_like, kd_comment)
- Task correlations need validation
- Weight tuning required to balance with main losses

**Success Probability**: **High (60%)** **Potential Impact**: **Medium-High
(+0.03-0.10% NE)** **QPS Impact**: **0%**

---

### H2: PRISM-T - Task Complexity-Aware Tower Scaling

**Source ID**: `1219251556003744_P50`

**Category**: Task Tower | **Goal**: NE Improvement

**Description**: Replace uniform task towers with learnable complexity-aware
capacity allocation. Different tasks get different tower sizes (1x, 2x, 4x)
based on learned complexity scores.

**Key Innovation**:

- **Only hypothesis targeting TaskArchROOWrap directly**
- Backed by real P50 results: -0.08% NE on 2:click with +1.7% QPS improvement
- Learnable base complexity + input-dependent adjustment
- Soft mixture of scaled tower outputs

**Current State** (lines 730-737):

```python
task_arch=TaskArchROOWrap(
    task_names=model_config.task_names,
    task_configs=model_config.task_arch_configs,
    input_dim=256,
    embedding_groups=model_config.embedding_groups,
)
```

**Proposed Change**:

```python
class PRISMTScaleModule(nn.Module):
    def __init__(self, num_tasks, base_hidden_dims=[256, 128]):
        self.scale_levels = [1, 2, 4]  # 1x, 2x, 4x capacity
        self.task_base_complexity = nn.Parameter(torch.ones(num_tasks))
        self.complexity_estimator = nn.Sequential(
            nn.Linear(256, 64),
            SwishLayerNorm([64]),
            nn.Linear(64, num_tasks),
            nn.Sigmoid(),
        )
        # Build scaled towers for each task
        self.task_towers_per_scale = nn.ModuleList([...])

    def forward(self, ple_outputs, task_to_gate_idx):
        # Compute complexity scores
        complexity = self.complexity_estimator(ple_outputs)
        # Soft routing through scaled towers
        return weighted_tower_outputs
```

**Supporting Evidence**:

- P50 achieved -0.08% NE on 2:click with task aggregator scaling [2048, 1024,
  512, 256, 1]
- Different tasks have different complexity profiles (CTR vs CVR)
- Click prediction benefits from larger capacity
- Training dynamics optimization (constant→linear LR) also helps

**Contradicting Evidence**:

- Complex implementation with multiple tower copies
- May increase memory footprint
- Complexity estimator adds overhead

**Success Probability**: **Medium-High (55%)** **Potential Impact**: **High
(+0.05-0.15% NE)** **QPS Impact**: **0-3%**

---

### H3: SYNAPSE-NEXUS - HSTU-Guided Expert Routing

**Source ID**: `1219251556003744_P37`

**Category**: PLE Routing | **Goal**: NE Improvement

**Description**: Use HSTU's rich sequential user representation to dynamically
guide PLE expert selection, making the routing context-aware rather than purely
input-dependent.

**Key Innovation**:

- HSTU output becomes **active control signal** (not just passive feature)
- Soft modulation of existing gate outputs based on user context
- Sequential behavior patterns inform expert selection
- First hypothesis to connect HSTU directly to PLE routing

**Current State**: HSTU output concatenated to interaction_module_out but
doesn't influence routing.

**Proposed Change**:

```python
class SYNAPSENexusGating(nn.Module):
    def __init__(self, hstu_dim=512, num_expert_groups=5):
        self.routing_network = nn.Sequential(
            nn.Linear(hstu_dim, 256),
            SwishLayerNorm([256]),
            nn.Linear(256, num_expert_groups),
            nn.Softmax(dim=-1),
        )

    def forward(self, hstu_signal):
        return self.routing_network(hstu_signal)

# In CGCModule forward:
if nexus_weights is not None:
    task_routing_weight = nexus_weights[:, gate_index:gate_index+1]
    gate_blob = gate_blob * (1 + 0.1 * task_routing_weight)
    gate_blob = gate_blob / gate_blob.sum(dim=1, keepdim=True)
```

**Supporting Evidence**:

- HSTU learns rich temporal patterns encoding engagement recency, session
  dynamics
- Context-aware routing should improve task-specific expert utilization
- Similar gating mechanisms successful in MoE literature
- Makes PLE routing adaptive to user behavior

**Contradicting Evidence**:

- Adds complexity to CGCModule
- Routing signal may not be informative
- May cause gradient interference

**Success Probability**: **Medium (50%)** **Potential Impact**: **High
(+0.05-0.15% NE)** **QPS Impact**: **~3%**

---

### H4: Symmetric PLE Gating Design

**Category**: Model Module Refactor | **Goal**: NE Improvement

**Consensus Sources**: ALL original reports (unanimous)

**Description**: Convert asymmetric CGCModule gating to symmetric design where
ALL 4 task groups can access both task-specific AND shared experts.

**Current State** (Lines 505-519):

```python
# Gates 0,1: output_dim=1 (NO gating, trivial softmax)
# Gates 2,3: output_dim=2 (access task + shared experts)
out_features=num_expert_in_group * (2 if group_idx >= 2 else 1)
```

**Proposed Change**:

```python
# ALL gates access task-specific + shared experts
out_features=num_expert_in_group + num_shared_experts  # For ALL groups
```

**Supporting Evidence**:

- Original PLE paper uses symmetric design
- Groups 0,1 currently have **NO effective gating**
- Documentation flags this as "potential optimization opportunity"

**Success Probability**: **Medium-High (50-75%)** **Potential Impact**: **Medium
(+0.05-0.3% NE)** **Model Size Change**: Minimal

---

### H5: AURORA - Temporal Prediction Calibration

**Source ID**: `1219251556003744_P37_AURORA`

**Category**: Post-Prediction Calibration | **Goal**: NE Improvement

**Description**: Use HSTU-derived temporal signals to dynamically calibrate
task-level predictions and loss weighting, adapting confidence based on
engagement freshness.

**Key Innovation**:

- HSTU as **active control signal** for prediction calibration
- Temporal-based confidence scaling
- Loss weight modulation per user engagement pattern
- **Only hypothesis targeting post-prediction pipeline**

**Implementation**:

```python
class AURORACalibrationNetwork(nn.Module):
    def __init__(self, num_tasks, num_temporal_signals=4):
        self.temporal_encoder = nn.Sequential(
            nn.Linear(512, 256),  # hstu_dim
            SwishLayerNorm([256]),
            nn.Linear(256, num_temporal_signals),
            nn.Sigmoid(),
        )
        self.calibration_network = nn.Sequential(
            nn.Linear(num_temporal_signals, 64),
            SwishLayerNorm([64]),
            nn.Linear(64, num_tasks),
        )

    def forward(self, predictions, hstu_signal):
        temporal_signals = self.temporal_encoder(hstu_signal)
        calibration_factors = self.calibration_network(temporal_signals)
        return predictions * torch.sigmoid(calibration_factors)
```

**Supporting Evidence**:

- HSTU learns engagement recency, session dynamics, interest evolution
- Calibration helps on "predictable" vs "unpredictable" users
- Similar calibration approaches used in probabilistic forecasting

**Success Probability**: **Medium (50%)** **Potential Impact**: **Medium
(+0.03-0.08% NE)** **QPS Impact**: +1-2%

---

### H6: TSWSP - Temporal Saliency-Weighted Sequence Pooling

**Source ID**: `1394659704917730_P19`

**Category**: HSTU Enhancement | **Goal**: NE Improvement

**Description**: Add explicit temporal saliency signals (recency, engagement,
position) as attention bias in the HSTU PMA pooling mechanism.

**Key Innovation**:

- Explicit temporal saliency replaces implicit learning
- Uses **verified available** metadata: timestamps, presence_weights bitmask
- Learnable gating for when to use saliency hints
- **Only hypothesis targeting PMA pooling directly**

**Current State** (lines 1486-1490 in pytorch_modules_roo.py):

```python
compressed_user_embed = self.pma(
    query=repeated_seed_emb,
    key=user_only_embeddings_padded,
    value=user_only_embeddings_padded,
)
```

**Proposed Change**:

```python
class TemporalSaliencyWeightedPMA(nn.Module):
    def compute_temporal_saliency(self, timestamps, engagement_signals, current_time):
        recency_score = torch.exp(-time_deltas / (3600 * 24))  # 24h decay
        engagement_score = torch.log1p(engagement_signals.float()) / 5.0
        position_score = torch.linspace(0, 1, seq_len)
        return self.saliency_encoder(torch.stack([recency, engagement, position]))

    def forward(self, query, key, value, temporal_saliency=None):
        attn_scores = torch.matmul(q, k.T) * scale
        if temporal_saliency is not None:
            attn_scores = attn_scores + saliency_bias  # Add temporal bias
        return attended_output
```

**Supporting Evidence**:

- Session EBF principle: temporal segments have varying importance
- Verified data availability: `sparse_query_time`, `presence_weights`
- Engagement encoding available: like=1, comment=2, share=4, etc.

**Success Probability**: **Medium (50%)** **Potential Impact**: **Medium
(+0.02-0.08% NE)** **QPS Impact**: -1-3%

---

### H7: HSTU Depth Scaling (2 → 6 Layers)

**Category**: Model Architecture Change | **Goal**: NE Improvement

**Description**: Increase HSTU encoder depth from 2 layers to 6 layers following
Slimper V2 approach.

**Current State**: HSTU uses 2 encoder layers

**Proposed Change**: 6 encoder layers with activation checkpointing

**Supporting Evidence**:

- Slimper V2: Deeper model (3→6 layers) contributed significantly to +0.405% NE
- Deeper transformers learn more complex temporal dependencies
- Combined with other improvements provides compound gains

**Contradicting Evidence**:

- More layers = more latency and compute
- Diminishing returns beyond certain depth
- Need activation checkpointing for memory efficiency

**Success Probability**: **High (65-80%)** **Potential Impact**: **Medium-High
(+0.2-0.4% NE)** **Model Size Change**: +15-25% in HSTU parameters

---

## Tier 2: Medium Priority Hypotheses

### H8: Multi-Layer Progressive PLE

**Category**: Model Module Change | **Goal**: NE Improvement

**Description**: Stack 2-3 CGC layers to implement true "Progressive Layered
Extraction" instead of current single-layer design.

**Current State**: Single CGCModule layer

**Proposed Change**:

```python
num_ple_layers = 2
ple_layers = nn.ModuleList([
    CGCModule(task_experts_l1, gates_l1, shared_experts_l1),
    CGCModule(task_experts_l2, gates_l2, shared_experts_l2),
])
```

**Supporting Evidence**:

- "Progressive" in PLE refers to stacked layers
- Later layers can specialize more for individual tasks
- Model has 65+ tasks that could benefit

**Success Probability**: **Medium (50-65%)** **Potential Impact**: **Medium-High
(+0.1-0.3% NE)** **QPS Impact**: -10-15%

---

### H9: CASCADE - Asymmetric RO/NRO Depth + Cross-Stream Bridges

**Source ID**: `1219251556003744_P24`

**Category**: MHTA Enhancement | **Goal**: NE Improvement

**Description**: Apply asymmetric depth scaling to MHTA: RO stream gets 2
stacks, NRO stream gets 1 stack, plus cross-stream attention bridges.

**Key Innovation**:

- Uses existing RO/NRO infrastructure (lines 113-145) for selective capacity
- RO features are higher ROI → allocate more capacity
- Cross-stream bridges enable information exchange
- Inspired by LLaTTE's selective layer stacking

**Current State**:

```python
mhta_num_stack = 1  # Same depth for all heads
```

**Proposed Change**:

```python
mhta_ro_num_stack = 2   # Deeper for high-ROI RO features
mhta_nro_num_stack = 1  # Standard depth for NRO features

class CrossStreamBridge(nn.Module):
    def forward(self, ro_repr, nro_repr):
        q = self.query_proj(ro_repr)
        k, v = self.key_proj(nro_repr), self.value_proj(nro_repr)
        return ro_repr + self.out_proj(attention(q, k, v))
```

**Supporting Evidence**:

- LLaTTE achieved ~0.08% NE from 1→2 layer stacking
- Infrastructure for asymmetric treatment already exists
- Prior knowledge: "Deeper model is key to performance gain"

**Success Probability**: **Medium (45%)** **Potential Impact**: **Medium
(+0.05-0.10% NE)** **QPS Impact**: -8-12%

---

### H10: PEIN-DCN - Progressive Expert Interaction Networks

**Source ID**: `1394659704917730_P16`

**Category**: PLE Expert Enhancement | **Goal**: NE Improvement

**Description**: Replace simple MLP experts with lightweight DCN-enhanced
experts for feature cross interactions within each task group.

**Key Innovation**:

- **First hypothesis targeting PLE expert internals**
- DCN enables explicit feature cross interactions
- Follows principle: specialized modules benefit from dedicated interaction
  mechanisms
- Lightweight: low_rank_dim=64

**Current State** (lines 621-638):

```python
task_specific_experts_layer = [
    [nn.Sequential(
        nn.Linear(ple_input_dim, expert_layer_size),
        SwishLayerNorm([expert_layer_size]),
        nn.Linear(expert_layer_size, expert_layer_size),
        SwishLayerNorm([expert_layer_size]),
    ) for _ in range(num_expert_in_group)]
    for _ in range(num_expertgroup)
]
```

**Proposed Change**:

```python
class InteractionEnhancedExpert(nn.Module):
    def __init__(self, input_dim, output_dim, low_rank_dim=64):
        self.cross_net = DeepCrossNet(
            input_dim=input_dim,
            low_rank_dim=low_rank_dim,
            num_layers=2,
            output_dim=output_dim,
            residual_connection=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            SwishLayerNorm([output_dim]),
        )
```

**Supporting Evidence**:

- DCN already proven in model (MHTA heads)
- AFOC proposal showed DotCompressPP helps specialized modules
- Task-specific feature cross patterns differ by task group

**Success Probability**: **Medium (45%)** **Potential Impact**: **Low-Medium
(+0.02-0.06% NE)** **QPS Impact**: -3-6%

---

### H11: MHTA Head Reduction for QPS Improvement

**Category**: Model Module Refactor | **Goal**: QPS Improvement

**Description**: Reduce MHTA heads from 8 to 4-6 based on attention importance
analysis.

**Supporting Evidence**:

- Transformer pruning research shows 25-50% heads often redundant
- 8 parallel DCN heads is computationally expensive
- QPS secondary goal explicitly allows marginal NE regression

**Success Probability**: **Medium-High (50-65%)** **Potential Impact**: **High
for QPS (+10-20%)** **NE Risk**: -0.02% to -0.1%

---

### H12: Increase Experts per Task Group

**Category**: Model Module Change | **Goal**: NE Improvement

**Description**: Increase from 1 expert per task group to 2-3 experts.

**Current State**: `num_expert_in_group = 1`

**Supporting Evidence**:

- With only 1 expert, gating is trivial (100% weight)
- More experts allow learning different "views"
- DeepSeek-V3 uses 256 experts for hyper-specialization

**Success Probability**: **Medium (45-55%)** **Potential Impact**: **Medium
(+0.05-0.15% NE)**

---

### H13: Expert Bottleneck Architecture

**Category**: Model Module Refactor | **Goal**: QPS Improvement

**Description**: Apply bottleneck design to PLE experts.

**Proposed Change**:

```python
nn.Sequential(
    nn.Linear(ple_input_dim, 64),   # Down-project
    SwishLayerNorm([64]),
    nn.Linear(64, 128),             # Process
    SwishLayerNorm([128]),
    nn.Linear(128, 256),            # Up-project
    SwishLayerNorm([256]),
)
```

**Success Probability**: **Medium (45-55%)** **Potential Impact**: **Medium for
QPS (+15-25%)**

---

### H14: SEDOT Module Efficiency Optimization

**Category**: Model Module Refactor | **Goal**: QPS Improvement

**Description**: Optimize SEDOT by reducing `n_repeat` from 4 to 2 and
`num_output_tokens` from 32 to 16.

**Success Probability**: **Medium-High (45-60%)** **Potential Impact**: **Medium
for QPS (+5-10%)**

---

### H15: DCN Gating Enhancement (GDCN-style)

**Category**: Model Module Refactor | **Goal**: NE Improvement

**Description**: Add information gates to DCN layers within MHTA heads.

**Proposed Change**:

```python
GatedDeepCrossNet(
    input_dim=...,
    low_rank_dim=768,
    num_layers=4,  # Increased from 2
    use_information_gates=True,  # Key GDCN feature
)
```

**Supporting Evidence**:

- GDCN shows gating delays performance saturation from depth 3-4 to 8-10
- Current DCN uses only 2 layers

**Success Probability**: **Medium (50-55%)** **Potential Impact**: **Medium
(+0.1-0.3% NE)**

---

### H16: MHTA Reconfiguration (16×512 vs 8×1024)

**Category**: Model Module Refactor | **Goal**: NE Improvement

**Description**: Change MHTA from 8×1024 to 16×512 (same total dim) for more
diverse feature interactions.

**Success Probability**: **Medium (50%)** **Potential Impact**: **Low-Medium
(+0.05-0.2% NE)**

---

## Tier 3: Exploratory Hypotheses

### H17: SPECTRUM-HARVEST - Importance-Weighted Feature Selection

**Source ID**: `1219251556003744_P66`

**Category**: Feature Processing | **Goal**: NE Improvement

**Description**: Add learnable per-feature importance scores that implement
soft-pruning through attention gating.

**Key Innovation**:

- Quality-over-quantity feature selection
- Context-aware importance modulation
- Inspired by Ascent's feature pruning (rank > 1200, coverage < 0.2)

**Implementation**:

```python
class SPECTRUMHarvestModule(nn.Module):
    def __init__(self, num_features, embedding_dim):
        self.base_importance = nn.Parameter(torch.ones(num_features) / num_features)
        self.context_importance_net = nn.Sequential(
            nn.Linear(embedding_dim * num_features, 64),
            SwishLayerNorm([64]),
            nn.Linear(64, num_features),
        )

    def forward(self, features):
        importance_scores = torch.softmax(
            self.base_importance + self.context_importance_net(features), dim=-1
        )
        return features * importance_scores.unsqueeze(-1)
```

**Success Probability**: **Medium (40%)** **Potential Impact**: **Low-Medium
(+0.03-0.08% NE)** **QPS Impact**: -2-5%

---

### H18: SPECTRUM-FUSION - Multi-Pathway Quality-Weighted Integration

**Source ID**: `1219251556003744_P66`

**Category**: Feature Fusion | **Goal**: NE Improvement

**Description**: Add quality-weighted fusion across processing pathways (RO,
NRO, dense, sparse, MHTA) before PLE layer.

**Key Innovation**:

- Cross-pathway attention for interaction
- Learned quality weights per pathway
- Unified fusion replacing simple concatenation

**Success Probability**: **Low-Medium (40%)** **Potential Impact**: **Low-Medium
(+0.02-0.06% NE)** **QPS Impact**: -2-5%

---

### H19: Semantic ID Integration

**Category**: Feature Addition | **Goal**: NE Improvement

**Description**: Integrate RQ-VAE or RQ-KMeans generated semantic IDs as
additional embedding features.

**Supporting Evidence**:

- Meta research: +0.11% teen TS, +0.22% reshare on IG Reels
- 682× compression ratio enables efficient representation

**Success Probability**: **Medium-High (55-70%)** **Potential Impact**: **Medium
(+0.1-0.2% NE)**

---

### H20: DCN Low-Rank Dimension Reduction

**Category**: Model Ad-hoc Change | **Goal**: QPS Improvement

**Description**: Reduce DCN `low_rank_dim` from 768 to 384-512.

**Success Probability**: **High (65%)** **Potential Impact**: **Medium for QPS
(+10-20%)**

---

### H21: Multi-Embedding Architecture

**Category**: Architecture Change | **Goal**: NE Improvement

**Description**: Implement multi-embedding tables per feature (Tencent KDD
2024).

**Supporting Evidence**:

- ICML 2024 paper shows multi-embedding alleviates embedding collapse
- Tencent KDD 2024 reports +3.9% GMV

**Success Probability**: **Medium (40-45%)** **Potential Impact**: **High
(+0.2-0.5% NE)**

---

### H22: SYNAPSE-QUANTUM - Semantic Codebook Compression

**Source ID**: `1219251556003744_P37`

**Category**: Embedding Compression | **Goal**: NE/Memory Improvement

**Description**: Learn a discrete vocabulary of semantic tokens that captures
essential embedding structure using RQ-VAE style codebooks.

**Key Innovation**:

- Learnable codebook vocabulary (256 entries, 4 codebooks)
- Cross-attention from codebook queries to embeddings
- Semantic-preserving compression

**Implementation Sketch**:

```python
class SYNAPSEQuantumCodebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_codebook_entries=256):
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_codebook_entries, embedding_dim) * 0.1)
            for _ in range(4)  # 4 residual quantization layers
        ])
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)

    def forward(self, embeddings):
        # Residual quantization through codebooks
        # Cross-attention: codebook queries attend to embeddings
        return compressed_output, commitment_loss
```

**Success Probability**: **Medium (40%)** **Potential Impact**: **Medium
(+0.05-0.12% NE)** **QPS Impact**: ~5%

---

### H23-H30: Remaining Exploratory Hypotheses

| ID  | Hypothesis                            | Category | Impact         | Priority |
| --- | ------------------------------------- | -------- | -------------- | -------- |
| H23 | CI Deep Integration                   | Features | +0.05-0.15% NE | ⭐⭐     |
| H24 | DCNv3-Style Head Supervision          | MHTA     | +0.1-0.3% NE   | ⭐⭐     |
| H25 | Remove/Simplify CI Module             | Module   | +10-15% QPS    | ⭐       |
| H26 | Dual-Flow Sequence Encoding           | HSTU     | +2-4x QPS      | ⭐       |
| H27 | Aux Loss-Free Load Balancing          | PLE      | +0.02-0.08% NE | ⭐       |
| H28 | Task Group Reorganization by Gradient | PLE      | +0.05-0.1% NE  | ⭐       |
| H29 | Learnable Auxiliary Loss Scaling      | Loss     | +0.02-0.05% NE | ⭐       |
| H30 | Replace Tanh with GELU in DCFN        | DCFN     | +0.01-0.03% NE | ⭐       |

**Note**: H29 (Learnable Auxiliary Loss Scaling) demoted due to MultiBalance
neutral results - similar gradient/loss weight approaches likely have limited
impact.

---

## Implementation Roadmap

### Phase A: Quick Wins (1-2 weeks)

Zero/Low inference cost, fast validation

1. **H1: Cross-Task Soft Label Enrichment** - **0% inference** - Add unified
   distillation loss
2. **H4: Symmetric PLE Gating** - Simple config change

### Phase B: High-Impact NE (3-4 weeks)

Highest documented gains

3. **H2: PRISM-T-SCALE** - Task-adaptive towers
4. **H3: SYNAPSE-NEXUS** - HSTU-guided routing

### Phase C: Architecture Enhancements (4-6 weeks)

Medium-risk, proven approaches

5. **H5: AURORA** - Temporal calibration
6. **H6: TSWSP** - Saliency-weighted PMA
7. **H7: HSTU Depth Scaling** - 2→6 layers
8. **H8: Multi-Layer Progressive PLE**
9. **H9: CASCADE** - Asymmetric MHTA
10. **H10: PEIN-DCN** - DCN in experts

### Phase D: QPS Focus (Parallel track)

Secondary goal optimization

11. **H11: MHTA Head Reduction**
12. **H13: Expert Bottleneck**
13. **H14: SEDOT Optimization**
14. **H20: DCN Low-Rank Reduction**

### Phase E: Exploratory (Long-term)

Higher risk, potential high reward

15. **H17-H18: SPECTRUM** (HARVEST, FUSION)
16. **H22: SYNAPSE-QUANTUM** - Codebook compression
17. **H19: Semantic ID Integration**
18. **H21: Multi-Embedding Architecture**

---

## Complementarity Matrix

All new ideas target different components and can be combined:

```
Component Targets:
├── Loss Function: H1 (Cross-Task Soft Labels), H29 (Aux Loss - deprioritized)
├── TaskArch: H2 (PRISM-T) ← UNIQUE
├── PLE Routing: H3 (NEXUS), H4
├── PLE Experts: H10 (PEIN), H12, H13
├── Post-Prediction: H5 (AURORA) ← UNIQUE
├── HSTU PMA: H6 (TSWSP) ← UNIQUE
├── HSTU Encoder: H7
├── MHTA: H9 (CASCADE), H11, H15, H16
├── Features: H17-18 (SPECTRUM), H19, H23
└── Embeddings: H22 (QUANTUM)
```

**Removed**: MultiBalance Gradient Balancing (found neutral), SIM (already
proven/implemented)

---

## New Ideas Source Reference

| Hypothesis                 | Source ID                                   | Original Diff | Description              |
| -------------------------- | ------------------------------------------- | ------------- | ------------------------ |
| H1: Cross-Task Soft Labels | 1394659704917730_P63 + 1219251556003744_P37 | D84092558     | Merged TASLE+CATALYST    |
| H2: PRISM-T                | 1219251556003744_P50                        | D83085174     | CMF task arch scaling    |
| H3: NEXUS                  | 1219251556003744_P37                        | -             | SYNAPSE framework        |
| H5: AURORA                 | 1219251556003744_P37_AURORA                 | -             | POLARIS adaptation       |
| H6: TSWSP                  | 1394659704917730_P19                        | D84101796     | Session EBF              |
| H9: CASCADE                | 1219251556003744_P24                        | D82166497     | COFFEE LLaTTE            |
| H10: PEIN                  | 1394659704917730_P16                        | D84006565     | AFOC specialized modules |
| H17-18: SPECTRUM           | 1219251556003744_P66                        | -             | Harvest Bundle           |
| H22: QUANTUM               | 1219251556003744_P37                        | -             | SYNAPSE framework        |

---

## Risk Assessment & Mitigation

| Risk                 | Hypotheses Affected | Mitigation Strategy                             |
| -------------------- | ------------------- | ----------------------------------------------- |
| Training instability | H1, H8              | Use moving-average gradients, careful LR tuning |
| NE regression        | H11, H13, H14, H25  | Set regression threshold (<0.05%), ablate first |
| Increased latency    | H7, H8, H9, H10     | Activation checkpointing, batch size adjustment |
| Expert collapse      | H12                 | Monitor utilization, add load balancing         |
| Memory overflow      | H7, H21             | Gradient checkpointing, batch size reduction    |
| Implementation       | H2, H3, H9, H22     | Prototype separately, validate incrementally    |

---

## Success Metrics

### Primary Goal (NE Improvement)

- **Target**: +0.10% NE improvement (achievable with H1+H2+H3)
- **Stretch Goal**: +0.25% NE improvement (H1+H2+H3+H4 combined)
- **Minimum Acceptable**: +0.05% NE improvement

### Secondary Goal (QPS Improvement)

- **Target**: +15% QPS improvement with NE-neutral
- **Stretch Goal**: +25% QPS improvement
- **Maximum Acceptable NE Regression**: -0.02%

---

## Testing Protocol

1. **Baseline**: Current `model_roo_v0.py` configuration
2. **A/B Testing**: Minimum 7-day test period for statistical significance
   (p<0.05)
3. **Ablation Strategy**: For composite hypotheses, test individually first
4. **Rollback Criteria**:
   - NE regression > 0.05%: Immediate rollback
   - QPS degradation > 15%: Review and optimize
   - Training instability: Reduce LR or revert

---

## References

### Original References

1. **MultiBalance** - Meta (2024): +0.738% NE improvement
2. **GradCraft** - Kuaishou KDD 2024: Global gradient direction alignment
3. **Embedding Collapse** - Tsinghua/Tencent ICML 2024
4. **GDCN** - Gated Deep Cross Network
5. **DCNv3** - Tri-BCE loss for interpretable interactions
6. **LinRec** - Linear attention for efficiency
7. **PLE** - RecSys 2020: Progressive Layered Extraction
8. **HSTU** - Meta ICML 2024: Generative recommenders at scale
9. **DeepSeek-V3** - Auxiliary-loss-free load balancing
10. **Slimper V2 + SIM** - Meta internal: +0.575% NE combined
11. **DFGR** - Meituan: Dual-flow sequence encoding

### New Idea Source Proposals

12. **1394659704917730_P63** - Post-click FM v4 (TASLE): D84092558
13. **1219251556003744_P37** - POLARIS/SYNAPSE framework: CATALYST, NEXUS,
    QUANTUM
14. **1219251556003744_P37_AURORA** - POLARIS adaptation: Temporal calibration
15. **1219251556003744_P50** - CMF Task Arch Scaling (PRISM-T): D83085174
16. **1394659704917730_P19** - Session EBF (TSWSP): D84101796
17. **1219251556003744_P24** - COFFEE LLaTTE (CASCADE): D82166497
18. **1394659704917730_P16** - AFOC Specialized Modules (PEIN): D84006565
19. **1219251556003744_P66** - Harvest Bundle (SPECTRUM): HARVEST, FUSION

---

_Report consolidated and updated with 10 new innovative proposals_ _Generated:
December 15, 2025_ _Target model: model_roo_v0.py_ _Total hypotheses: 30
(removed 3 familiar + MultiBalance neutral + SIM proven, added 10 new, merged
TASLE+CATALYST)_
