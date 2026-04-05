# Q3 Proposal: PRISM Hypernetwork for User-Conditioned Embeddings

**Research Query**: Contextual and Polysemous Representations for Recommendation
**Theme**: Representational Innovation via Shared Hypernetworks
**Innovation Target**: Scalable user-conditioned item embeddings solving cold-start and polysemy

---

## Executive Summary

This proposal addresses **Validity Wall 2: Semantic Void of ID-Based Learning** through the PRISM (Personalized Representation via Intersection-based Subspace Modeling) hypernetwork architecture:

- **User-conditioned embeddings**: Same item → different representations for different users
- **Scalable architecture**: O(num_items × code_dim + generator_size) vs O(num_items × generator_size)
- **Cold-start solution**: Content-derived codes for new items without retraining

---

## Section 1: Problem Statement

### 1.1 The Polysemy Problem in Recommendation

NLP faced and solved the polysemy problem:
- **Word2Vec (2013)**: "bank" = single vector (river? money?)
- **ELMo (2018)**: "bank" = context-dependent vector
- **BERT (2019)**: Full contextual understanding

**RecSys is stuck at 2013**: Every item has exactly one embedding regardless of user context.

### 1.2 Concrete Examples

| Item | User Context | Desired Meaning | Current System |
|------|--------------|-----------------|----------------|
| "Barbie" | Feminist scholar | Social satire | Same vector |
| "Barbie" | Parent | Family movie | Same vector |
| "Barbie" | Meme enthusiast | Barbenheimer | Same vector |
| "iPhone" | Developer | Dev device | Same vector |
| "iPhone" | Photographer | Camera quality | Same vector |
| "iPhone" | Business user | Productivity | Same vector |

**Impact**: Massive personalization opportunity lost. The same item should have different relevance signals for different users.

### 1.3 Why Current Approaches Fail

| Approach | What It Does | Why It Fails |
|----------|--------------|--------------|
| Larger embeddings | More parameters | Still user-agnostic |
| Multi-embedding tables | Multiple views | Each view is still static |
| Pre-trained encoders | Add content | Content is not personalized |
| User-item interaction | Late fusion | Embedding itself is not user-conditioned |

---

## Section 2: The PRISM Solution

### 2.1 Core Idea: Shared Hypernetwork with Item Codes

**Key Insight**: Instead of storing full embeddings or full generators per item, store:
1. **Small item codes** (16-64 dims per item)
2. **Shared hypernetwork** (generates embeddings from user_state + item_code)

```python
# Standard: Static lookup
item_embedding = embedding_table[item_id]  # Same for all users

# PRISM: User-conditioned generation
item_embedding = shared_generator(user_state, item_code[item_id])  # Different per user
```

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRISM Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐         │
│  │  Item Code  │───▶│     Shared      │───▶│   Output    │         │
│  │   (64-dim)  │    │   Hypernetwork  │    │  Embedding  │         │
│  └─────────────┘    │                 │    │  (64-dim)   │         │
│        ↑            │  Linear → GELU  │    └─────────────┘         │
│        │            │  Linear → GELU  │           ↑                │
│   item_id           │  Linear         │           │                │
│                     └─────────────────┘      user-conditioned      │
│                            ↑                                       │
│                     ┌─────────────┐                                │
│                     │ User State  │                                │
│                     │ (128-dim)   │                                │
│                     └─────────────┘                                │
│                            ↑                                       │
│                     from SSD-FLUID                                 │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation

```python
class PRISMEmbedding(nn.Module):
    """Scalable user-conditioned item embeddings.

    Memory Analysis:
    - Item codes: 10M × 64 × 4 bytes = 2.5 GB (manageable)
    - Shared generator: ~1M params = ~4 MB (constant)
    - Total: ~2.5 GB vs ~1 TB for per-item generators
    """

    def __init__(
        self,
        num_items: int,
        code_dim: int = 64,
        user_dim: int = 128,
        item_dim: int = 64,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Per-item: small code vector
        self.item_codes = nn.Embedding(num_items, code_dim)

        # Shared: one generator for all items
        self.shared_generator = nn.Sequential(
            nn.Linear(user_dim + code_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, item_dim)
        )

        # Optional: residual from base embedding
        self.base_embedding = nn.Embedding(num_items, item_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, item_ids: Tensor, user_state: Tensor) -> Tensor:
        """Generate user-conditioned embeddings."""
        codes = self.item_codes(item_ids)
        combined = torch.cat([user_state, codes], dim=-1)
        dynamic = self.shared_generator(combined)

        # Blend with static base
        base = self.base_embedding(item_ids)
        return torch.sigmoid(self.alpha) * dynamic + (1 - torch.sigmoid(self.alpha)) * base
```

---

## Section 3: Scalability Analysis

### 3.1 Memory Comparison

| Approach | Formula | 10M Items | 100M Items | Feasible? |
|----------|---------|-----------|------------|-----------|
| Per-item generators | num_items × generator_size | ~1 TB | ~10 TB | ❌ |
| **PRISM (Shared)** | num_items × code_dim + generator | ~2.5 GB | ~25 GB | ✅ |
| Static embeddings | num_items × embed_dim | ~2.5 GB | ~25 GB | ✅ |

**Key Insight**: PRISM has **same memory** as static embeddings but **user-conditioned output**.

### 3.2 Computational Overhead

| Operation | Static Embedding | PRISM | Overhead |
|-----------|-----------------|-------|----------|
| Forward (per item) | O(1) lookup | O(1) lookup + O(d²) MLP | ~2× |
| Backward (per item) | O(d) | O(d²) | ~d× |
| Training batch | Negligible | ~10-20% | Acceptable |

**Conclusion**: Modest computational overhead for significant capability gain.

---

## Section 4: Cold-Start Extension

### 4.1 The Cold-Start Problem

New items have zero interactions → zero learned embeddings → cannot recommend.

Traditional solutions:
- Random initialization (poor initial quality)
- Content-based fallback (separate system, no personalization)
- Popularity-based (ignores user preferences)

### 4.2 PRISM Cold-Start Solution

For new items, **generate item codes from content**:

```python
class ContentConditionedPRISM(PRISMEmbedding):
    """Extended PRISM with content-based codes for cold-start."""

    def __init__(self, num_items: int, text_encoder: nn.Module, **kwargs):
        super().__init__(num_items, **kwargs)
        self.text_encoder = text_encoder  # Pre-trained, frozen
        self.content_projection = nn.Linear(
            text_encoder.output_dim,
            kwargs.get('code_dim', 64)
        )

    def get_code_from_content(self, text: str) -> Tensor:
        """Generate item code from text description."""
        with torch.no_grad():
            text_emb = self.text_encoder(text)
        return self.content_projection(text_emb)

    def forward_with_content(
        self,
        item_ids: Tensor,
        content_codes: Tensor,
        user_state: Tensor,
        is_new_item: Tensor
    ) -> Tensor:
        """Handle both warm and cold items."""
        codes = self.item_codes(item_ids)
        codes = torch.where(is_new_item.unsqueeze(-1), content_codes, codes)

        combined = torch.cat([user_state, codes], dim=-1)
        return self.shared_generator(combined)
```

**Benefit**: Cold items get **user-conditioned** recommendations from day 1.

---

## Section 5: Theoretical Foundation

### 5.1 Why PRISM Works: Low-Rank Factorization View

PRISM can be understood as implicit low-rank factorization:

```
Static: E = embedding_table[item_id]  ∈ R^d

PRISM: E = f(W_user · user_state + W_code · item_code)
       ≈ U · user_state + V · item_code  (linearized)
       = UV^T interaction implicitly
```

The shared generator learns **how user preferences interact with item characteristics**.

### 5.2 Expressiveness Analysis

| Model | User-Item Interaction | Personalization | Scalability |
|-------|----------------------|-----------------|-------------|
| Matrix Factorization | Bilinear | Per-user factors | O(users × d) |
| Deep Learning | Implicit in layers | Late fusion | O(items × d) |
| **PRISM** | Explicit generation | Early fusion | O(items × code + gen) |

PRISM achieves **early fusion** of user and item signals, enabling richer personalization.

---

## Section 6: Integration with SYNAPSE

### 6.1 Data Flow

```
User History [items, timestamps]
        │
        ▼
┌──────────────────┐
│    SSD-FLUID     │  ← Processes sequence with O(N) complexity
│     Backbone     │     and continuous-time awareness
└────────┬─────────┘
         │
    user_state (compressed representation)
         │
         ▼
┌──────────────────┐
│      PRISM       │  ← Generates user-conditioned item embeddings
│   Hypernetwork   │     for candidate items
└────────┬─────────┘
         │
    item_embeddings|user
         │
         ▼
┌──────────────────┐
│   Ranking Head   │  ← Scores candidates
└──────────────────┘
```

### 6.2 Why PRISM Needs SSD-FLUID

PRISM requires a **rich user state** to condition on:
- Poor user state → poor conditioning → no personalization gain
- SSD-FLUID provides **lifetime context** in compressed form
- The synergy: efficient backbone enables effective conditioning

---

## Section 7: Implementation Roadmap

### Phase 1: Core PRISM (Weeks 1-4)

- [ ] Implement shared hypernetwork architecture
- [ ] Integrate with base embedding (residual connection)
- [ ] Benchmark against static embeddings
- [ ] **Milestone**: User-conditioned generation verified

### Phase 2: Cold-Start Extension (Weeks 5-6)

- [ ] Add content encoder integration
- [ ] Implement content → code projection
- [ ] Evaluate cold-start performance
- [ ] **Milestone**: Cold-start solution validated

### Phase 3: Integration (Weeks 7-8)

- [ ] Integrate with SSD-FLUID backbone
- [ ] End-to-end training pipeline
- [ ] Ablation studies
- [ ] **Milestone**: Full PRISM + SSD-FLUID system

---

## Section 8: Expected Results

### 8.1 Quantitative Targets

| Metric | Baseline | PRISM Target | Improvement |
|--------|----------|--------------|-------------|
| Cold-Start NDCG@10 | 0.05-0.07 | 0.08-0.10 | +40-60% |
| Sparse User NDCG@10 | 0.25 | 0.30 | +20% |
| Memory Overhead | 0 | ~2.5 GB | Acceptable |
| Training Overhead | 0 | ~10-20% | Acceptable |

### 8.2 Qualitative Benefits

1. **Polysemous Items**: "Barbie" means different things to different users
2. **Cold-Start**: New items get personalized recommendations immediately
3. **Transfer Learning**: User state enables cross-domain personalization
4. **Interpretability**: Item codes can be analyzed for semantic meaning

---

## Section 9: Risk Analysis

| Risk | Probability | Mitigation | Go/No-Go |
|------|-------------|------------|----------|
| PRISM overhead without gains | 25% | Start with residual to base | Revert if <1% cold-start improvement |
| Generator collapse | 15% | Diversity regularization | Monitor embedding variance |
| Cold-start codes poor | 20% | Better content encoders | Graceful degradation to base |

---

## Section 10: Conclusion

PRISM addresses Validity Wall 2 (Semantic Void) by:

1. **User-Conditioned Generation**: E_item|user = f(user_state, item_code)
2. **Scalable Architecture**: Same memory as static embeddings
3. **Cold-Start Solution**: Content → code for new items

This is the second pillar of SYNAPSE, enabling personalized item semantics at scale.

---

*Document Version: 1.0*
*Research Query: Q3 - Contextual Representations*
*Themed Focus: PRISM Hypernetwork - User-Conditioned Embeddings*
