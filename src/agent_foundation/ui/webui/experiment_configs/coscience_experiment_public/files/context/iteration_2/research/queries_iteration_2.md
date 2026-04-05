# Iteration 2 Deep Research Queries: Multi-Timescale Temporal Modeling

## Purpose

This document contains targeted research queries for Iteration 2 of the SYNAPSE recommendation system. These queries are designed for independent execution by AI deep research engines, with each query containing full context to enable standalone analysis.

---

## Background: SYNAPSE Architecture

**SYNAPSE** (**S**equential **Y**ield **N**etwork via **A**nalytical **P**rocessing and **S**calable **E**mbeddings) is a next-generation sequential recommendation architecture with four core components:

| Component | Function | Innovation |
|-----------|----------|------------|
| **SSD-FLUID** | Sequence backbone | O(N) training via State Space Duality, O(1) streaming inference |
| **PRISM** | Item representation | User-conditioned polysemous embeddings via shared hypernetwork |
| **FLUID** | Temporal modeling | Analytical closed-form decay instead of numerical ODE solvers |
| **Multi-Token** | Cross-sequence interaction | Orthogonal Alignment for complement discovery |

### FLUID Temporal Layer (Current Design)

FLUID uses analytical exponential decay to model temporal dynamics:

```
h(t) = h(t₀) · exp(-(t - t₀)/τ) + ∫ f(x(s))/τ · exp(-(t-s)/τ) ds
```

**Current limitation**: Fixed τ=24h for all content types.

---

## Iteration 1 Results Summary

### Performance by Component

| Component | Metric | Result | Target | Status |
|-----------|--------|--------|--------|--------|
| SSD-FLUID | Throughput | **8×** | 5-15× | ✅ Good |
| SSD-FLUID | NDCG | -1.8% | 0% | ⚠️ Trade-off |
| PRISM | Cold-start CTR | +2.5% | +2-5% | ✅ Meets |
| **FLUID** | **Re-engagement** | **+1.2%** | +2-4% | ⚠️ **Below target** |
| Multi-Token | NDCG | +0.8% | +0.5-1% | ✅ Good |
| Multi-Token | Latency | +18% | <10% | ❌ Exceeds budget |

### Critical Finding: Temporal Items Performance Gap

| Content Type | Re-engagement | Expected | Issue |
|--------------|---------------|----------|-------|
| Temporal items (news, events) | **-1.0%** | +1-2% | ❌ **NEGATIVE** |
| Non-temporal items (movies) | **+1.8%** | +1-3% | ✅ OK |

**Root Cause Identified**: Fixed τ=24h is fundamentally wrong for different content types:
- **News/Trending**: Needs τ ≈ 2-4 hours (τ=24h is 8× too slow → stale recommendations)
- **Movies/Default**: τ=24h is appropriate
- **Albums/Books**: Needs τ ≈ 168+ hours (τ=24h is too fast → forgets too quickly)

---

## Iteration 2 Research Hypothesis

**If we replace fixed τ=24h with learned per-category timescales (τ_fast ≈ 3h, τ_medium ≈ 24h, τ_slow ≈ 168h), temporal-sensitive items should improve from -1.0% to +1.5% re-engagement while maintaining non-temporal item performance.**

---

## Ranked Query List

| Rank | Query ID | Focus Area | Innovation Target |
|------|----------|------------|-------------------|
| 🥇 1 | Q1 | Multi-Timescale Architecture | Learnable per-category decay timescales |
| 🥈 2 | Q2 | Cross-Sequence Temporal Interactions | SSM Renaissance + Orthogonal Alignment |
| 🥉 3 | Q3 | Stable Timescale Learning | Training stability for learnable τ parameters |
| 4 | Q4 | Enhanced Multi-Token v2 | Latency reduction (+18% → +7%) |

---

## Detailed Query Specifications

### Query 1: Multi-Timescale Temporal Decay Architectures
**Rank**: 🥇 1 | **Category**: Temporal Modeling | **Innovation Potential**: ⭐⭐⭐⭐⭐

```
RESEARCH QUERY:

CONTEXT:
SYNAPSE is a sequential recommendation architecture where the FLUID temporal layer
uses analytical exponential decay: h(t) = h(t₀)·exp(-(t-t₀)/τ). In Iteration 1,
FLUID achieved +1.2% overall re-engagement but HURT temporal-sensitive items by
-1.0% (vs +1.8% for non-temporal). Root cause: fixed τ=24h is fundamentally wrong
for different content types—news needs ~3h decay (τ=24h is 8× too slow, causing
stale recommendations), while albums/books need ~168h (τ=24h forgets too quickly).

RESEARCH OBJECTIVE:
Investigate multi-timescale temporal decay architectures that learn content-specific
timescales to fix the τ mismatch problem identified in Iteration 1.

1. MULTI-TIMESCALE ARCHITECTURE PATTERNS:
   - Survey approaches that use multiple decay timescales:
     * TiM4Rec (SIGIR 2024): Category-specific decay in recommendations
     * Multi-timescale recurrent networks (MS-RNN, Clockwork RNN)
     * Hierarchical temporal modeling in continuous-time transformers
   - Analyze how many timescales are optimal: 2-tier, 3-tier, or continuous?
   - Investigate routing mechanisms: hard assignment vs soft attention over timescales

2. TIMESCALE LEARNING APPROACHES:
   - How can τ be learned end-to-end without divergence?
   - Compare parameterization strategies:
     * Sigmoid bounding: τ = τ_min + σ(x) × (τ_max - τ_min)
     * Softplus: τ = τ_min + softplus(x)
     * Log-space: τ = exp(log_τ) for numerical stability
   - Analyze gradient flow through τ parameters
   - Investigate whether τ should be item-specific, category-specific, or user-specific

3. CONTENT-AWARE TIMESCALE ASSIGNMENT:
   - What signals should determine timescale assignment?
     * Content type (news, movies, albums, books)
     * Item age and freshness
     * User engagement patterns
     * Popularity dynamics
   - Survey content-aware recommendation approaches that adapt to item characteristics
   - Investigate whether timescale should be static or dynamic per interaction

4. EMPIRICALLY OPTIMAL TIMESCALES:
   - What are optimal τ values for different content categories?
   - Survey industry practices for temporal sensitivity:
     * Social media feeds: minutes to hours
     * News: hours to days
     * Movies: days to weeks
     * Books/Albums: weeks to months
   - Analyze A/B test results from literature on temporal recommendation

5. SYNTHESIS:
   - Design a 3-tier multi-timescale FLUID architecture:
     * τ_fast ≈ 2-4 hours (news, trending, live events)
     * τ_medium ≈ 24 hours (movies, default content)
     * τ_slow ≈ 168 hours (albums, books, evergreen)
   - Propose soft routing mechanism between timescales
   - Estimate expected improvement: temporal items from -1.0% to +1.5%

Return: Comprehensive analysis with specific architectural recommendations for
multi-timescale FLUID, including parameterization strategies, routing mechanisms,
and expected performance improvements over fixed τ=24h.
```

**Reasoning**: This is the PRIMARY research focus for Iteration 2. The -1.0% vs +1.8% performance gap directly points to fixed τ as the bottleneck. Multi-timescale approaches from TiM4Rec and MS-RNN literature provide proven patterns to adapt.

---

### Query 2: Cross-Sequence Temporal Interactions (SSM Renaissance + Orthogonal Alignment)
**Rank**: 🥈 2 | **Category**: Cross-Domain Transfer | **Innovation Potential**: ⭐⭐⭐⭐⭐

```
RESEARCH QUERY:

CONTEXT:
SYNAPSE's Multi-Token Interaction component uses cross-sequence attention to discover
complementary information across user sequences, based on the Orthogonal Alignment
Thesis (cross-attention surfaces information orthogonal to self-attention). In
Iteration 1, Multi-Token achieved +0.8% NDCG but with +18% latency overhead (target:
<10%). Recent SSM Renaissance research (2024-2025) shows State Space Models achieving
competitive or superior performance to Transformers on sequence modeling tasks.

RESEARCH OBJECTIVE:
Investigate how SSM Renaissance findings and the Orthogonal Alignment Thesis can be
combined to enhance temporal modeling while reducing Multi-Token latency overhead.

1. SSM RENAISSANCE (2024-2025):
   - Survey recent advances in State Space Models:
     * Mamba-2 and its dual formulation (Linear Attention ≡ SSM)
     * Multi-Scale SSM (MS-SSM) for hierarchical temporal patterns
     * SIGMA's bidirectional SSM for offline training
     * Hydra architectures combining SSM with attention
   - Analyze how SSMs handle multi-timescale patterns natively
   - Investigate whether SSM's selective mechanism can replace explicit timescale routing

2. ORTHOGONAL ALIGNMENT IN TEMPORAL CONTEXT:
   - How does the Orthogonal Alignment Thesis apply to temporal sequences?
   - What complementary temporal information can cross-sequence interaction surface?
     * Seasonal patterns (same time yesterday, last week, last year)
     * Contextual similarity (similar situations across time)
     * Interest evolution patterns
   - Investigate temporal attention mechanisms that leverage Orthogonal Alignment

3. SSM-BASED CROSS-SEQUENCE INTERACTION:
   - Can SSM-style recurrence replace attention for cross-sequence interaction?
   - Analyze complexity: SSM O(N) vs attention O(N²) for cross-sequence
   - Investigate hybrid approaches: SSM for within-sequence, attention for cross-sequence
   - Survey dual representation approaches (GenSR, IADSR) for efficiency

4. LATENCY OPTIMIZATION:
   - How can Multi-Token latency be reduced from +18% to <10%?
   - Investigate techniques:
     * Grouped Query Attention (GQA) for key-value sharing
     * Sparse attention patterns (local + global)
     * Early-exit mechanisms for simple queries
     * Distillation from full attention to efficient variants
   - Analyze quality-latency tradeoffs for each approach

5. SYNTHESIS:
   - Propose Enhanced Multi-Token v2 architecture that:
     * Leverages SSM Renaissance for efficient temporal modeling
     * Applies Orthogonal Alignment for cross-sequence discovery
     * Achieves <10% latency overhead (down from +18%)
     * Maintains or improves +0.8% NDCG gain
   - Estimate implementation complexity and expected gains

Return: Analysis of SSM Renaissance techniques applicable to cross-sequence
interaction, with specific architectural proposals for Enhanced Multi-Token v2
that reduces latency while preserving quality gains.
```

**Reasoning**: Multi-Token's +18% latency exceeds production budget. SSM Renaissance research provides efficient alternatives to attention, while Orthogonal Alignment ensures we don't lose the complementary information discovery capability.

---

### Query 3: Stable Training for Learnable Timescale Parameters
**Rank**: 🥉 3 | **Category**: Training Methodology | **Innovation Potential**: ⭐⭐⭐⭐

```
RESEARCH QUERY:

CONTEXT:
SYNAPSE's FLUID layer will be extended with learnable timescale parameters
(τ_fast, τ_medium, τ_slow) that route different content types to appropriate
decay rates. Timescale parameters interact with exponential functions
(exp(-Δt/τ)), which can cause gradient instability: very small τ leads to
vanishing gradients, while very large τ causes slow learning. The goal is to
learn τ values that fix the temporal item performance gap (-1.0% → +1.5%).

RESEARCH OBJECTIVE:
Investigate training methodologies that enable stable end-to-end learning of
timescale parameters without gradient explosion, collapse, or degenerate solutions.

1. GRADIENT DYNAMICS FOR TIMESCALE PARAMETERS:
   - Analyze gradient flow through exponential decay: ∂L/∂τ = ∂L/∂h × ∂h/∂τ
   - What causes instability? (exp overflow, vanishing gradients at extreme τ)
   - How do different τ ranges affect learning dynamics?
   - Investigate second-order effects and curvature of the loss landscape

2. PARAMETERIZATION STRATEGIES:
   - Compare approaches for bounded τ learning:
     * Sigmoid bounding: τ = τ_min + σ(θ) × (τ_max - τ_min)
       - Pros: Strict bounds, smooth gradients
       - Cons: Saturation at extremes
     * Softplus: τ = τ_min + softplus(θ)
       - Pros: Unbounded above, smooth near zero
       - Cons: No upper bound
     * Log-space: τ = exp(log_τ)
       - Pros: Scale-invariant learning
       - Cons: Requires careful initialization
   - What initialization works best for each parameterization?
   - How does parameterization affect convergence speed?

3. TIMESCALE SEPARATION AND COLLAPSE PREVENTION:
   - How do we prevent all timescales from collapsing to the same value?
   - Investigate regularization techniques:
     * Separation loss: λ × |τ_fast - τ_slow|^(-1)
     * Diversity regularization on routing weights
     * Minimum usage constraints per timescale
   - Analyze temperature annealing for soft routing between timescales
   - How to balance regularization strength vs learning flexibility?

4. CURRICULUM AND WARM-START STRATEGIES:
   - Should τ be frozen initially and fine-tuned later?
   - Investigate curriculum learning for timescale parameters:
     * Start with fixed τ (known good values from literature)
     * Gradually increase learning rate for τ parameters
     * Progressive unfreezing of timescale layers
   - How long should warm-up be? What metrics indicate τ is ready to learn?

5. VALIDATION AND MONITORING:
   - What metrics indicate healthy timescale learning?
     * τ value distribution across training
     * Gradient magnitude for τ parameters
     * Per-timescale routing entropy
   - How to detect degenerate solutions early?
   - What checkpointing strategy for timescale parameters?

6. SYNTHESIS:
   - Propose a training recipe for Multi-Timescale FLUID:
     * Parameterization: Sigmoid bounding with log-space initialization
     * Initialization: τ_fast=3h, τ_medium=24h, τ_slow=168h (from literature)
     * Regularization: Cosine separation with temperature annealing
     * Curriculum: Freeze τ for 10% of training, then gradual unfreezing
   - Provide monitoring checklist for τ parameter health
   - Estimate training overhead vs fixed τ baseline

Return: Comprehensive training methodology for learnable timescale parameters,
including parameterization choice, initialization, regularization, curriculum
strategy, and monitoring guidelines.
```

**Reasoning**: Learnable timescales are prone to instability (exponential gradients) and degenerate solutions (all τ collapse to same value). This query ensures we have stable training before implementing multi-timescale FLUID.

---

### Query 4: Enhanced Multi-Token v2 Latency Optimization
**Rank**: 4 | **Category**: Efficiency | **Innovation Potential**: ⭐⭐⭐⭐

```
RESEARCH QUERY:

CONTEXT:
SYNAPSE's Multi-Token Interaction component uses cross-sequence attention to
discover complementary items, achieving +0.8% NDCG improvement. However, the
+18% latency overhead exceeds the production budget of <10%. In Iteration 2,
we need to reduce latency to +7% while maintaining the NDCG gain.

The current implementation uses standard multi-head attention:
- Keys/Values: item embeddings from other sequences
- Queries: current user's hidden states
- Full O(N×M) computation where N=sequence length, M=candidate pool

RESEARCH OBJECTIVE:
Investigate techniques to reduce Multi-Token latency from +18% to +7% while
preserving the +0.8% NDCG gain from cross-sequence interaction.

1. GROUPED QUERY ATTENTION (GQA):
   - How does GQA reduce memory and compute?
   - What grouping factor (K=4, K=8) provides best quality-latency tradeoff?
   - Survey GQA applications in production systems (Llama 2, Mistral)
   - Analyze GQA impact on cross-sequence attention specifically
   - Estimate: GQA with K=4 typically provides 2-4× speedup with <0.1% quality loss

2. SPARSE ATTENTION PATTERNS:
   - What sparse patterns work for cross-sequence interaction?
     * Local attention: only attend to temporally nearby items
     * Global tokens: attend to a small set of summary tokens
     * Top-k attention: only compute attention for most relevant items
   - Survey efficient attention implementations (Flash Attention, xFormers)
   - How much sparsity is possible before quality degrades?
   - Investigate learned vs fixed sparsity patterns

3. CACHING AND REUSE:
   - What can be precomputed and cached for cross-sequence attention?
     * Item key/value projections (static per item)
     * Candidate pool embeddings (slowly changing)
   - Analyze cache invalidation strategies for dynamic recommendations
   - Estimate memory vs latency tradeoff for different cache sizes

4. DISTILLATION APPROACHES:
   - Can full attention be distilled into efficient variants?
   - Investigate:
     * Teacher: Full cross-sequence attention
     * Student: GQA + sparse attention
     * Distillation loss: KL divergence on attention distributions
   - How much of the +0.8% NDCG is preserved after distillation?

5. ARCHITECTURE ALTERNATIVES:
   - Can cross-sequence interaction be achieved without full attention?
     * Pooling-based interaction (mean/max over candidate pool)
     * Low-rank factorization of attention matrix
     * Retrieval-augmented approaches (retrieve top-k, then attend)
   - Compare quality-latency tradeoffs for each alternative

6. SYNTHESIS:
   - Propose Enhanced Multi-Token v2 architecture:
     * Base: GQA with K=4 (grouping factor)
     * Attention: Local (window=32) + Global (8 summary tokens)
     * Caching: Pre-computed item key-value pairs
     * Expected: +7% latency (down from +18%), +0.7% NDCG (preserved)
   - Provide implementation roadmap with complexity estimates
   - Identify go/no-go criteria for production deployment

Return: Detailed analysis of latency optimization techniques for cross-sequence
attention, with specific architectural recommendations for Enhanced Multi-Token v2
that achieves +7% latency with preserved NDCG gains.
```

**Reasoning**: Multi-Token's +18% latency is a blocker for production deployment. GQA and sparse attention are proven techniques that can achieve 2-3× speedup with minimal quality loss. This query ensures we have a concrete path to production-ready latency.

---

## Expected Outcomes

| Query | Expected Finding | Application to SYNAPSE v2 |
|-------|------------------|---------------------------|
| Q1 | 3-tier timescale architecture with soft routing | Multi-Timescale FLUID base design |
| Q2 | SSM-based cross-sequence with GQA | Enhanced Multi-Token v2 architecture |
| Q3 | Sigmoid bounding + cosine separation | Stable τ parameter training |
| Q4 | GQA K=4 + local-global attention | Production-ready latency |

---

## Success Criteria

Iteration 2 research is successful if we can answer:

1. **What architecture?** → Multi-Timescale FLUID with 3-tier learned τ (fast/medium/slow)
2. **How to train?** → Stable gradient flow with sigmoid-bounded τ and separation regularization
3. **What parameters?** → τ_fast≈3h, τ_medium≈24h, τ_slow≈168h (learned from initialization)
4. **What efficiency?** → Enhanced Multi-Token v2 with GQA K=4, achieving +7% latency

---

## Target Metrics (Iteration 2)

| Metric | Iteration 1 | Iteration 2 Target | Improvement |
|--------|-------------|-------------------|-------------|
| Overall re-engagement | +1.2% | **+2.2%** | +1.0% |
| Temporal items | **-1.0%** | **+1.5%** | +2.5% swing |
| Non-temporal items | +1.8% | +2.5% | +0.7% |
| Multi-Token latency | +18% | **+7%** | -11% |
| Throughput | 8× | 8× | Maintained |

---

## Machine-Readable JSON

```json
{
  "metadata": {
    "version": "2.0",
    "iteration": 2,
    "created": "2026-01-08",
    "purpose": "Targeted research queries for SYNAPSE v2 temporal modeling improvements",
    "context": "Iteration 1 showed fixed τ=24h HURTS temporal items (-1.0%). Iteration 2 focuses on multi-timescale FLUID to fix this gap.",
    "architecture": "SYNAPSE: SSD-FLUID backbone + PRISM hypernetwork + FLUID temporal + Multi-Token interaction"
  },
  "iteration_1_summary": {
    "ssd_fluid_throughput": "8×",
    "prism_cold_start": "+2.5%",
    "fluid_overall": "+1.2%",
    "fluid_temporal_items": "-1.0%",
    "fluid_non_temporal": "+1.8%",
    "multi_token_ndcg": "+0.8%",
    "multi_token_latency": "+18%",
    "root_cause": "Fixed τ=24h is 8× too slow for news (needs ~3h) and too fast for albums (needs ~168h)"
  },
  "queries": [
    {
      "id": "Q1",
      "rank": 1,
      "title": "Multi-Timescale Temporal Decay Architectures",
      "category": "Temporal Modeling",
      "innovation_potential": 5,
      "focus_areas": ["TiM4Rec", "MS-RNN", "Learnable timescales", "3-tier decay"],
      "expected_insights": ["Optimal number of timescales", "Routing mechanisms", "τ parameterization"],
      "target_improvement": "Temporal items: -1.0% → +1.5%",
      "query_summary": "Investigate multi-timescale architectures to learn content-specific decay timescales, replacing fixed τ=24h with τ_fast≈3h, τ_medium≈24h, τ_slow≈168h"
    },
    {
      "id": "Q2",
      "rank": 2,
      "title": "Cross-Sequence Temporal Interactions (SSM Renaissance)",
      "category": "Cross-Domain Transfer",
      "innovation_potential": 5,
      "focus_areas": ["SSM Renaissance", "Mamba-2", "Orthogonal Alignment", "MS-SSM"],
      "expected_insights": ["SSM for cross-sequence", "Temporal Orthogonal Alignment", "Hybrid architectures"],
      "target_improvement": "Leverage SSM efficiency for cross-sequence interaction",
      "query_summary": "Combine SSM Renaissance findings with Orthogonal Alignment Thesis for efficient temporal cross-sequence modeling"
    },
    {
      "id": "Q3",
      "rank": 3,
      "title": "Stable Training for Learnable Timescale Parameters",
      "category": "Training Methodology",
      "innovation_potential": 4,
      "focus_areas": ["Gradient stability", "τ parameterization", "Separation regularization"],
      "expected_insights": ["Sigmoid bounding strategy", "Collapse prevention", "Curriculum learning"],
      "target_improvement": "Stable τ learning without degenerate solutions",
      "query_summary": "Ensure stable end-to-end learning of τ parameters with proper parameterization, initialization, and regularization"
    },
    {
      "id": "Q4",
      "rank": 4,
      "title": "Enhanced Multi-Token v2 Latency Optimization",
      "category": "Efficiency",
      "innovation_potential": 4,
      "focus_areas": ["GQA", "Sparse attention", "Caching", "Distillation"],
      "expected_insights": ["GQA grouping factor", "Sparse patterns", "Cache strategies"],
      "target_improvement": "Latency: +18% → +7%",
      "query_summary": "Reduce Multi-Token latency from +18% to +7% using GQA, sparse attention, and caching while preserving +0.8% NDCG"
    }
  ]
}
```

---

*Document Version: 2.0*
*Iteration: 2*
*Created: January 2026*
*Purpose: Targeted deep research queries for SYNAPSE v2 multi-timescale temporal modeling*
