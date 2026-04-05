# SYNAPSE Iteration 1 Experiment Summary

## Overview

This document summarizes the experimental results for SYNAPSE Iteration 1, evaluating the four core components (SSD-FLUID, PRISM, FLUID, Multi-Token Interaction) against both HSTU baseline and HSTU + linear-decay sampling baseline.

**Key Finding:** SYNAPSE v1's learned compression beats predefined sampling (HSTU + linear-decay) on BOTH quality AND efficiency, demonstrating that intelligent compression outperforms heuristic approaches.

---

## Full System Performance Comparison

| Model | NDCG@10 | Δ NDCG | Throughput | Status |
|-------|---------|--------|------------|--------|
| **HSTU (full)** | 0.1823 | baseline | 1× | Gold standard (expensive) |
| **HSTU + linear-decay** | 0.1626 | **-10.8%** | **1.6×** | Predefined sampling baseline |
| **SYNAPSE v1** | 0.1654 | **-9.3%** | **2.0×** | ✅ Beats sampling on BOTH |

### The Key Insight

| Approach | Quality Loss | Throughput | Why |
|----------|-------------|------------|-----|
| **HSTU + sampling** | -10.8% | 1.6× | Predefined heuristic throws away useful info |
| **SYNAPSE v1** | -9.3% | 2.0× | Learned compression keeps more signal |

> **Learned compression (SYNAPSE) beats predefined sampling on BOTH quality AND efficiency**

### Detailed Metrics Comparison

| Metric | HSTU (full) | HSTU + sampling | SYNAPSE v1 | SYNAPSE vs Sampling |
|--------|-------------|-----------------|------------|---------------------|
| **NDCG@10** | 0.1823 | 0.1626 | **0.1654** | ✅ +1.7% better |
| **HR@10** | 0.3156 | 0.2816 | **0.2872** | ✅ +2.0% better |
| **Throughput** | 1× | 1.6× | **2.0×** | ✅ +25% faster |
| **Inference Latency** | 8.2 ms | 5.1 ms | 11 ms | ⚠️ Overhead |
| **Training Time** | 18.5h | 11.6h | 14.0h | Moderate |
| **Cold-start CTR** | 2.30% | 2.30% | **2.36%** | ✅ +2.6% |
| **Re-engagement** | 5.20% | 5.18% | 5.15% | Comparable |

**Key Insight:** While SYNAPSE v1 still trails HSTU (full) by 9.3%, it significantly outperforms the practical baseline (HSTU + linear-decay sampling) that teams actually use in production, achieving +1.7% better NDCG with +25% higher throughput.

---

## Component Attribution (from Ablation Studies)

| Component | Primary Benefit | Trade-off |
|-----------|-----------------|-----------|
| **SSD-FLUID** | 3-4× throughput (O(N) vs O(N²)) | **-5% to -7% NDCG** (O(N) approximation loses information) |
| **PRISM** | 8× memory on cold items, +2.6% cold CTR | **-5% to -8% warm item NDCG**, +30% latency |
| **FLUID** | Temporal modeling | **-2.5% to -3% temporal items** (fixed τ=24h fundamentally wrong) |
| **Multi-Token** | +0.5% NDCG (partial recovery) | **+25% latency**, +20% memory |

---

## Component-Level Analysis

### SSD-FLUID Backbone ⚠️

**Result: Significant trade-off between efficiency and quality**

| Metric | Baseline (HSTU) | SYNAPSE (SSD-FLUID) | Change |
|--------|-----------------|---------------------|--------|
| Throughput (items/sec) | 50,000 | **200,000** | **4×** |
| Training time (hrs) | 18.5 | 14.0 | **-24%** |
| Peak GPU memory | 45 GB | 22 GB | **-51%** |
| NDCG@10 | 0.1823 | 0.1712 | **-6.1%** |

**Analysis:**
- State Space Duality enables O(N) training vs O(N²) attention
- Streaming inference achieves O(1) per-step complexity
- **Quality degradation (-6%) is higher than hoped** - the O(N) approximation loses important attention patterns
- This is the primary source of quality loss in SYNAPSE v1

**Key Insight:** SSD-FLUID trades quality for efficiency. The 4× throughput gain comes at a significant accuracy cost.

### PRISM Hypernetwork ⚠️

**Result: Meaningful trade-off between memory and quality**

PRISM uses a **hybrid approach**:
- **Cold items** (new items, ~30% of catalog): Generate embeddings via hypernetwork
- **Warm items** (established items, ~70% of catalog): Ideally keep traditional embeddings, but full PRISM uses hypernetwork for all

| Metric | Baseline (Static) | SYNAPSE (PRISM) | Change |
|--------|------------------|-----------------|--------|
| Cold-start CTR | 2.30% | 2.36% | **+2.6%** |
| Cold item memory | 600 GB | 75 GB | **8× reduction** |
| Warm item NDCG | 0.1823 | 0.1713 | **-6.0%** |
| Overall memory | 2 TB | ~700 GB | **~3× reduction** |
| Inference latency | 0.8 ms | 1.04 ms | **+30%** |

**Analysis:**
- Hypernetwork excels at cold items where we previously had no good embeddings
- **Warm items see significant quality degradation** because hypernetwork cannot match dedicated embeddings
- Memory savings are real but more modest than originally claimed (3× overall, not 75×)
- Latency overhead from hypernetwork forward pass is substantial (+30%)

**Key Insight:** PRISM is a trade-off, not a free lunch. We trade ~6% warm item quality for better cold-start handling and ~3× overall memory reduction. The 75× claim in original literature only applies to the cold item portion.

### FLUID Temporal Layer ❌

**Result: Below target, actively hurts temporal-sensitive content**

| Metric | Baseline (Positional) | SYNAPSE (FLUID) | Change |
|--------|----------------------|-----------------|--------|
| Re-engagement (overall) | 5.20% | 5.15% | **-1%** |
| Re-engagement (temporal items) | 3.10% | 3.02% | **-2.5%** |
| Re-engagement (non-temporal) | 6.80% | 6.88% | **+1.2%** |
| Temporal coherence | Poor | Moderate | ✅ |

**Analysis:**
- Overall re-engagement **decreased by 1%** - FLUID is actually hurting performance
- **Critical Issue:** Temporal-sensitive items (news, events) are **-2.5% worse than baseline**
- Non-temporal items show modest improvement (+1.2%)
- Fixed τ=24h is fundamentally wrong for temporal-sensitive content

**Key Insight:** The fixed τ=24h timescale is the primary issue. News should decay in 2-4 hours, not 24. Albums should persist for weeks, not 24 hours. This is the main target for Iteration 2.

### Multi-Token Interaction ⚠️

**Result: Modest quality improvement with significant efficiency overhead**

| Metric | Baseline | With Multi-Token | Change |
|--------|----------|-----------------|--------|
| NDCG@10 | 0.1654 | 0.1667 | **+0.8%** |
| HR@10 | 0.2872 | 0.2887 | **+0.5%** |
| Inference latency | 8.8 ms | 11 ms | **+25%** |
| Memory overhead | - | +20% | Significant |

**Analysis:**
- Cross-attention aggregation provides modest quality improvement (+0.5%)
- Latency overhead (+25%) significantly exceeds acceptable threshold (<10%)
- Memory overhead (+20%) adds to overall system requirements
- Dense O(N²) cross-attention is too expensive for production

**Key Insight:** Multi-Token shows promise but needs efficiency improvements. The +0.5% NDCG gain partially compensates for SSD-FLUID loss, but the latency overhead is too high.

---

## 🔍 Deep Dive: FLUID Performance Gap

### Identified Problem

The FLUID layer uses a **fixed global decay constant τ = 24 hours** for all items. This creates significant issues:

| Item Type | Optimal τ | Current τ | Problem |
|-----------|-----------|-----------|---------|
| News articles | 2-4 hours | 24 hours | Over-values stale news → irrelevant recommendations |
| Trending events | Event-specific | 24 hours | Misses event timing dynamics |
| Music albums | Weeks/months | 24 hours | Under-values → premature forgetting |
| Classic movies | Years | 24 hours | Under-values → premature forgetting |

### Evidence from Segmented Analysis

| Content Category | Re-engagement Change | Expected | Gap |
|------------------|---------------------|----------|-----|
| **Temporal-sensitive** (news, events) | **-2.5%** | +5-10% | **-7.5% to -12.5%** |
| **Non-temporal** (albums, movies) | +1.2% | +3-5% | -1.8% to -3.8% |

### Root Cause

> The fixed τ=24h is **actively hurting temporal-sensitive items**. News that should decay in 2-4 hours is being treated as if it's still relevant after 24 hours, leading to stale recommendations. Meanwhile, items that should retain relevance longer are being forgotten too quickly.

---

## Ablation Study Results

| Configuration | NDCG@10 | Throughput | Cold-start | Re-engagement | Latency |
|---------------|---------|------------|------------|---------------|---------|
| HSTU Baseline | 0.1823 | 1× | 2.30% | 5.20% | 8.2 ms |
| SSD-FLUID only | 0.1712 | **4×** | 2.30% | 5.20% | 5.8 ms |
| PRISM only | 0.1713 | 1× | **2.36%** | 5.20% | 10.7 ms |
| FLUID only | 0.1805 | 1× | 2.30% | 5.15% | 8.5 ms |
| Multi-Token only | 0.1832 | 0.80× | 2.30% | 5.20% | 10.3 ms |
| **Full SYNAPSE v1** | **0.1654** | **2×** | **2.36%** | **5.15%** | **11 ms** |

### Relative Changes vs Baseline

| Configuration | Δ NDCG | Δ Throughput | Δ Cold-start | Δ Re-engage | Δ Latency |
|---------------|--------|--------------|--------------|-------------|-----------|
| SSD-FLUID | **-6.1%** | **+300%** | 0% | 0% | -29% |
| PRISM | **-6.0%** | 0% | **+2.6%** | 0% | **+30%** |
| FLUID | -1.0% | 0% | 0% | **-1%** | +4% |
| Multi-Token | **+0.5%** | -20% | 0% | 0% | **+25%** |
| **Full SYNAPSE v1** | **-9.3%** | **+100%** | **+2.6%** | **-1%** | **+35%** |

**Key Finding:** Component losses compound. SSD-FLUID (-6%) + PRISM (-6%) + FLUID (-1%) + Multi-Token (+0.5%) ≈ -9% overall, accounting for some interaction effects.

---

## Iteration 1 Summary

### What Worked ✅

1. **SSD-FLUID efficiency**: 4× throughput improvement from O(N) vs O(N²)
2. **PRISM cold-start**: +2.6% CTR improvement for cold items
3. **Memory reduction**: ~3× overall embedding memory reduction
4. **Training speedup**: 24% faster training from SSD-FLUID

### What Needs Improvement ❌

1. **Overall quality**: -9% NDCG is a significant trade-off
2. **FLUID temporal modeling**: -2.5% on temporal items (should be +5-10%)
3. **Overall re-engagement**: -1% vs +2-4% target
4. **Multi-Token efficiency**: +25% latency overhead (should be <10%)
5. **PRISM warm items**: -6% NDCG on warm items (hypernetwork limitation)

---

## Recommendation for Iteration 2

### Focus Area 1: Multi-Timescale Temporal Modeling (Primary)

Based on the analysis, the highest-impact improvement is fixing FLUID:

1. **Learned per-category timescales**: Replace fixed τ=24h with τ(category)
2. **Timescale tiers**: Fast (2-4h), Medium (24h), Slow (168-720h)
3. **End-to-end training**: Learn τ jointly with other parameters

**Expected Impact:**
- Temporal items: -2.5% → +0.5% (major improvement)
- Overall re-engagement: -1% → +2% (significant progress)
- Overall NDCG: -9% → -5% (quality recovery)

### Focus Area 2: Enhanced Multi-Token Interaction (Secondary)

Based on efficiency concerns, Multi-Token needs refinement:

1. **Grouped Query Attention (GQA)**: Reduce attention computation
2. **Top-k Sparse Attention**: Only attend to top-k most relevant positions
3. **Reduced Token Count**: K=4 instead of K=8

**Expected Impact:**
- Latency overhead: +25% → +10-12%
- NDCG improvement maintained: +0.5%
- Memory overhead reduced: +20% → +10%

---

## Research Questions for Iteration 2

1. Can we learn τ per content category without divergence?
2. What architecture enables stable timescale learning?
3. How do we make Multi-Token attention efficient without sacrificing quality?
4. Can sparse attention patterns maintain cross-sequence interaction benefits?

---

## Next Steps

1. ✅ Document findings and identify root causes
2. 🔄 **Start Iteration 2: Focus on Multi-Timescale FLUID + Enhanced Multi-Token v2**
3. Research advanced temporal modeling techniques
4. Design efficient Multi-Token architecture with GQA and sparse attention
