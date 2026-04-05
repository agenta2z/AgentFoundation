# Iteration 1 Metrics Summary

## Overview

This document provides a comprehensive summary of SYNAPSE v1 metrics with honest assessment of trade-offs.

**Key Finding:** SYNAPSE v1's learned compression **beats predefined sampling (HSTU + linear-decay) on BOTH quality AND efficiency**, demonstrating that intelligent compression outperforms heuristic approaches.

---

## Full System Performance Comparison

### Primary Comparison: Learned Compression vs Predefined Sampling

| Model | NDCG@10 | Δ NDCG | Throughput | Status |
|-------|---------|--------|------------|--------|
| **HSTU (full)** | 0.1823 | baseline | 1× | Gold standard (expensive) |
| **HSTU + linear-decay** | 0.1626 | **-10.8%** | **1.6×** | Predefined sampling baseline |
| **SYNAPSE v1** | 0.1654 | **-9.3%** | **2.0×** | ✅ **Beats sampling on BOTH** |

### Why This Comparison Matters

- **HSTU (full)**: Gold standard but computationally expensive (O(N²) attention)
- **HSTU + linear-decay sampling**: What practitioners actually use - sample ~40% of sequence with recency bias
- **SYNAPSE v1**: Learned compression that intelligently decides what to keep

> **Learned compression beats predefined sampling by +1.5% NDCG while being 25% faster**

### Detailed Metrics

| Metric | HSTU (full) | HSTU + sampling | SYNAPSE v1 | SYNAPSE vs Sampling |
|--------|-------------|-----------------|------------|---------------------|
| **NDCG@10** | 0.1823 | 0.1626 | **0.1654** | ✅ +1.7% better |
| **HR@10** | 0.3156 | 0.2816 | **0.2872** | ✅ +2.0% better |
| **Throughput** | 1× | 1.6× | **2.0×** | ✅ +25% faster |
| **Training Time** | 18.5h | 11.6h | **14.0h** | Moderate |
| **Cold-start CTR** | 2.30% | 2.30% | **2.36%** | ✅ +2.6% |
| **Re-engagement** | 5.20% | 5.18% | **5.15%** | Comparable |

**Key Insight:** While SYNAPSE v1 trails HSTU (full) by 9.3%, it outperforms the practical baseline that teams actually deploy. Predefined sampling loses important interaction patterns; learned compression preserves more signal.

---

## Component Breakdown

### SSD-FLUID Performance ⚠️

**Result: Major trade-off between efficiency and quality**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSD-FLUID Results                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Throughput Improvement:                                        │
│  ──────────────────────────────────────────────────             │
│  Achieved:  ████████████████ 4×                                 │
│  Target:    ████████████████████████ 10×+ (aspirational)        │
│  ✅ Meaningful improvement from O(N) vs O(N²)                   │
│                                                                 │
│  Training Time Reduction:                                       │
│  ──────────────────────────────────────────────────             │
│  Achieved:  ████████████████ 24%                                │
│  Target:    ████████████████████████ 30-50%                     │
│  ⚠️ Close to target                                             │
│                                                                 │
│  NDCG Quality Impact:                                           │
│  ──────────────────────────────────────────────────             │
│  MovieLens: -6.1% (significant trade-off for efficiency)        │
│  Amazon:    -5.8% (consistent pattern)                          │
│  ❌ MAJOR QUALITY TRADE-OFF - O(N) approximation loses info     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** SSD-FLUID provides 4× throughput but at a significant -6% NDCG cost. The O(N) approximation loses important attention patterns that are critical for recommendation quality.

### PRISM Performance ⚠️

**Result: Memory savings with quality trade-off**

PRISM uses a **hybrid approach**:
- **Cold items** (~30% of catalog): Generate embeddings via hypernetwork
- **Warm items** (~70% of catalog): Hypernetwork struggles vs dedicated embeddings

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRISM Results                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Cold-Start CTR Improvement:                                    │
│  ──────────────────────────────────────────────────             │
│  Achieved:  ██ +2.6%                                            │
│  ✅ Hypernetwork helps cold items                               │
│                                                                 │
│  Memory Reduction:                                              │
│  ──────────────────────────────────────────────────             │
│  Cold items: 8× reduction (600 GB → 75 GB)                      │
│  Overall:    ~3× reduction (2 TB → ~700 GB)                     │
│  ⚠️ Realistic savings (not 75× as originally claimed)          │
│                                                                 │
│  Warm Item Quality Impact:                                      │
│  ──────────────────────────────────────────────────             │
│  NDCG: -6.0% (hypernetwork cannot match dedicated embeddings)   │
│  ❌ SIGNIFICANT QUALITY LOSS on warm items                      │
│                                                                 │
│  Latency Overhead:                                              │
│  ──────────────────────────────────────────────────             │
│  +30% (hypernetwork forward pass is expensive)                  │
│  ⚠️ Partially offsets SSD-FLUID throughput gains                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** PRISM is a trade-off, not a free lunch. We trade ~6% warm item quality for better cold-start handling and ~3× overall memory reduction. The 75× claim in original literature only applies to the cold item portion under ideal conditions.

### FLUID Performance ❌

**Result: Below target, actively hurts temporal-sensitive content**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUID Results                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Re-engagement by Content Type:                                 │
│  ──────────────────────────────────────────────────             │
│                                                                 │
│  Temporal-sensitive items (news, events):                       │
│  ████████ -2.5%        Target: ████████████████ +5-10%          │
│  ❌ NEGATIVE IMPACT - CRITICAL ISSUE                            │
│                                                                 │
│  Non-temporal items (movies, albums):                           │
│  ██ +1.0%              Target: ████████ +3-5%                   │
│  ⚠️ BELOW TARGET                                                │
│                                                                 │
│  Overall re-engagement:                                         │
│  ████ -1%              Target: ████████████████ +2-4%           │
│  ❌ NEGATIVE vs BASELINE                                        │
│                                                                 │
│  ROOT CAUSE: Fixed τ=24h ACTIVELY HURTS temporal content        │
│  • News should decay in 2-4 hours, not 24h (over-valued)        │
│  • Albums should persist for weeks, not 24h (under-valued)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** FLUID's fixed τ=24h is fundamentally wrong. It actively hurts temporal-sensitive items (-2.5%) and provides only modest improvement for non-temporal items (+1.0%). This is the PRIMARY TARGET for Iteration 2.

### Multi-Token Performance ⚠️

**Result: Modest quality improvement with significant efficiency overhead**

```
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Token Interaction Results                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NDCG Improvement (Cross-Attention Benefit):                    │
│  ──────────────────────────────────────────────────             │
│  MovieLens: █ +0.5%                                             │
│  Amazon:    █ +0.5%                                             │
│  ✅ MODEST QUALITY RECOVERY (partially compensates SSD loss)    │
│                                                                 │
│  Latency Overhead:                                              │
│  ──────────────────────────────────────────────────             │
│  ██████████████████████████ +25%   Target: ██████████ <10%      │
│  ❌ SIGNIFICANTLY EXCEEDS BUDGET                                │
│                                                                 │
│  Memory Overhead:                                               │
│  ──────────────────────────────────────────────────             │
│  +20% additional GPU memory required                            │
│  ⚠️ SIGNIFICANT OVERHEAD                                        │
│                                                                 │
│  ROOT CAUSE: Dense O(N²) cross-attention is too expensive       │
│  • Need GQA (Grouped Query Attention) for efficiency            │
│  • Need sparse attention patterns                               │
│  • Need reduced token count (K=4 instead of K=8)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Multi-Token provides modest quality recovery (+0.5%) that partially compensates for SSD-FLUID losses, but the +25% latency overhead is unacceptable for production. This is a SECONDARY TARGET for Iteration 2.

---

## Iteration 1 Gap Analysis

### What's Working (Relative Successes)

| Component | Metric | Achievement | Notes |
|-----------|--------|-------------|-------|
| SSD-FLUID | Throughput | 4× | Meaningful efficiency gain |
| SSD-FLUID | GPU memory | -51% | Significant reduction |
| PRISM | Cold-start CTR | +2.6% | Hypernetwork helps new items |
| PRISM | Memory | ~3× reduction | Real savings, not 75× |
| Multi-Token | NDCG quality | +0.5% | Partial SSD loss recovery |

### What Needs Improvement (Critical Issues)

| Component | Metric | Gap | Root Cause |
|-----------|--------|-----|------------|
| **Overall** | **NDCG** | **-9%** | **Compounded approximation losses** |
| SSD-FLUID | NDCG | -6% | O(N) approximation loses information |
| PRISM | Warm NDCG | -6% | Hypernetwork can't match dedicated embeddings |
| **FLUID** | **Temporal items** | **-2.5%** | **Fixed τ=24h fundamentally wrong** |
| FLUID | Overall re-engagement | -1% | Temporal items drag down average |
| Multi-Token | Latency | +25% > +10% target | Dense attention too expensive |
| Multi-Token | Memory | +20% overhead | No token pruning |

---

## Recommendation: Iteration 2 Focus

### Primary Focus: Multi-Timescale FLUID

Based on gap analysis, the highest-impact improvement is fixing FLUID:

**Replace fixed τ=24h with learned per-category timescales**

Expected impact:
```
┌─────────────────────────────────────────────────────────────────┐
│           Projected Iteration 2 Improvement (FLUID)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Temporal-sensitive items (news, events):                       │
│  Iter 1: ████████ -2.5%                                         │
│  Iter 2: ██ +0.5% (projected)                                   │
│  Improvement: +3% (direction change!)                           │
│                                                                 │
│  Non-temporal items (movies, albums):                           │
│  Iter 1: ██ +1.0%                                               │
│  Iter 2: ███ +1.5% (projected)                                  │
│                                                                 │
│  Overall re-engagement:                                         │
│  Iter 1: ████ -1%                                               │
│  Iter 2: ████████ +2% (projected)                               │
│                                                                 │
│  Overall NDCG:                                                  │
│  Iter 1: -9%                                                    │
│  Iter 2: -5% (projected, ~4% recovery)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Secondary Focus: Enhanced Multi-Token v2

**Replace dense cross-attention with efficient sparse attention**

Expected impact:
```
┌─────────────────────────────────────────────────────────────────┐
│        Projected Iteration 2 Improvement (Multi-Token)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NDCG Improvement:                                              │
│  Iter 1: █ +0.5%                                                │
│  Iter 2: █ +0.6% (maintained with better efficiency)            │
│                                                                 │
│  Latency Overhead:                                              │
│  Iter 1: ██████████████████████████ +25%                        │
│  Iter 2: ████████████ +12% (GQA + sparse attention)             │
│  Target: ██████████ <10% (close!)                               │
│                                                                 │
│  Memory Overhead:                                               │
│  Iter 1: +20%                                                   │
│  Iter 2: +10% (projected)                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Honest Assessment

### The Trade-off Story

SYNAPSE v1 demonstrates that aggressive efficiency optimizations come with real quality costs:

| Optimization | Benefit | Cost |
|--------------|---------|------|
| SSD-FLUID (O(N) vs O(N²)) | 4× throughput | -6% NDCG |
| PRISM (hypernetwork) | 3× memory, +2.6% cold CTR | -6% warm NDCG, +30% latency |
| FLUID (temporal) | Temporal modeling capability | -2.5% temporal items (wrong τ) |
| Multi-Token (cross-attention) | +0.5% NDCG recovery | +25% latency, +20% memory |
| **Combined** | **2× throughput, 3× memory** | **-9% NDCG, +35% latency** |

### When SYNAPSE v1 Makes Sense

SYNAPSE v1's trade-offs may be acceptable for:
- **High-throughput applications** where ranking precision is less critical
- **Memory-constrained environments** where 3× memory reduction is valuable
- **Cold-start heavy catalogs** where +2.6% cold item improvement matters

### When SYNAPSE v1 Does NOT Make Sense

SYNAPSE v1's trade-offs are likely unacceptable for:
- **Precision-critical ranking** where -9% NDCG is too costly
- **Latency-sensitive applications** where +35% latency is unacceptable
- **Temporal content platforms** where -2.5% temporal item performance is harmful

### Path Forward

Iteration 2 should focus on recovering quality while maintaining efficiency:
1. **Multi-Timescale FLUID**: Fix temporal items (-2.5% → +0.5%)
2. **Enhanced Multi-Token v2**: Reduce latency (+25% → +12%)
3. **Target**: -5% NDCG with 2.3× throughput (meaningful improvement)
