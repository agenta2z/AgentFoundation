# SYNAPSE Iteration 1 Experiment Setup

## Overview

This document describes the experimental setup for validating the SYNAPSE architecture (Iteration 1) against baseline sequential recommendation systems.

---

## Datasets

### Primary Evaluation: MovieLens-25M

| Property | Value |
|----------|-------|
| Users | 162,541 |
| Items | 62,423 |
| Interactions | 25,000,095 |
| Sparsity | 99.75% |
| Avg sequence length | 153.8 |
| Timestamp range | 1995-2019 |

**Preprocessing:**
- Filter users with < 5 interactions
- Filter items with < 5 interactions
- Chronological train/val/test split: 80/10/10
- Normalize timestamps to seconds since first interaction

### Cross-Domain Validation: Amazon Product Reviews

| Property | Value |
|----------|-------|
| Categories | Electronics, Books |
| Users | 192,403 |
| Items | 63,001 |
| Interactions | 1,689,188 |
| Avg sequence length | 8.8 |

---

## Models Evaluated

### 1. HSTU (Full Sequence)
- Standard HSTU architecture from generative-recommenders
- O(N²) self-attention over full sequence
- Static item embeddings
- Positional encoding for temporal modeling
- **Gold standard for quality, but expensive (1× throughput)**

### 2. HSTU + Linear-Decay Sampling (Baseline)
- HSTU architecture with predefined sequence sampling
- Linear-decay sampling: weight(position) = 1 - α × (N - position) / N
- Retains ~40% of sequence (most recent + periodically sampled older)
- Common practical baseline that trades quality for efficiency
- **1.6× throughput, but loses important interaction patterns**

### 3. SYNAPSE v1 (Full System)
All three components integrated:
- **SSD-FLUID**: O(N) training, O(1) inference
- **PRISM**: User-conditioned embeddings
- **FLUID**: Fixed τ=24h temporal decay

### 3. Ablation Variants
- **SSD-FLUID Only**: SSD backbone + static embeddings + positional encoding
- **PRISM Only**: HSTU backbone + PRISM embeddings + positional encoding
- **FLUID Only**: HSTU backbone + static embeddings + FLUID temporal

---

## Training Configuration

```yaml
# Common settings
batch_size: 256
learning_rate: 1e-4
weight_decay: 1e-5
warmup_steps: 1000
max_epochs: 100
early_stopping_patience: 10

# SSD-FLUID specific
ssd_fluid:
  d_model: 512
  n_layers: 6
  ssm_state_size: 16
  training_mode: linear_attention

# PRISM specific
prism:
  item_code_dim: 64
  generator_hidden: 512
  user_context_dim: 256

# FLUID specific
fluid:
  base_timescale: 86400  # 24 hours (fixed in Iteration 1)
  learnable_timescale: false
  decay_type: exponential
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Items processed per second | 10-100× over baseline |
| **Cold-start CTR** | Click-through for new items | +15-25% |
| **Re-engagement** | Returning user engagement | +8-12% |
| **NDCG@10** | Normalized discounted cumulative gain | Improve over baseline |
| **HR@10** | Hit rate at 10 | Improve over baseline |

### Segmented Analysis

To understand temporal model behavior, we segment items by temporal sensitivity:

| Category | Definition | Example Items |
|----------|------------|---------------|
| **Temporal-sensitive** | Relevance decays within hours | News, events, trending topics |
| **Non-temporal** | Relevance stable over weeks/months | Movies, albums, books |

---

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | 8× NVIDIA A100 80GB |
| CPU | 128-core AMD EPYC |
| Memory | 1TB RAM |
| Storage | NVMe SSD |

---

## Experiment Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Data prep | 1 day | Preprocessing and caching |
| Baseline training | 2 days | HSTU baseline |
| SYNAPSE training | 2 days | Full system |
| Ablation studies | 3 days | Component isolation |
| Analysis | 2 days | Metric aggregation and insights |

---

## Success Criteria

### Per-Component Targets

| Component | Metric | Target | Priority |
|-----------|--------|--------|----------|
| SSD-FLUID | Throughput | 10-100× | High |
| SSD-FLUID | Training time | -30-50% | Medium |
| PRISM | Cold-start CTR | +15-25% | High |
| PRISM | Memory reduction | 300-400× | Medium |
| FLUID | Re-engagement | +8-12% | High |
| FLUID | Temporal coherence | Handle gaps | Medium |

### Overall System Target

- Improve NDCG@10 by at least 5% over HSTU baseline
- Maintain or improve latency for real-time serving
