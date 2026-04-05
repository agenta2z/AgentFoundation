# SYNAPSE v2 Experiment Setup

## Overview

This document describes the experimental setup for evaluating SYNAPSE v2 with
multi-timescale FLUID against the SYNAPSE v1 baseline.

## Hypothesis

**H1**: Multi-timescale FLUID will significantly improve re-engagement prediction
for temporal-sensitive items (+10% vs +4% in v1) while maintaining performance
on non-temporal items (+7% in v1).

**H2**: The overall re-engagement improvement will increase from +6% (v1) to
+10.5% (v2), exceeding our target range of +8-12%.

## Model Configurations

### SYNAPSE v1 (Baseline)
| Component | Configuration |
|-----------|---------------|
| SSD-FLUID Backbone | 6 layers, 256 hidden dim |
| PRISM Hypernetwork | 64-dim code encoder |
| FLUID Layer | **Fixed τ=24h** |
| Training | AdamW, lr=1e-4, 100 epochs |

### SYNAPSE v2 (Experimental)
| Component | Configuration |
|-----------|---------------|
| SSD-FLUID Backbone | 6 layers, 256 hidden dim (unchanged) |
| PRISM Hypernetwork | 64-dim code encoder (unchanged) |
| FLUID Layer | **Multi-timescale (3 learnable τ)** |
| Training | AdamW, lr=1e-4, 100 epochs |

### HSTU + Linear-Decay Sampling (Baseline)
| Component | Configuration |
|-----------|---------------|
| Architecture | Standard HSTU from generative-recommenders |
| Sequence Sampling | Linear-decay: weight(position) = 1 - α × (N - position) / N |
| Sequence Retention | ~40% of sequence (most recent + periodically sampled older) |
| Rationale | Common practical baseline that trades quality for efficiency |
| Performance | 1.6× throughput, -10.8% NDCG vs HSTU (full) |

> **Why include this baseline?** HSTU + linear-decay sampling is what practitioners actually use in production when HSTU (full) is too expensive. SYNAPSE v2 should beat this baseline to demonstrate that learned compression outperforms predefined sampling heuristics.

### Multi-Timescale Configuration
```python
MultiTimescaleConfig(
    num_timescales=3,
    init_timescales=(3.0, 24.0, 168.0),  # hours
    predictor_hidden=64,
    initial_temperature=5.0,
    final_temperature=0.1,
    separation_loss_weight=0.01,
    max_grad_norm=1.0,
)
```

## Datasets

### MovieLens-25M (Primary)
| Property | Value |
|----------|-------|
| Users | 162,541 |
| Items | 62,423 |
| Interactions | 25,000,095 |
| Temporal Coverage | 1995-2019 |
| Split | 80/10/10 (train/val/test) |

### Amazon Product Reviews (Validation)
| Domain | Items | Interactions |
|--------|-------|--------------|
| Movies & TV | 208,321 | 4,607,047 |
| Books | 929,264 | 22,507,155 |
| Electronics | 476,002 | 7,824,482 |

## Evaluation Metrics

### Primary Metrics
| Metric | Description |
|--------|-------------|
| Re-engagement@24h | User returns within 24 hours |
| Re-engagement@7d | User returns within 7 days |
| NDCG@10 | Ranking quality |
| Hit Rate@10 | At least one hit in top 10 |

### Temporal-Stratified Metrics
| Stratum | Description |
|---------|-------------|
| Fast-decay items | News, trending topics (expected τ < 12h) |
| Medium-decay items | Movies, games (expected τ ≈ 24h) |
| Slow-decay items | Albums, books (expected τ > 72h) |

## Training Configuration

### Optimizer
```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)
```

### Learning Rate Schedule
```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6,
)
```

### Temperature Schedule (v2 only)
```python
temp_scheduler = TemperatureScheduler(
    T_max=5.0,
    T_min=0.1,
    total_epochs=100,
    warmup_epochs=10,
)
```

### Loss Function
```python
# Main task loss
main_loss = BCEWithLogitsLoss(reduction='mean')

# Regularization (v2 only)
reg_loss = model.fluid.get_regularization_loss()

# Total loss
total_loss = main_loss + reg_loss
```

## Ablation Studies

### A1: Number of Timescales
| Config | Timescales |
|--------|------------|
| A1.1 | 2 (fast, slow) |
| A1.2 | 3 (fast, medium, slow) ← Default |
| A1.3 | 5 (very fine-grained) |

### A2: Temperature Schedule
| Config | Schedule |
|--------|----------|
| A2.1 | No annealing (T=1.0 constant) |
| A2.2 | Linear annealing |
| A2.3 | Cosine annealing ← Default |

### A3: Separation Regularization
| Config | Weight |
|--------|--------|
| A3.1 | 0 (no regularization) |
| A3.2 | 0.01 ← Default |
| A3.3 | 0.1 |

## Expected Results

Based on Iteration 1 analysis and research findings:

| Metric | v1 Baseline | v2 Expected | Improvement |
|--------|-------------|-------------|-------------|
| Temporal items | +4% | +14% | +10% (3.5×) |
| Non-temporal items | +7% | +7% | 0% |
| Overall | +6% | +10.5% | +4.5% |

## Compute Resources

| Resource | Allocation |
|----------|------------|
| GPUs | 8× A100 80GB |
| Training time | ~12 hours per run |
| Total experiments | 15 runs (main + ablations) |
| Total GPU hours | ~180 hours |

## Reproducibility

### Random Seeds
- Seed 1: 42 (primary)
- Seed 2: 2024
- Seed 3: 314159

### Checkpointing
- Save every 10 epochs
- Keep best 3 checkpoints by validation NDCG
- Final evaluation on held-out test set
