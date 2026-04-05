# SYNAPSE v2 Implementation Plan

## Overview

This document outlines the implementation plan for SYNAPSE v2, focusing on the
multi-timescale FLUID temporal layer to address the fixed τ=24h limitation
identified in Iteration 1.

## Change Summary

| Component | Change Type | Priority |
|-----------|-------------|----------|
| FLUID Temporal Layer | **Major Rewrite** | P0 |
| SSD-FLUID Backbone | No Change | - |
| PRISM Hypernetwork | No Change | - |
| Training Pipeline | Minor Updates | P1 |

## Implementation Phases

### Phase 1: Multi-Timescale Architecture (Week 1)

#### 1.1 Advanced FLUID Decay Module
**File:** `advanced_fluid_decay.py`

```python
class AdvancedFLUIDDecay(nn.Module):
    """Multi-timescale exponential decay with learnable parameters."""

    def __init__(self, num_timescales: int = 3):
        # Learnable log-timescales (in hours)
        self.log_tau = nn.Parameter(torch.tensor([
            math.log(3.0),    # Fast: news, viral
            math.log(24.0),   # Medium: movies, games
            math.log(168.0),  # Slow: albums, books
        ]))

    def forward(self, delta_t: Tensor, weights: Tensor) -> Tensor:
        tau = torch.exp(self.log_tau)  # [num_timescales]
        decay = torch.exp(-delta_t.unsqueeze(-1) / tau)  # [B, N, num_timescales]
        return (decay * weights).sum(-1)  # [B, N] - weighted decay
```

#### 1.2 Multi-Timescale Layer
**File:** `multi_timescale_layer.py`

```python
class MultiTimescaleFLUIDLayer(nn.Module):
    """FLUID v2: Content-aware multi-timescale temporal decay."""

    def __init__(self, hidden_dim: int = 256, num_timescales: int = 3):
        self.decay_module = AdvancedFLUIDDecay(num_timescales)
        self.timescale_predictor = TimescalePredictor(hidden_dim, num_timescales)

    def forward(self, h: Tensor, delta_t: Tensor, item_emb: Tensor) -> Tensor:
        # Predict timescale mixture weights from item embeddings
        weights = self.timescale_predictor(item_emb)  # [B, N, num_timescales]

        # Compute content-aware decay
        decay = self.decay_module(delta_t, weights)  # [B, N]

        return h * decay.unsqueeze(-1)  # [B, N, D]
```

#### 1.3 Timescale Predictor
**File:** `temporal_attention.py`

```python
class TimescalePredictor(nn.Module):
    """Predicts timescale mixture weights from item embeddings."""

    def __init__(self, hidden_dim: int, num_timescales: int, temperature: float = 1.0):
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_timescales)
        )
        self.temperature = temperature

    def forward(self, item_emb: Tensor) -> Tensor:
        logits = self.proj(item_emb) / self.temperature
        return F.softmax(logits, dim=-1)
```

### Phase 2: Training Stabilization (Week 2)

#### 2.1 Temperature Scheduling
```python
class TemperatureScheduler:
    """Cosine annealing from soft to hard selection."""

    def __init__(self, T_max: float = 5.0, T_min: float = 0.1, total_epochs: int = 100):
        self.T_max = T_max
        self.T_min = T_min
        self.total_epochs = total_epochs

    def get_temperature(self, epoch: int) -> float:
        progress = epoch / self.total_epochs
        return self.T_min + 0.5 * (self.T_max - self.T_min) * (1 + math.cos(math.pi * progress))
```

#### 2.2 Timescale Separation Loss
```python
def timescale_separation_loss(log_tau: Tensor, min_ratio: float = 4.0) -> Tensor:
    """Encourage timescales to be well-separated."""
    tau = torch.exp(log_tau)
    n = len(tau)
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ratio = tau[j] / tau[i] if tau[j] > tau[i] else tau[i] / tau[j]
            # Penalize if ratio < min_ratio
            loss += F.relu(min_ratio - ratio)
    return loss / (n * (n - 1) / 2)
```

#### 2.3 Gradient Clipping for Log-Tau
```python
def clip_log_tau_gradients(model: nn.Module, max_norm: float = 1.0):
    """Prevent explosive gradients on timescale parameters."""
    for name, param in model.named_parameters():
        if 'log_tau' in name and param.grad is not None:
            param.grad.data.clamp_(-max_norm, max_norm)
```

### Phase 3: Integration (Week 3)

#### 3.1 Updated SYNAPSE v2 Model
```python
class SYNAPSEv2(nn.Module):
    """SYNAPSE v2 with multi-timescale FLUID."""

    def __init__(self, config: SynapseConfig):
        self.backbone = SSDFLUIDBackbone(config)  # Unchanged from v1
        self.prism = PRISMHypernetwork(config)    # Unchanged from v1
        self.fluid = MultiTimescaleFLUIDLayer(    # NEW: Multi-timescale
            hidden_dim=config.hidden_dim,
            num_timescales=config.num_timescales
        )

    def forward(self, user_seq, item_emb, timestamps):
        # Backbone encoding
        h = self.backbone(user_seq)

        # Item-conditioned user representation
        h = self.prism(h, item_emb)

        # Multi-timescale temporal decay (NEW)
        delta_t = self.compute_time_deltas(timestamps)
        h = self.fluid(h, delta_t, item_emb)

        return self.predict(h, item_emb)
```

## Testing Strategy

### Unit Tests
- [ ] `test_advanced_fluid_decay.py`: Verify decay computation correctness
- [ ] `test_timescale_predictor.py`: Verify softmax output sums to 1
- [ ] `test_multi_timescale_layer.py`: End-to-end layer test

### Integration Tests
- [ ] `test_synapse_v2_forward.py`: Full forward pass
- [ ] `test_backward_stability.py`: Gradient flow verification
- [ ] `test_temperature_scheduling.py`: Annealing behavior

### Regression Tests
- [ ] Verify v1 behavior when all weight on medium timescale
- [ ] Ensure no performance regression on non-temporal items

## Rollout Plan

| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Architecture complete | All modules pass unit tests |
| 2 | Training stable | No NaN gradients, loss converges |
| 3 | Integration complete | Full pipeline runs end-to-end |
| 4 | Evaluation complete | Achieves +10% on temporal items |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training instability | Medium | High | Temperature scheduling + gradient clipping |
| Timescale collapse | Low | High | Separation regularization loss |
| Increased latency | Low | Medium | Profile and optimize softmax computation |
| Category overfitting | Medium | Medium | Hold-out category evaluation |

## Success Metrics

| Metric | Baseline (v1) | Target (v2) | Stretch |
|--------|---------------|-------------|---------|
| Temporal-sensitive re-engagement | +4% | +12% | +15% |
| Non-temporal re-engagement | +7% | +7% | +8% |
| Overall re-engagement | +6% | +10% | +12% |
| P99 latency | 12ms | <15ms | <12ms |
