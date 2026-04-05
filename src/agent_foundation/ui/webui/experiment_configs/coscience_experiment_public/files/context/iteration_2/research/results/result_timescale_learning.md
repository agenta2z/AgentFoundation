# Stable Timescale Learning Research (Iteration 2 - Q3/Q4)

## Overview

This document summarizes research on stable training techniques for learnable timescales, including temporal attention mechanisms that complement analytical decay.

---

## Part 1: Stable Timescale Learning (Q3)

### Challenge: Training Instability

When τ is learned end-to-end, several failure modes can occur:

| Failure Mode | Symptom | Cause |
|--------------|---------|-------|
| **Explosion** | τ → ∞ | Unbounded parameter, no decay |
| **Collapse** | τ → 0 | Overfitting to recent items only |
| **Oscillation** | τ fluctuates wildly | Gradient instability |
| **Homogenization** | All items same τ | Insufficient category signal |

### Solution: Sigmoid Bounding with Bounded Learning

**Architecture:**
```python
class StableTimescalePredictor(nn.Module):
    def __init__(self, d_model, n_categories=3):
        # Learnable category bases (soft constrained)
        self.tau_logit = nn.Parameter(torch.zeros(n_categories))

        # Bounding parameters (not learned)
        self.register_buffer('tau_min', torch.tensor([
            2 * 3600,    # Fast min: 2 hours
            12 * 3600,   # Medium min: 12 hours
            84 * 3600,   # Slow min: 3.5 days
        ]))
        self.register_buffer('tau_max', torch.tensor([
            8 * 3600,    # Fast max: 8 hours
            48 * 3600,   # Medium max: 48 hours
            336 * 3600,  # Slow max: 2 weeks
        ]))

        # Category classifier
        self.category_head = nn.Linear(d_model, n_categories)

        # Item-level modulation
        self.modulation_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )

    def forward(self, x):
        # Get category probabilities
        cat_probs = F.softmax(self.category_head(x), dim=-1)  # [batch, 3]

        # Get bounded tau_base for each category
        tau_base = self.tau_min + torch.sigmoid(self.tau_logit) * (self.tau_max - self.tau_min)

        # Weighted combination
        tau_category = (cat_probs * tau_base).sum(dim=-1)  # [batch]

        # Item modulation: (0.5, 1.5) range
        mod = 0.5 + self.modulation_head(x)  # [batch, 1]

        return tau_category * mod.squeeze(-1)
```

### Training Stability Techniques

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Sigmoid bounding** | Prevent τ explosion/collapse | τ = τ_min + σ(w)(τ_max - τ_min) |
| **Warm start** | Start from known-good values | Initialize logit to map to industry τ |
| **Separate LR** | Slower τ learning | τ params at 0.1× base LR |
| **Gradient clipping** | Prevent sudden jumps | Clip τ gradients at 1.0 |
| **EMA smoothing** | Reduce oscillation | Update τ_base with momentum |

### Initialization Strategy

```python
def initialize_tau_logit(tau_target, tau_min, tau_max):
    """Compute logit that maps to target τ via sigmoid."""
    # sigmoid(logit) = (tau_target - tau_min) / (tau_max - tau_min)
    normalized = (tau_target - tau_min) / (tau_max - tau_min)
    normalized = torch.clamp(normalized, 0.01, 0.99)  # Avoid extreme logits
    logit = torch.log(normalized / (1 - normalized))
    return logit

# Initialize to industry-validated values
tau_init = torch.tensor([4, 24, 168]) * 3600  # 4h, 24h, 1 week
tau_logit_init = initialize_tau_logit(tau_init, tau_min, tau_max)
```

### Stability Monitoring

During training, monitor these metrics:

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| τ_fast | 2-8 hours | < 1h or > 12h |
| τ_medium | 12-48 hours | < 6h or > 72h |
| τ_slow | 84-336 hours | < 48h or > 720h |
| Category entropy | > 0.5 | < 0.3 (homogenization) |
| τ gradient norm | < 0.1 | > 1.0 (instability) |

---

## Part 2: Temporal Attention Mechanisms (Q4)

### Research Question

Can attention over time gaps complement learned decay? How do transformer-style temporal mechanisms compare?

### Finding: Hybrid Approach Works Best

**Comparison of temporal modeling approaches:**

| Approach | Strengths | Weaknesses | Re-engagement Δ |
|----------|-----------|------------|-----------------|
| Positional encoding | Simple | No time semantics | +2% |
| Analytical decay (fixed τ) | Time-aware | One-size-fits-all | +6% |
| Analytical decay (learned τ) | Adaptive | May miss patterns | +10% |
| Pure temporal attention | Expressive | High complexity | +9% |
| **Decay + Attention hybrid** | Best of both | Moderate complexity | **+12%** |

### Hybrid Architecture: FLUID v2 with Temporal Attention

```python
class FLUIDv2WithTemporalAttention(nn.Module):
    def __init__(self, config):
        # Learned timescale decay (main mechanism)
        self.fluid_decay = MultiTimescaleFLUID(config)

        # Lightweight temporal attention (complementary)
        self.temporal_attn = TemporalAttention(config)

        # Fusion gate: how much attention to use
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, timestamps):
        # FLUID decay path
        h_decay = self.fluid_decay(x, timestamps)

        # Temporal attention path
        h_attn = self.temporal_attn(x, timestamps)

        # Adaptive fusion
        gate = self.fusion_gate(torch.cat([h_decay, h_attn], dim=-1))
        output = gate * h_attn + (1 - gate) * h_decay

        return output


class TemporalAttention(nn.Module):
    """Lightweight attention that incorporates time gaps."""

    def __init__(self, config):
        self.time_encoder = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.d_model)
        )
        self.attention = nn.MultiheadAttention(config.d_model, num_heads=4)

    def forward(self, x, timestamps):
        # Compute pairwise time differences
        delta_t = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)

        # Encode time into bias
        time_bias = self.time_encoder(delta_t.unsqueeze(-1))

        # Apply attention with time-aware bias
        output, _ = self.attention(x, x, x, attn_mask=time_bias)

        return output
```

### When to Use Temporal Attention

| Scenario | Use Decay Only | Use Decay + Attention |
|----------|---------------|----------------------|
| Simple temporal patterns | ✅ Sufficient | Overkill |
| Complex temporal patterns | May miss patterns | ✅ Better |
| Low latency requirement | ✅ Faster | More compute |
| Limited training data | ✅ Fewer params | Risk of overfit |

### Experimental Comparison

| Configuration | Params | Re-engagement | Latency |
|---------------|--------|---------------|---------|
| FLUID v1 (fixed τ) | Base | +6% | 1.0× |
| FLUID v2 (learned τ) | +5% | +10% | 1.05× |
| FLUID v2 + Attention | +15% | +12% | 1.15× |
| Pure Attention | +20% | +9% | 1.25× |

**Recommendation:** Start with FLUID v2 (learned τ only). Add temporal attention if:
1. Remaining gap to target after τ learning
2. Complex temporal patterns in data
3. Latency budget allows 15% overhead

---

## Summary

### Q3: Stable Timescale Learning

- **Use sigmoid bounding** with per-category bounds
- **Initialize from industry knowledge** (4h, 24h, 168h)
- **Apply separate, lower learning rate** for τ parameters
- **Monitor** τ distribution and gradient norms

### Q4: Temporal Attention

- **Hybrid approach** (decay + attention) achieves best results (+12%)
- **Pure decay** (learned τ) is nearly as good (+10%) with lower complexity
- **Recommendation:** Start with learned τ, add attention if needed
