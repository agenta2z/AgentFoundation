# Proposal: Multi-Timescale Temporal Refinement for FLUID v2

## Executive Summary

Based on Iteration 1 analysis, we identified that the fixed τ=24h timescale in FLUID v1
is the primary bottleneck limiting performance on temporal-sensitive items (+4% vs +10-15%
target). This proposal introduces multi-timescale FLUID with learned per-category timescales.

## Problem Statement

### Current Limitation
FLUID v1 uses a single fixed timescale τ=24h for all items, which:
- Over-values stale news articles (should decay within 2-4 hours)
- Under-values evergreen content like music albums (should persist 168+ hours)
- Results in suboptimal temporal-sensitive item performance (+4% vs +10-15% target)

### Impact Analysis
| Content Category | Optimal τ | Current τ | Performance Gap |
|------------------|-----------|-----------|-----------------|
| Breaking News    | 2-4h      | 24h       | -8% (over-values stale content) |
| Movies           | 24-48h    | 24h       | ~0% (near optimal) |
| Music Albums     | 168h+     | 24h       | -4% (under-values evergreen) |
| Viral Content    | 6-12h     | 24h       | -5% (slow decay) |

## Proposed Solution

### 3-Tier Timescale System
We propose a learnable 3-tier timescale system:

1. **Fast Tier (τ_fast = 2-4h)**: News, trending topics, viral content
2. **Medium Tier (τ_medium = 24h)**: Movies, TV shows, games
3. **Slow Tier (τ_slow = 168h)**: Music albums, books, evergreen content

### Architecture Changes

```python
class MultiTimescaleFLUID(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_timescales: int = 3):
        super().__init__()
        # Learnable timescale parameters (in hours)
        self.log_tau = nn.Parameter(torch.tensor([
            math.log(3.0),    # Fast: ~3 hours
            math.log(24.0),   # Medium: ~24 hours
            math.log(168.0),  # Slow: ~168 hours (1 week)
        ]))

        # Category-to-timescale predictor
        self.timescale_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_timescales),
            nn.Softmax(dim=-1)
        )

    def compute_decay(self, h: Tensor, delta_t: Tensor, item_emb: Tensor) -> Tensor:
        # Get timescale mixture weights
        weights = self.timescale_predictor(item_emb)  # [B, 3]

        # Get learned timescales
        tau = torch.exp(self.log_tau)  # [3]

        # Compute mixed decay
        decay_factors = torch.exp(-delta_t.unsqueeze(-1) / tau)  # [B, N, 3]
        mixed_decay = (decay_factors * weights.unsqueeze(1)).sum(-1)  # [B, N]

        return h * mixed_decay.unsqueeze(-1)
```

### Training Stability Considerations

Based on research findings (result_timescale_learning.md), we will implement:

1. **Temperature Scheduling**: Soft mixture at start, hard selection at end
2. **Regularization**: Encourage timescale separation via orthogonality loss
3. **Gradient Clipping**: Prevent explosive gradients on log_tau parameters

## Expected Results

### Performance Targets
| Metric | FLUID v1 | FLUID v2 (Target) | Improvement |
|--------|----------|-------------------|-------------|
| Temporal-sensitive items | +4% | +14% | +10% (3.5×) |
| Non-temporal items | +7% | +7% | 0% (maintain) |
| Overall re-engagement | +6% | +10.5% | +4.5% |

### Why These Targets Are Achievable
1. **Timescale alignment**: Each content category gets appropriate decay rate
2. **Research validation**: Literature shows 2-3× improvements from learned timescales
3. **Ablation evidence**: Fixed τ=6h achieves +7% for fast-decaying items (vs +4% at τ=24h)

## Implementation Plan

### Phase 1: Architecture (Week 1)
- [ ] Implement MultiTimescaleFLUID module
- [ ] Add timescale predictor network
- [ ] Integrate with existing SYNAPSE backbone

### Phase 2: Training (Week 2)
- [ ] Temperature scheduling for stable training
- [ ] Timescale separation regularization
- [ ] Gradient clipping on log_tau

### Phase 3: Evaluation (Week 3)
- [ ] MovieLens-25M experiments
- [ ] Amazon cross-domain validation
- [ ] Per-category ablation analysis

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Training instability | Medium | Temperature scheduling + gradient clipping |
| Timescale collapse | Low | Separation regularization |
| Overfitting to categories | Medium | Hold-out category evaluation |

## Conclusion

Multi-timescale FLUID addresses the core limitation identified in Iteration 1 by allowing
the model to learn appropriate decay rates for different content categories. The expected
3.5× improvement on temporal-sensitive items would bring overall performance from +6% to
+10.5%, exceeding our target of +8-12%.
