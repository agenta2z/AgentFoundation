# Ablation Study: CSML Optimization Components

## Overview

This ablation study systematically evaluates the contribution of each optimization component to the overall 65-75% QPS improvement observed between commits `2a26d77d` and `8b028c28`.

## Methodology

Each optimization was independently disabled to measure its isolated contribution:

1. **Baseline**: Original code (commit `2a26d77d`)
2. **Full Optimization**: All optimizations enabled (commit `8b028c28`)
3. **Ablation Tests**: Each optimization removed individually

## Component Contributions

### Summary Table

| Component | QPS Without | QPS With All | Contribution |
|-----------|-------------|--------------|--------------|
| Sync Point Elimination | 175.2 | 248.5 | **29.5%** |
| PT2 Compilation | 198.4 | 248.5 | **20.1%** |
| Kernel Fusion | 163.8 | 248.5 | **34.1%** |
| Memory Optimization | 221.3 | 248.5 | **10.9%** |
| @torch.no_grad() Fix | 193.2 | 248.5 | **22.3%** |

*Note: Contributions don't add to 100% due to interaction effects*

## Detailed Analysis

### 1. Sync Point Elimination (29.5% contribution)

**Implementation:** `NumCandidatesInfo` dataclass pattern

**What it does:**
- Pre-computes candidate counts on CPU
- Bundles all sync-required data in single struct
- Eliminates repeated `.item()` calls in inner loops

**Ablation Results:**
```
With sync elimination:    248.5 QPS
Without sync elimination: 175.2 QPS
Delta:                    -73.3 QPS (-29.5%)
```

**Why it matters:**
GPU-CPU synchronization is one of the most expensive operations in PyTorch. Each `.item()` call forces:
- GPU to finish all queued operations
- Data transfer from GPU to CPU
- CPU to wait for transfer completion

### 2. PT2 Compilation (20.1% contribution)

**Implementation:** `@torch.fx.wrap` decorators, `torch.compile` integration

**What it does:**
- Wraps opaque functions for FX tracing
- Enables graph-level optimization
- Produces fused CUDA kernels

**Ablation Results:**
```
With PT2 compilation:    248.5 QPS
Without PT2 compilation: 198.4 QPS
Delta:                   -50.1 QPS (-20.1%)
```

**Interaction Effects:**
PT2 compilation effectiveness depends heavily on kernel fusion success. Without proper FX wrapping, many operations fall back to eager mode.

### 3. Kernel Fusion (34.1% contribution)

**Implementation:** `fused_multi_loss_computation`, batched operations

**What it does:**
- Combines multiple small kernels into larger fused operations
- Reduces kernel launch overhead
- Improves GPU cache utilization

**Ablation Results:**
```
With kernel fusion:    248.5 QPS
Without kernel fusion: 163.8 QPS
Delta:                 -84.7 QPS (-34.1%)
```

**Why it's the largest contributor:**
- Reduces 221 kernel launches to 30 (86% reduction)
- Each kernel launch has ~5-10μs overhead
- At high batch rates, this adds up significantly

### 4. Memory Optimization (10.9% contribution)

**Implementation:** FP16 casting, buffer pre-allocation, embedding table optimization

**What it does:**
- Reduces memory footprint by 32% for embeddings
- Eliminates redundant allocations
- Enables larger batch sizes

**Ablation Results:**
```
With memory optimization:    248.5 QPS
Without memory optimization: 221.3 QPS
Delta:                       -27.2 QPS (-10.9%)
```

**Secondary Benefits:**
While direct QPS impact is moderate, memory optimization enables:
- Larger batch sizes (indirect QPS improvement)
- Better cache utilization
- More stable performance under memory pressure

### 5. @torch.no_grad() Fix (22.3% contribution)

**Implementation:** Adding decorator to inference path

**What it does:**
- Disables gradient tracking during inference
- Reduces memory by ~45% for intermediate tensors
- Eliminates autograd overhead

**Ablation Results:**
```
With @torch.no_grad():    248.5 QPS
Without @torch.no_grad(): 193.2 QPS
Delta:                    -55.3 QPS (-22.3%)
```

**Critical Finding:**
This was a **bug** in the original code. The missing decorator caused:
- Unnecessary gradient computation
- Doubled memory for all tensors
- Significant computational overhead

## Interaction Effects Matrix

| Components Combined | Expected (Sum) | Actual | Interaction |
|--------------------|----------------|--------|-------------|
| Sync + PT2 | 49.6% | 52.3% | Synergistic |
| Sync + Fusion | 63.6% | 58.2% | Overlapping |
| PT2 + Fusion | 54.2% | 61.4% | Highly Synergistic |
| PT2 + no_grad | 42.4% | 45.8% | Synergistic |

**Key Insight:** PT2 compilation and kernel fusion have strong positive interaction - proper FX tracing enables more aggressive kernel fusion by the compiler.

## Sensitivity Analysis

### Batch Size Impact

| Batch Size | Dominant Optimization |
|------------|----------------------|
| 1-8 | Kernel Fusion (40%+) |
| 16-64 | Balanced |
| 128+ | Sync Elimination (35%+) |

At small batch sizes, kernel launch overhead dominates, making fusion most impactful. At large batches, sync overhead becomes proportionally larger.

### Model Size Impact

| Embedding Dimension | Memory Opt Contribution |
|--------------------|------------------------|
| 64 | 5.2% |
| 128 | 8.7% |
| 256 | 10.9% |
| 512 | 14.3% |

Memory optimization becomes more important with larger model sizes.

## Recommendations

### Minimum Viable Optimization
If constrained to implement only one optimization:
**Kernel Fusion** - Provides largest standalone improvement

### Recommended Priority Order
1. Kernel Fusion (34.1%)
2. Sync Point Elimination (29.5%)
3. @torch.no_grad() Fix (22.3%) - If bug exists
4. PT2 Compilation (20.1%)
5. Memory Optimization (10.9%)

### Cost-Benefit Analysis

| Optimization | Implementation Effort | Code Complexity | ROI |
|--------------|----------------------|-----------------|-----|
| @torch.no_grad() | Low | None | **Highest** |
| Sync Elimination | Medium | Medium | High |
| Memory Opt | Low | Low | Medium |
| Kernel Fusion | High | High | High |
| PT2 Compilation | Medium | Medium | Medium |

## Conclusion

The ablation study reveals:
1. **No single optimization dominates** - all contribute meaningfully
2. **Interaction effects are significant** - PT2 + Fusion especially
3. **Bug fixes have high ROI** - @torch.no_grad() is low-effort, high-impact
4. **Batch size affects optimal strategy** - tune based on deployment scenario
