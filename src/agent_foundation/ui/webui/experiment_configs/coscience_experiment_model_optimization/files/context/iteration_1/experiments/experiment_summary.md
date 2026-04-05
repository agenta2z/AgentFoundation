# CSML Optimization: Local Benchmark Summary

## Experiment Overview

**Objective**: Benchmark 5 optimization proposals for HSTU Transducer CInt
**Environment**: H100 GPU, batch size 512, 10 warmup + 100 timed iterations
**Primary Metric**: Isolated module latency (ms)

---

## Proposals Benchmarked

### 1. TransducerTune (Sync Elimination) - **Top Performer**
**Technique**: Eliminate CPU-GPU synchronization in transducer hot path

```python
# BEFORE: Multiple syncs per batch (15-20 × 50μs = 750-1000μs overhead)
for i in range(num_candidates.item()):  # SYNC
    if i < num_ro.item():               # SYNC
        process(data[i])

# AFTER: Pre-compute, sync-free hot path
info = NumCandidatesInfo(
    num_ro=num_ro.item(),    # Single sync
    num_nro=num_nro.item(),  # Single sync
)
for i in range(info.total):  # No sync
    process(data[i])
```

**Local Result**: **-43.1%** latency (1.74ms → 0.99ms)

---

### 2. TensorForge (Kernel Fusion)
**Technique**: Fuse QKV projections into single operation, reducing HBM round-trips

```python
# BEFORE: Separate projections - 5 HBM round-trips
q = self.q_proj(x)  # HBM write/read
k = self.k_proj(x)  # HBM write/read
v = self.v_proj(x)  # HBM write/read

# AFTER: Single fused projection - 2 HBM round-trips
qkv = self.qkv_proj(x)  # Single fused operation
q, k, v = qkv.chunk(3, dim=-1)
```

**Local Result**: **-22.6%** latency (12.4ms → 9.6ms)

---

### 3. FlashGuard (SDPA Backend)
**Technique**: Enable Flash Attention SDPA backend, preventing silent fallback to Math backend

```python
# Enforce CuDNN/Flash backend (not Math)
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
    attn_out = F.scaled_dot_product_attention(q, k, v)
```

**Local Result**: **-18.7%** latency (12.4ms → 10.1ms)

---

### 4. PrecisionPilot (Autocast Optimization)
**Technique**: Optimize mixed precision autocast scope, preventing float32 trap

```python
# Verify and cast to bf16 before SDPA
if q.dtype == torch.float32:
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

with torch.autocast('cuda', dtype=torch.bfloat16):
    logits = model.forward(embeddings)
```

**Local Result**: **-12.4%** latency (35.2ms → 30.8ms)

---

### 5. GradientSentry (Gradient Flow Fix)
**Technique**: Fix `@torch.no_grad()` decorator that breaks gradient flow

```python
# BEFORE (buggy): @torch.no_grad() wraps entire function
@torch.no_grad()
def get_embeddings(seq_embeddings, mask):
    return seq_embeddings[mask]  # requires_grad=False!

# AFTER (fixed): Only wrap index computation
def get_embeddings(seq_embeddings, mask):
    with torch.no_grad():
        mask = compute_mask(...)
    return seq_embeddings[mask]  # Gradients flow correctly
```

**Local Result**: **-2.8%** latency (4.82ms → 4.69ms)
**Note**: Primarily a quality fix (improves model convergence), not a throughput optimization.

---

## Summary Table

| Proposal | Module | Baseline (ms) | Optimized (ms) | Improvement |
|----------|--------|---------------|----------------|-------------|
| TransducerTune | Transducer Hot Path | 1.74 | 0.99 | **-43.1%** |
| TensorForge | MHA Forward | 12.4 | 9.6 | **-22.6%** |
| FlashGuard | Attention Layer | 12.4 | 10.1 | **-18.7%** |
| PrecisionPilot | Full Attention | 35.2 | 30.8 | **-12.4%** |
| GradientSentry | Embedding Module | 4.82 | 4.69 | **-2.8%** |

---

## Benchmark Methodology

### Hardware
- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: Intel Xeon Platinum 8480+
- **Memory**: 512GB DDR5

### Configuration
- Batch size: 512
- Sequence length: 256
- Embedding dimension: 128
- Number of heads: 8

### Timing
- Warmup: 10 iterations (excludes JIT compilation)
- Measured: 100 iterations
- Metric: Median latency with p10/p90 bounds
- Synchronization: `torch.cuda.synchronize()` before each timing

---

## Critical Notes

### ⚠️ Module-Level Benchmarks ≠ E2E Performance

These local benchmarks measure **isolated module latency** for each proposal's target component. They do NOT predict E2E training QPS because:

1. **Module fraction** - Each module is only 3-20% of total training step time
2. **JIT compilation** - Triton/PT2 compilation occurs at training start, not in warmup
3. **Production config** - Some optimizations (SDPA) may already be enabled
4. **Full backward pass** - Gradient checkpointing costs not fully captured
5. **Multi-GPU communication** - NCCL overhead not measured
6. **Dynamic shapes** - PT2 recompilation triggers in production

**Example**: TransducerTune shows -43% on its 1.74ms module, but this module is only ~3% of the full training step. Expected E2E impact: 43% × 3% ≈ 1.3% (actual MAST result: +1.23%).

**Recommendation**: Run full MAST experiments to validate E2E training QPS impact.

---

## Next Steps

1. **Submit to MAST** - Run full training experiments on H100 cluster
2. **Measure var_step_qps** - Primary E2E training efficiency metric
3. **Compare lifetime averages** - Use `variable_step_qps/global/lifetime/train`
4. **Check statistical significance** - Ensure p < 0.05 for claimed improvements

---

## Validation Requirements

Before production deployment:

- [ ] Run MAST experiment (2000+ training steps)
- [ ] Verify var_step_qps improvement
- [ ] Check p-value for statistical significance
- [ ] Confirm production config compatibility
- [ ] Test gradient flow for GradientSentry fix
