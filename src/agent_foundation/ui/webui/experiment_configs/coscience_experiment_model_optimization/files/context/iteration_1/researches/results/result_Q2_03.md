# PyTorch Inductor fusion transforms normalization kernels from 3 to 1

Switching from manual normalization to `F.normalize()` under `torch.compile()` will achieve **3→1 kernel fusion** per normalization operation, but normalization will **not fuse with adjacent Linear layers**. For recommendation models with hundreds of normalization operations, this provides significant launch overhead reduction. However, the `activation_memory_budget=0.05` setting does not break these fusions—it actually complements fusion by encouraging recomputation of cheap ops. Critical consideration: the default `eps=1e-12` is unsafe for mixed precision and should be explicitly set to `eps=1e-6`.

## F.normalize fuses internally but not with Linear layers

TorchInductor's fusion scheduler treats operations differently based on their type. **F.normalize() decomposes into three operations**—`torch.linalg.vector_norm` (reduction), `torch.clamp` (pointwise), and division (pointwise)—and these **will fuse into a single Triton kernel** because Inductor supports epilogue fusion of pointwise operations following reductions.

However, **normalization will not fuse with adjacent nn.Linear layers**. Linear operations dispatch to highly optimized external kernels (cuBLAS, CUTLASS, or Triton GEMM templates) treated as `ExternKernelSchedulerNode` in the fusion scheduler. These nodes cannot be fused with other operations because:

- External kernels are pre-optimized and self-contained
- Prologue fusion (operations before templates) is not supported
- The scheduler explicitly excludes ExternKernel nodes from fusion candidates

This means the pattern `x → Linear → F.normalize → Linear` produces **three separate kernel launches**: the first Linear, the fused normalize, and the second Linear. For your recommendation model, expect each normalization site to achieve 3→1 reduction independently, but boundaries at Linear layers remain.

The fusion scoring algorithm in `scheduler.py` uses: `score = memory_saved + read_count * shared_buffer_size`. Pointwise chains like `add → mul → relu` and reduction epilogues like `sum → div` achieve high scores and fuse automatically.

## Low memory budget enhances fusion rather than breaking it

The `activation_memory_budget=0.05` parameter does **not break up fused kernels**. It controls a different mechanism: the min-cut/max-flow partitioner that decides which activations to save versus recompute during the backward pass. A low budget (0.05, close to full recomputation) actually **enables more fusion** in the backward pass because:

- Recomputed pointwise ops in backward can fuse with gradient computations
- Fewer saved activations means fewer memory transfers between passes
- Fusion-friendly ops like normalization become candidates for recomputation

PyTorch's AOT Autograd benchmarks demonstrate this effect:
```
Eager,       Fwd = 740.77us, Bwd = 1560.52us
AOT,         Fwd = 713.85us, Bwd = 909.12us
AOT_Recomp,  Fwd = 712.22us, Bwd = 791.46us  ← Recomputation is faster
```

The key insight from the documentation: "We can recompute fusion-friendly operators to save memory, and then rely on the fusing compiler to fuse the recomputed operators. This reduces both memory and runtime."

For your workload, the **0.05 budget will work well** because normalization operations are lightweight compute (compared to matmuls) and ideal recomputation candidates. The forward pass fusion remains unchanged—only backward graph partitioning is affected.

| Budget Value | Behavior | Best For |
|--------------|----------|----------|
| 1.0 (default) | No extra recomputation | Speed-critical inference |
| 0.5 | Balanced tradeoff | Most training scenarios |
| 0.05-0.1 | Aggressive recomputation | Memory-constrained training |

## Kernel counts drop from 3 to 1 with compilation

Both manual normalization and `F.normalize()` produce **3 kernel launches in eager mode**—the operations are equivalent. Under `torch.compile()` with Inductor, both patterns fuse to **1 kernel**.

**Eager mode kernels (per normalization):**
```python
# Manual pattern - 3 kernels:
norm = torch.linalg.norm(x, dim=-1, keepdim=True)  # kernel 1: reduction
norm = torch.clamp(norm, min=eps)                   # kernel 2: pointwise
result = x / norm                                   # kernel 3: pointwise

# F.normalize - also 3 kernels internally:
# norm() → clamp_min(eps) → divide
```

**Compiled mode kernels (per normalization):**
Inductor generates a single Triton kernel pattern:
```python
@triton.jit
def triton_normalize(in_ptr, out_ptr, eps, ...):
    # Compute L2 norm (reduction)
    acc = tl.zeros(...)
    for roffset in range(0, rnumel, RBLOCK):
        x = tl.load(in_ptr + ...)
        acc += x * x
    norm = tl.math.sqrt(acc)
    norm = tl.maximum(norm, eps)  # clamp fused

    # Normalize (pointwise fused into same kernel)
    for roffset in range(0, rnumel, RBLOCK):
        x = tl.load(in_ptr + ...)
        tl.store(out_ptr + ..., x / norm)
```

For your model with hundreds of normalizations, switching all sites from manual patterns to `F.normalize()` under compilation provides **N × 2 fewer kernel launches** where N is your normalization count.

To verify fusion, use this profiling approach:
```python
import torch

# Enable debug output
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    result = compiled_model(x)
    torch.cuda.synchronize()

# Count distinct kernel types
cuda_events = [e for e in prof.key_averages()
               if e.device_type == torch.profiler.DeviceType.CUDA]
print(f"Kernel count: {len(cuda_events)}")
```

## Numerical precision requires explicit eps=1e-6 for stability

The most critical precision difference is the **epsilon handling**. `F.normalize()` defaults to `eps=1e-12`, which is **too small for float16/bfloat16** where it underflows to zero and produces NaN outputs. This is documented as PyTorch GitHub issue #32137.

**Recommended epsilon values by dtype:**

| Dtype | Safe eps Range | Recommended |
|-------|----------------|-------------|
| float32 | 1e-12 to 1e-8 | 1e-8 |
| float16/bfloat16 | 1e-6 to 1e-5 | **1e-6** |
| float64 | 1e-12 | 1e-12 |

For mixed-precision training, always explicitly specify `eps`:
```python
# Safe for mixed precision
result = F.normalize(x, p=2, dim=-1, eps=1e-6)
```

Your manual pattern `torch.clamp(norm, min=eps)` is mathematically equivalent to F.normalize's internal `clamp_min(eps)` when eps values match. The fusion under torch.compile preserves this equivalence, but test with representative edge cases:

```python
# Test numerical equivalence
from torch._dynamo.utils import same

eager_result = F.normalize(x, dim=-1, eps=1e-6)
compiled_fn = torch.compile(lambda x: F.normalize(x, dim=-1, eps=1e-6))
compiled_result = compiled_fn(x)

assert same(eager_result, compiled_result, tol=1e-4)
```

## Debugging tools for verifying fusion

Multiple tools exist to inspect and verify kernel fusion. The most effective approach combines environment variables with Python configuration:

**Quick verification with TORCH_COMPILE_DEBUG:**
```bash
TORCH_COMPILE_DEBUG=1 python your_model.py
```
This creates a debug directory containing `ir_pre_fusion.txt` and `ir_post_fusion.txt`—compare these to see exactly which operations were fused. Generated kernels appear in `output_code.py`.

**Modern trace analysis with tlparse:**
```bash
pip install tlparse
TORCH_TRACE="/tmp/tracedir" python your_model.py
tlparse /tmp/tracedir --output report.html
```

**Programmatic configuration for production debugging:**
```python
import torch._inductor.config as config
config.trace.enabled = True
config.trace.ir_pre_fusion = True
config.trace.ir_post_fusion = True
config.trace.output_code = True
```

**Pattern matcher counters for fusion verification:**
```python
from torch._dynamo.utils import counters
counters.clear()
output = compiled_model(input)
print(f"Fusions applied: {counters['inductor']['pattern_matcher_count']}")
```

The generated Triton kernel names follow a pattern that indicates fusion: `triton_poi_fused_*` for pointwise fused operations and `triton_red_fused_*` for reduction fused operations. Look for names like `triton_red_fused_clamp_div_norm` confirming your normalization operations fused.

## Practical recommendations for your recommendation model

Based on the research findings, here is the recommended approach for your large-scale recommendation model:

**Immediate changes:**
1. Replace all manual normalization patterns with `F.normalize(x, p=2, dim=-1, eps=1e-6)`
2. Keep `activation_memory_budget=0.05`—it complements fusion, doesn't break it
3. Use `torch.compile(model, mode="default")` rather than max-autotune for initial testing

**Validation checklist:**
```python
# 1. Verify numerical equivalence with eps=1e-6
def manual_normalize(x, eps=1e-6):
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm

x_test = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
manual_result = manual_normalize(x_test)
fn_result = F.normalize(x_test, p=2, dim=-1, eps=1e-6)
torch.testing.assert_close(manual_result, fn_result, rtol=1e-3, atol=1e-3)

# 2. Profile kernel reduction
compiled_model = torch.compile(model)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    compiled_model(sample_input)
prof.export_chrome_trace("fusion_trace.json")  # Visualize in chrome://tracing
```

**Expected impact:**
- Each normalization site: 3 kernels → 1 kernel
- With hundreds of normalizations: potential reduction of 200+ kernel launches per forward pass
- Combined with your previous 221→30 loss optimization: compounding efficiency gains

The Linear layer boundaries will remain as separate kernels, but the internal normalization fusion should significantly reduce overall launch overhead and improve memory bandwidth utilization—each fused kernel loads data once instead of three times.
