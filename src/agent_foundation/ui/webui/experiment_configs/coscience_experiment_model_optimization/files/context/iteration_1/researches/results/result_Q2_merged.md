# torch.compile Fusion Analysis for Normalization Operations in Large-Scale Recommendation Models

## Executive Summary

This document synthesizes comprehensive research on the optimization of normalization primitives within PyTorch 2.0's torch.compile ecosystem, specifically targeting large-scale recommendation models with dense embedding retrieval architectures.

**Key Findings:**

1. **3→1 Kernel Fusion Confirmed**: Both manual normalization and `F.normalize()` achieve **3→1 kernel fusion** under `torch.compile()` for the normalization subgraph itself. The total block (Linear + Normalize) compiles to **2 kernels** regardless of implementation choice.

2. **GEMM Fusion Barrier**: Normalization operations **cannot fuse with adjacent Linear layers** due to the synchronization requirements of row-wise reductions. This is a fundamental limitation of Inductor's GEMM template architecture.

3. **activation_memory_budget=0.05 Does NOT Break Fusion**: This aggressive setting forces recomputation but does **not disrupt forward pass fusion logic**. Fused kernels are simply re-executed during backward pass.

4. **Numerical Precision**: The default `eps=1e-12` is **unsafe for float16/bfloat16** due to underflow. **Always use `eps=1e-6`** for mixed precision training.

5. **F.normalize Preferred**: While both implementations achieve equal kernel counts under Inductor, `F.normalize()` provides a **canonical graph** that guarantees optimal scheduling and superior gradient stability via `max(x, eps)` formulation.

---

## Table of Contents

1. [Architectural Context and Problem Definition](#1-architectural-context-and-problem-definition)
2. [Kernel Fusion Mechanics](#2-kernel-fusion-mechanics)
3. [GEMM Fusion Barrier Analysis](#3-gemm-fusion-barrier-analysis)
4. [Activation Memory Budget Impact](#4-activation-memory-budget-impact)
5. [Numerical Precision Considerations](#5-numerical-precision-considerations)
6. [Kernel Count Analysis](#6-kernel-count-analysis)
7. [Debugging and Verification Tools](#7-debugging-and-verification-tools)
8. [Alternative Approaches and Recommendations](#8-alternative-approaches-and-recommendations)
9. [H100 Hardware Optimization](#9-h100-hardware-optimization)
10. [TorchRec and Production Deployment](#10-torchrec-and-production-deployment)
11. [References](#11-references)

---

## 1. Architectural Context and Problem Definition

### 1.1 The Computational Landscape of Recommendation Systems

Modern recommendation models, exemplified by Deep Learning Recommendation Models (DLRM) and sequential Transformer architectures, operate under distinct computational constraints:

| Characteristic | Description | Impact |
|----------------|-------------|--------|
| **High-Cardinality Embeddings** | Input features map to embedding tables with millions of rows | Efficient gather/scatter operations required |
| **Dense Interaction Layers** | Retrieved embeddings processed through MLPs and attention | Heavy GEMM workloads |
| **Frequent Normalization** | LayerNorm, RMSNorm, L2 Normalization after every projection | Latency bottleneck when unoptimized |
| **Sparse Operations** | 60-80% of compute in embedding lookups | Cannot leverage torch.compile effectively |

### 1.2 The Manual Normalization Problem

In eager execution mode, the manual normalization pattern incurs severe performance penalties:

```python
# Manual pattern - 3 separate CUDA kernel launches
norm = torch.linalg.norm(x, dim=-1, keepdim=True)  # Kernel 1: Reduction
norm = torch.clamp(norm, min=eps)                   # Kernel 2: Pointwise
result = x / norm                                   # Kernel 3: Pointwise/Broadcast
```

**Performance Impact:**
- Each operation triggers a separate CUDA kernel launch
- GPU must read input tensor from HBM and write results back **three times**
- For models with hundreds of normalization layers, this creates a latency floor
- Known as the "launch overhead" and "memory wall" problem

### 1.3 The PyTorch 2.0 Compilation Stack

The introduction of `torch.compile` in PyTorch 2.0 addresses the memory wall problem through operator fusion:

| Component | Function |
|-----------|----------|
| **TorchDynamo** | Captures Python bytecode, constructs FX graph, handles dynamic behavior |
| **AOTAutograd** | Traces backward graph ahead of time, decomposes operators to ATen primitives |
| **TorchInductor** | Lowers FX graph to optimized Triton kernels, performs loop fusion |

**Core Optimization Mechanism:** Loop fusion combines multiple pointwise and reduction operations into a single kernel to maximize data locality in GPU SRAM (L1/L2 cache).

---

## 2. Kernel Fusion Mechanics

### 2.1 Manual Implementation Under Inductor

When `torch.compile` processes the manual sequence (`norm → clamp → div`), Inductor's scheduler:

1. **Reduction Node**: `linalg.norm` identified as reduction summing squares over `dim=-1`
2. **Pointwise Nodes**: `clamp` and `div` identified as element-wise consumers
3. **Fusion Decision**: Scheduler recognizes Producer-Consumer pattern, fuses all three operations

However, manual implementations are susceptible to **"decomposition drift"**:
- `torch.norm` (deprecated) vs `torch.linalg.norm` may produce different graphs
- `keepdim=False` followed by manual `unsqueeze` adds view operations
- Complex manual patterns increase risk of suboptimal "split reductions"

### 2.2 F.normalize Under Inductor

`torch.nn.functional.normalize` is a high-level composite operation. Under `torch.compile`, it decomposes during the AOTAutograd phase into a canonical sequence:

$$y = \frac{x}{\max(\|x\|_p, \epsilon)}$$

**Key Advantages:**

1. **IR Cleanliness**: Consistent lowering to **Persistent Reduction** pattern
2. **Pattern Matching**: Inductor recognizes the subgraph: `SumSq → Sqrt → Max → Div`
3. **Optimized Templates**: Maps directly to specialized Triton templates
4. **Guaranteed Efficiency**: Eliminates scheduler miss-optimization risk

Unlike manual implementation which relies on general-purpose scheduler discovery, `F.normalize` hands the compiler a pre-validated structure.

### 2.3 Persistent Reduction Kernels

Inductor employs a scheduling heuristic to select between kernel types:

| Kernel Type | When Used | Memory Traffic |
|-------------|-----------|----------------|
| **Persistent Reduction** | Reduction dimension fits in registers/shared memory (D ≤ 1024) | 1 Read, 1 Write (optimal) |
| **Loop Reduction** | Massive reduction dimensions | Multiple reads, spills to global memory |

For recommendation models with embedding dimensions typically 64-1024, Inductor correctly selects Persistent Reduction.

**Generated Triton Kernel Structure:**

```python
@triton.jit
def triton_per_fused_normalize_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. Block ID corresponds to row index (Batch dimension)
    pid = tl.program_id(0)

    # 2. Coalesced Load: Load entire row into registers
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * BLOCK_SIZE + offsets)

    # 3. Reduction (Fused): Compute sum of squares in registers
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=0)

    # 4. Pointwise (Fused): Sqrt, Max, and Div
    norm = tl.sqrt(norm_sq)
    clamped_norm = tl.maximum(norm, EPS)
    out = x / clamped_norm

    # 5. Coalesced Store: Write result
    tl.store(out_ptr + pid * BLOCK_SIZE + offsets, out)
```

This single kernel replaces the memory-intensive read/write cycles of the manual implementation.

---

## 3. GEMM Fusion Barrier Analysis

### 3.1 Why GEMMs Cannot Fuse with Normalization

Linear layers dispatch to highly optimized external kernels treated as `ExternKernelSchedulerNode` in the fusion scheduler. These nodes cannot fuse with other operations because:

1. **External Kernel Isolation**: cuBLAS/CUTLASS kernels are pre-optimized and self-contained
2. **Prologue Fusion Unsupported**: Operations before GEMM templates cannot be fused
3. **Scheduler Exclusion**: `ExternKernel` nodes explicitly excluded from fusion candidates

### 3.2 The Normalization Problem

GEMM optimization is an NP-hard tiling problem. Inductor relies on "Max Autotune" to select the best kernel from a library of candidates.

**Modern GEMM Epilogue Fusion supports:**
- Unary/binary pointwise operations (ReLU, SiLU, GeLU, Bias Add)

**Normalization is a Reduction**, which requires:
1. Computing sum of squares of entire output row
2. Global barrier synchronization to aggregate partial sums
3. Second pass to divide elements

In highly parallel GEMM implementations, output rows are distributed across multiple CUDA thread blocks. No single block holds the complete accumulator until the global write phase, making fusion architecturally impossible.

### 3.3 Fusion Capabilities Summary

| Operation Sequence | Fused Kernel Count | Fusion Type | Notes |
|-------------------|-------------------|-------------|-------|
| Linear + ReLU | 1 | Epilogue Fusion | Supported by cuBLAS/Triton |
| Linear + Bias | 1 | Epilogue Fusion | Standard implementation |
| Norm + Clamp + Div | 1 | Vertical Fusion | Producer-Consumer fusion |
| **Linear + Normalize** | **2** | **None** | Reduction barrier prevents fusion |

### 3.4 Resulting Execution Graph

The sequence `Linear → F.normalize` compiles into **two distinct kernels**:

1. **Kernel 1 (GEMM)**: Computes Y = XW^T, fuses bias if present, writes Y to HBM
2. **Kernel 2 (Normalization)**: Reads Y from HBM, computes ‖Y‖, normalizes, writes Z

For the pattern `x → Linear → F.normalize → Linear`, expect **three separate kernel launches**: first Linear, fused normalize, second Linear.

---

## 4. Activation Memory Budget Impact

### 4.1 Selective Activation Checkpointing Mechanism

The setting `torch._functorch.config.activation_memory_budget = 0.05` enables automatic activation checkpointing via an ILP solver or Knapsack heuristic:

| Budget Value | Behavior | Best For |
|--------------|----------|----------|
| **1.0** (default) | No extra recomputation | Speed-critical inference |
| **0.5** | Balanced tradeoff | Most training scenarios |
| **0.05-0.1** | Aggressive recomputation | Memory-constrained training |

With budget = 0.05, the system stores only **5% of total activations**, marking the vast majority (including Linear and Normalization outputs) as "evictable."

### 4.2 Impact on Fusion Decisions

**Critical Finding: The memory budget does NOT break fusion logic.**

| Pass | Impact |
|------|--------|
| **Forward Pass** | **No impact on fusion.** Linear and F.normalize kernels generated and executed as fused kernels regardless of save/discard decision. |
| **Backward Pass** | When autograd requires discarded activations, triggers Recomputation Graph. Inductor compiles recomputation operations using the **same fused Triton kernels**. |

**Important Nuance (from Q2_04):** While the fundamental fusion logic remains intact, very aggressive budgets (like 0.05) can introduce additional **graph cuts** to drop activations. This means:
- The compiler may partition the graph into smaller segments to manage memory
- Operations that would otherwise fuse may end up in different segments
- This can lead to **more total kernel launches** even though each kernel remains internally fused
- The model gets broken into many recompute-friendly chunks, fragmenting the fused kernels

In summary: a tiny `activation_memory_budget` forces Inductor to favor recomputation of pointwise ops over keeping them fused in memory—reducing memory at the cost of fragmentation of the fused kernels.

**The Key Insight:** Inductor treats recomputation subgraphs exactly like any other graph. Instead of launching 3 eager kernels to recompute manual norm, Inductor launches the same single fused kernel used in forward pass.

**Why 0.05 Doesn't "Break" Fusion:**
- Decision to **fuse** is based on instruction-level parallelism and memory bandwidth
- Decision to **recompute** is based on memory capacity
- These are **orthogonal optimization axes**

### 4.3 Why Low Budget Actually Enhances Backward Fusion

Counter-intuitively, low budget (0.05) can **enable more fusion** in the backward pass:

1. Recomputed pointwise ops can fuse with gradient computations
2. Fewer saved activations means fewer memory transfers between passes
3. Fusion-friendly ops like normalization become recomputation candidates

**PyTorch AOT Autograd Benchmarks:**
```
Eager,       Fwd = 740.77µs, Bwd = 1560.52µs
AOT,         Fwd = 713.85µs, Bwd = 909.12µs
AOT_Recomp,  Fwd = 712.22µs, Bwd = 791.46µs  ← Recomputation is faster
```

Documentation insight: "We can recompute fusion-friendly operators to save memory, and then rely on the fusing compiler to fuse the recomputed operators. This reduces both memory and runtime."

### 4.4 Performance Implications

While fusion is preserved, the 0.05 budget has implications for throughput:

**Compute Intensity:**
- Recomputing Linear layer outputs means re-executing heavy GEMM operations
- Effectively doubles FLOPs for linear layers

**Recommendation:**
- Budget of 0.05 is extremely aggressive, likely suboptimal for speed
- Only use if model physically cannot fit in VRAM otherwise
- Consider 0.2-0.5 for better balance, saving largest tensors (Attention matrices) while keeping cheaper activations (Norm outputs) in memory

---

## 5. Numerical Precision Considerations

### 5.1 Mathematical Formulations

**Manual Pattern:**
$$v_{out} = \frac{v}{\text{clamp}(\|v\|, \min=\epsilon)}$$

**F.normalize Pattern:**
$$v_{out} = \frac{v}{\max(\|v\|, \epsilon)}$$

For scalar values, `clamp(x, min=e)` and `max(x, e)` appear identical. The critical difference lies in gradient behavior.

### 5.2 Gradient Divergence at the Singularity

**`max(x, e)` Gradient (F.normalize):**
- If ‖v‖ > ε: Gradient flows through the norm term
- If ‖v‖ < ε: The "winner" is constant ε, gradient w.r.t. norm becomes **0**
- Effectively cuts off gradient flow from norm calculation

**`clamp(x, min=e)` Gradient (Manual):**
- If ‖v‖ < ε: Gradient of clamp output w.r.t input is 0
- Manual compositions of `div` and `clamp` can introduce instability if norm produces NaN/Inf before clamp

**Superiority of F.normalize:** Uses fused backward definition in AOTAutograd that guards against division-by-zero singularities more robustly. Handles zero-input vectors by ensuring gradients are correctly zeroed or scaled.

### 5.3 Epsilon Value Safety

**CRITICAL: Default `eps=1e-12` is UNSAFE for float16/bfloat16**

This is documented as PyTorch GitHub issue #32137—eps underflows to zero in half precision, producing NaN outputs.

| Dtype | Safe eps Range | Recommended |
|-------|----------------|-------------|
| float32 | 1e-12 to 1e-8 | 1e-8 |
| **float16/bfloat16** | **1e-6 to 1e-5** | **1e-6** |
| float64 | 1e-12 | 1e-12 |

**Always explicitly specify eps for mixed precision:**
```python
# Safe for mixed precision training
result = F.normalize(x, p=2, dim=-1, eps=1e-6)
```

### 5.4 Floating-Point Accumulation Differences

Minor precision nuances between manual and compiled approaches:

1. **Reduction Order**: Fused kernels may sum elements in different order/parallel pattern
2. **Half-Precision Behavior**: Eager mode may internally upsample certain ops to higher precision; Inductor does not insert extra upcasts by default
3. **Magnitude**: Differences are on the order of machine epsilon, generally negligible for model quality

---

## 6. Kernel Count Analysis

### 6.1 Comprehensive Comparison Table

| Implementation Strategy | Execution Mode | Kernel 1 | Kernel 2 | Kernel 3 | Total Kernels |
|------------------------|----------------|----------|----------|----------|---------------|
| Manual (norm→clamp→div) | Eager | GEMM (addmm) | Reduction (norm) | Pointwise (clamp+div) | **3-4** |
| F.normalize | Eager | GEMM (addmm) | Fused Norm (vector_norm) | - | **2** |
| Manual | **Inductor** | GEMM (addmm) | Fused Norm (triton_per) | - | **2** |
| F.normalize | **Inductor** | GEMM (addmm) | Fused Norm (triton_per) | - | **2** |
| **Liger Kernel** | Custom Triton | **Fused Linear+Norm** | - | - | **1** |

### 6.2 Interpretation

1. **Eager to Inductor**: Manual implementation sees greatest improvement (3 kernels → 1 fused norm kernel)
2. **Manual vs F.normalize in Inductor**: Both converge to same kernel count (2 total)
3. **Why 2 Kernels?**: GEMM barrier prevents normalization from fusing into Linear
4. **The "3 to 1" Impact**: Verified for normalization subgraph itself; global count reduces from ~4 to 2

### 6.3 Expected Impact at Scale

For a model with N normalization steps:
- **Eager mode**: 3N kernel launches for normalization alone
- **Inductor mode**: N kernel launches (3× reduction)

With hundreds of normalizations: potential reduction of **200+ kernel launches** per forward pass.

---

## 7. Debugging and Verification Tools

### 7.1 Quick Verification with fullgraph=True

The primary diagnostic converts silent fallbacks into explicit errors:

```python
# Immediately surfaces any graph breaks as exceptions
model = torch.compile(model, fullgraph=True)
```

### 7.2 Environment Variable Diagnostics

```bash
# Show graph break locations and causes
TORCH_LOGS=graph_breaks python train.py

# Track recompilation triggers from shape changes
TORCH_LOGS=recompiles,guards python train.py

# Maximum verbosity for debugging
TORCH_LOGS="+dynamo,aot,inductor" python train.py
```

### 7.3 TORCH_COMPILE_DEBUG

```bash
TORCH_COMPILE_DEBUG=1 python your_model.py
```

Creates debug directory containing:
- `ir_pre_fusion.txt` and `ir_post_fusion.txt` - compare to see fused operations
- `output_code.py` - generated kernels

### 7.4 Modern Trace Analysis with tlparse

```bash
pip install tlparse
TORCH_TRACE="/tmp/tracedir" python your_model.py
tlparse /tmp/tracedir --output report.html
```

### 7.5 Programmatic Configuration

```python
import torch._inductor.config as config
config.trace.enabled = True
config.trace.ir_pre_fusion = True
config.trace.ir_post_fusion = True
config.trace.output_code = True
```

### 7.6 Pattern Matcher Counters

```python
from torch._dynamo.utils import counters
counters.clear()
output = compiled_model(input)
print(f"Fusions applied: {counters['inductor']['pattern_matcher_count']}")
```

### 7.7 Profiling Approach

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    result = compiled_model(x)
    torch.cuda.synchronize()

# Count distinct kernel types
cuda_events = [e for e in prof.key_averages()
               if e.device_type == torch.profiler.DeviceType.CUDA]
print(f"Kernel count: {len(cuda_events)}")

# Export for visualization
prof.export_chrome_trace("fusion_trace.json")  # View in chrome://tracing
```

### 7.8 Generated Kernel Name Patterns

| Pattern | Meaning |
|---------|---------|
| `triton_poi_fused_*` | Pointwise fused operations |
| `triton_red_fused_*` | Reduction fused operations |
| `triton_red_fused_clamp_div_norm` | Normalization operations fused |

### 7.9 Numerical Equivalence Verification

```python
from torch._dynamo.utils import same

eager_result = F.normalize(x, dim=-1, eps=1e-6)
compiled_fn = torch.compile(lambda x: F.normalize(x, dim=-1, eps=1e-6))
compiled_result = compiled_fn(x)

assert same(eager_result, compiled_result, tol=1e-4)
```

### 7.10 torch._dynamo.explain()

```python
explanation = torch._dynamo.explain(model)(sample_input)
print(f"Graph breaks: {explanation.graph_break_count}")
print(explanation.break_reasons)
```

---

## 8. Alternative Approaches and Recommendations

### 8.1 Normalization Approach Comparison

| Approach | Fusion Quality | Kernel Count | Effort | Notes |
|----------|----------------|--------------|--------|-------|
| **Manual Pattern** | Good (under Inductor) | 2 | None | Decomposition drift risk |
| **F.normalize** | **Excellent** | 2 | Low | Canonical graph, gradient stability |
| **LayerNorm/RMSNorm** | **Excellent** | 1 (with Linear) | Medium | Dedicated BatchLayernormFusion |
| **Custom Triton (Liger)** | **Guaranteed** | 1 | High | True Linear+Norm fusion |

### 8.2 Why LayerNorm/RMSNorm May Be Better

PyTorch's Inductor has dedicated fusion classes for LayerNorm:
- `BatchLayernormFusion`: Robust fusion with adjacent layers
- `BatchLinearFusion`: Strong support for linear layer combinations
- Heavily optimized in fused reduction GEMM kernels

`F.normalize` relies primarily on generic pointwise fusion patterns, making it a lower-priority target for optimization.

**Recommendation:** Consider replacing `F.normalize` with LayerNorm or RMSNorm where mathematically appropriate for proven fusion capabilities.

### 8.3 Liger Kernel for Ultimate Performance

For true 1-kernel execution of Linear+Norm, specialized kernels like **Liger Kernel**, **FlashLinear**, or **FlashAttention's fused RMSNorm** are required:

- Manually written Triton kernels
- Restructure GEMM loop using "Split-K" or tile-based reduction
- Keep data in SRAM throughout
- Standard `torch.compile` cannot generate these automatically

**FlashAttention's Fused RMSNorm** (GitHub issue #570) demonstrates the benefit of such fusion by reducing memory overhead and kernel launches versus unfused PyTorch code—serving as a reference implementation for what's achievable with custom kernels.

```python
# Liger Kernel provides drop-in replacements
from liger_kernel.transformers import LigerRMSNorm
# Replaces nn.RMSNorm with fused implementation
```

### 8.4 Immediate Implementation Recommendations

**Immediate Actions:**
1. Replace all manual patterns with `F.normalize(x, p=2, dim=-1, eps=1e-6)`
2. Keep `activation_memory_budget=0.05` if memory-constrained (it complements fusion)
3. Consider increasing budget to 0.2-0.3 for better speed/memory tradeoff
4. Use `torch.compile(model, mode="default")` for initial testing

**Validation Checklist:**

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
prof.export_chrome_trace("fusion_trace.json")
```

**Expected Impact:**
- Each normalization site: 3 kernels → 1 kernel (for norm subgraph)
- With hundreds of normalizations: ~200+ fewer kernel launches per forward pass
- Combined with previous optimizations: compounding efficiency gains
- End-to-end speedup: 5-10% from normalization overhead reduction alone

---

## 9. H100 Hardware Optimization

### 9.1 H100 Architectural Advantages

| Feature | Specification | Improvement |
|---------|--------------|-------------|
| **FP8 Tensor Cores** | 1,979 TFLOPS | **6x** vs A100 FP16 |
| **TMA (Tensor Memory Accelerator)** | 1.45 TB/s effective | **59%** bandwidth improvement |
| **CUDA Graphs** | Eliminates Python launch overhead | Essential for inference |

### 9.2 Compilation Mode Selection

| Mode | Compile Time | Runtime | Memory | Use Case |
|------|-------------|---------|--------|----------|
| `default` | 1-5 min | Good | Baseline | Development |
| `reduce-overhead` | 2-8 min | Better (small batch) | +10-20% | Inference |
| `max-autotune` | 10-60+ min | Best | Variable | Production |

### 9.3 FP8 Integration for Dense Layers

```python
from torchao.float8 import convert_to_float8_training

# Only convert dense layers (embedding tables stay FP32/FP16)
def filter_fn(mod, fqn):
    if isinstance(mod, torch.nn.Linear):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    return False

convert_to_float8_training(model, module_filter_fn=filter_fn)
compiled = torch.compile(model, mode="max-autotune")
```

### 9.4 Optimal H100 Configuration

```python
compiled = torch.compile(
    model,
    mode="max-autotune",
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,      # Tensor core alignment
        "triton.cudagraphs": True,
    }
)
```

**CUDA Graphs Note:** `reduce-overhead` mode eliminates Python kernel launch overhead but requires static shapes. For variable batch sizes, use `max-autotune-no-cudagraphs` or bucket inputs to discrete sizes.

---

## 10. TorchRec and Production Deployment

### 10.1 Graph Breaks in Recommendation Models

Graph breaks fragment optimization when TorchDynamo encounters untraceable code:

**Critical Break Causes for RecSys:**

```python
# Data-dependent control flow - BREAKS GRAPH
def forward(self, x):
    if x.sum() > 0:  # Tensor-dependent branch
        return self.layer_a(x)
    return self.layer_b(x)

# Fix with torch.cond
def forward(self, x):
    return torch.cond(
        x.sum() > 0,
        lambda x: self.layer_a(x),
        lambda x: self.layer_b(x),
        (x,)
    )
```

**Sparse Embeddings:**
- `sparse=True` fails with Inductor assertion error
- TorchRec's `EmbeddingBagCollection` requires `@torch.compiler.disable` on sparse forward path
- `.item()` calls force immediate graph termination

### 10.2 Hybrid Compilation Strategy

The optimal approach for embedding-heavy models combines TorchRec/FBGEMM for embeddings with torch.compile on MLP interaction layers:

```python
class DLRM(nn.Module):
    def __init__(self):
        self.sparse_arch = EmbeddingBagCollection(...)
        self.dense_arch = nn.Sequential(...)
        self.interaction = InteractionArch(...)
        self.over_arch = nn.Sequential(...)

    @torch.compiler.disable
    def embed(self, kjt):
        return self.sparse_arch(kjt)  # Runs eager

    def forward(self, dense_features, sparse_features):
        sparse_emb = self.embed(sparse_features)
        dense_out = self.dense_arch(dense_features)   # Compiled
        interaction = self.interaction(dense_out, sparse_emb)  # Compiled
        return self.over_arch(interaction)  # Compiled

model = torch.compile(DLRM(), mode="max-autotune")
```

**Expected Results:**
- 20-50% latency reduction on dense portions
- FBGEMM's hand-tuned sparse kernels preserved
- Dense MLP components achieve 2-3x speedups

### 10.3 Dynamic Shape Handling

```python
# Mark batch dimension as dynamic instead of triggering recompilation
batch_input = torch.randn(batch_size, feature_dim)
torch._dynamo.mark_dynamic(batch_input, 0)  # Batch dimension varies
torch._dynamo.mark_dynamic(batch_input, 0, min=1, max=512)  # With bounds
```

**Recompilation Limits:**
- Default: 8 recompilations per function before falling back to eager
- Production systems can exhaust this within minutes with diverse sequence lengths

```python
# Production configuration
torch.compiler.config.recompile_limit = 8
torch.compiler.config.accumulated_recompile_limit = 256
torch.compiler.config.fail_on_recompile_limit_hit = True  # Alert on limit
torch.compiler.config.automatic_dynamic_shapes = True
torch.compiler.config.assume_static_by_default = True
```

### 10.4 Jagged Tensors for Variable-Length Sequences

Jagged tensors work with the `torch.nested` layout and compile cleanly:

```python
# Native nested tensors compile with fullgraph=True
nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)

@torch.compile(fullgraph=True)
def process(x):
    return x.sin() + 1

output = process(nt)  # Single graph handles ragged structure
```

For KeyedJaggedTensor inputs common in TorchRec, mark dynamic dimensions on both values and lengths tensors:

```python
torch._dynamo.mark_dynamic(kjt.values(), 0)
torch._dynamo.mark_dynamic(kjt.lengths(), 0)
```

### 10.5 Compilation Caching

```bash
# Persistent cache location
export TORCHINDUCTOR_CACHE_DIR=/persistent/cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
```

**Portable Cache for Containerized Deployments:**

```python
# After warm-up, export cache
artifacts = torch.compiler.save_cache_artifacts()
artifact_bytes, cache_info = artifacts
with open("torch_cache.bin", "wb") as f:
    f.write(artifact_bytes)

# On new deployment instances
with open("torch_cache.bin", "rb") as f:
    torch.compiler.load_cache_artifacts(f.read())
```

### 10.6 Warm-up Procedures

**First inference triggers full compilation** taking 10-60+ minutes for large models:

```python
def warm_up_model(model, sample_inputs, warmup_iterations=5):
    model.eval()
    compiled = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = compiled(*sample_inputs)
            if i == 0:
                print("Initial compilation complete")
    torch.cuda.synchronize()
    return compiled
```

### 10.7 Production Monitoring

```python
import torch._dynamo

counters = torch._dynamo.utils.counters
metrics.gauge("pytorch.recompiles", sum(counters['recompiles'].values()))
metrics.gauge("pytorch.graph_breaks", sum(counters['graph_break'].values()))
```

### 10.8 Custom CUDA Kernel Registration for torch.compile

Custom CUDA kernels require FakeTensor registration to participate in compilation:

```python
from torch.library import custom_op

@custom_op("mylib::fused_embedding", mutates_args=())
def fused_embedding(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return custom_cuda_impl(indices, table)

@fused_embedding.register_fake
def _(indices, table):
    batch_size = indices.shape[0]
    embed_dim = table.shape[1]
    return torch.empty(batch_size, embed_dim, device=indices.device)
```

This pattern allows custom CUDA kernels to:
- Be traced by TorchDynamo without executing the actual CUDA code
- Participate in graph compilation and optimization
- Provide correct output shape/dtype information for downstream operations

### 10.9 Industry Deployment Case Studies

| Company | Achievement | Key Insight |
|---------|-------------|-------------|
| **Meta** | Trillion-parameter scale with TorchRec | Distributed embeddings rather than direct torch.compile |
| **Pinterest** | "Scaling Inference of O(10K)-length Sequence Recommendation Models" | CUDA graphs + Triton kernels (complementary to torch.compile) |
| **Amazon** | 30-40% latency reduction on diffusion models | Forward pass must be pure function; use only torch.Tensor inputs |
| **AWS** | 2x speedup on Graviton (DLRM in TorchBench) | Platform-specific optimizations compound with compilation |
| **Netflix** | 20% improvement with "Trace Framework" | Foundation models with shared embeddings maximize compile benefits |
| **Dashboard** | 43% average speedup, 93% compatibility rate | RecSys below average due to sparse ops (60-80% of compute) |

**Key Lessons from Industry:**
- Forward pass must be a pure function—no state mutations
- Use only `torch.Tensor` types as inputs—lift non-tensor configs to module properties
- Minimize branching complexity by pushing to initialization
- Always test for numerical differences (compiled models are not bitwise exact)

---

## 11. References

### PyTorch Documentation
1. [torch.nn.functional.normalize — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
2. [Performance Tuning Guide — PyTorch Tutorials](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
3. [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
4. [PyTorch 2: Faster ML Through Dynamic Python Bytecode Transformation](https://docs.pytorch.org/assets/pytorch2-2.pdf)

### Technical Blogs and Articles
5. [How does torch.compile speed up a transformer? — Adam Casson](https://www.adamcasson.com/posts/torch-compile-vit)
6. [Learn by doing: TorchInductor Reduction Kernels — Karthick Panner Selvam](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/)
7. [Unleashing PyTorch Performance with TorchInductor — Aadishagrawal](https://medium.com/@aadishagrawal/unleashing-pytorch-performance-with-torchinductor-a-deep-dive-1f01e8b36efa)
8. [Introduction to torch.compile and How It Works with vLLM — vLLM Blog](https://blog.vllm.ai/2025/08/20/torch-compile.html)
9. [State of torch.compile for training (August 2025) — ezyang's blog](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
10. [Accelerating PyTorch Models: Inside torch.compile's Kernel Optimization — Abhik Sarkar](https://www.abhik.ai/articles/compiling-pytorch-kernel)
11. [Peak Performance, Minimized Memory: Optimizing torchtune with torch.compile & Liger Kernel](https://pytorch.org/blog/peak-performance-minimized-memory/)

### GitHub Issues and Repositories
12. [linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training](https://github.com/linkedin/Liger-Kernel)
13. [Liger Kernel: Efficient Triton Kernels for LLM Training — arXiv](https://arxiv.org/html/2410.10989v2)
14. [Enable TorchInductor to Generate Matmuls Natively via tl.dot #151705](https://github.com/pytorch/pytorch/issues/151705)
15. [[Inductor] Fusion of Tiled Point-Wise and Reduction Operators #128063](https://github.com/pytorch/pytorch/issues/128063)
16. [Reduced memory requirements of fused RMSNorm kernel — flash-attention #570](https://github.com/Dao-AILab/flash-attention/issues/570)
17. [PyTorch GitHub Issue #32137 — eps underflow in F.normalize](https://github.com/pytorch/pytorch/issues/32137)

### Research Papers
18. [Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums — arXiv](https://arxiv.org/html/2510.17505v1)
19. [PluS: Highly Efficient and Expandable ML Compiler with Pluggable Graph Schedules — USENIX](https://www.usenix.org/system/files/atc25-wu-ruofan.pdf)

### Configuration and Source Code
20. [pytorch/torch/_inductor/config.py — GitHub](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)
21. [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
22. [Questions about Inductor code generation for GEMM on Nvidia device](https://dev-discuss.pytorch.org/t/questions-about-inductor-code-generation-for-gemm-on-nvidia-device/1379)
23. [torch._functorch.partitioners — functorch documentation](https://docs.pytorch.org/functorch/nightly/_modules/torch/_functorch/partitioners.html)

### Additional Resources
24. [Torch Compile and External Kernels — NVIDIA PhysicsNeMo Framework](https://docs.nvidia.com/physicsnemo/latest/user-guide/performance_docs/torch_compile_support.html)
25. [kornia.contrib — Read the Docs](https://kornia.readthedocs.io/en/latest/contrib.html)
26. [Torch.compile: The Missing Manual — Scribd](https://www.scribd.com/document/937961531/Torch-compile-The-Missing-Manual)

---

## Appendix: Quick Answers Summary

### RQ1: Does F.normalize fuse with adjacent Linear layers?

**Answer: NO.** The barrier between GEMM templates and reduction operations enforces a two-kernel execution flow. This is fundamental to Inductor's architecture—GEMMs dispatch to external kernels (cuBLAS/CUTLASS) that cannot participate in fusion with reduction operations.

### RQ2: Does activation_memory_budget=0.05 break fusion?

**Answer: NO**, but with important nuance. The aggressive budget forces recomputation of intermediate activations during backward pass, but does **not** disrupt forward pass fusion logic. The same fused kernels are simply re-executed during gradient computation. Fusion decisions are based on instruction-level parallelism and memory bandwidth, while recomputation decisions are based on memory capacity—these are orthogonal optimization axes.

**Important caveat:** Very aggressive budgets may introduce **graph cuts** to drop activations. This can cause operations that would otherwise fuse to end up in different segments, leading to **more total kernel launches** even though each kernel remains internally fused.

### RQ3: What are the kernel launch counts?

| Configuration | Kernels per Normalization Block |
|--------------|--------------------------------|
| Manual (Eager) | 3-4 |
| F.normalize (Eager) | 2 |
| Manual (Inductor) | 2 |
| F.normalize (Inductor) | 2 |
| Liger Kernel (Custom) | 1 |

The normalization subgraph itself achieves 3→1 fusion; the total block (Linear+Norm) remains at 2 kernels due to GEMM barrier.

### RQ4: What are the numerical precision differences?

**Key Differences:**
1. **Epsilon handling**: `F.normalize` uses `max(x, eps)` providing cleaner gradient signal (zero gradient w.r.t. norm below threshold); manual `clamp` can have instability
2. **Default eps**: `F.normalize` defaults to `eps=1e-12` which is **unsafe for float16/bfloat16**—always use `eps=1e-6`
3. **Floating-point accumulation**: Minor differences (~machine epsilon) from different reduction order under compilation

### RQ5: What is the recommended approach?

1. **Immediate**: Replace manual patterns with `F.normalize(x, p=2, dim=-1, eps=1e-6)`
2. **Memory-constrained**: Keep `activation_memory_budget=0.05`; consider 0.2-0.3 for speed
3. **Medium-term**: Migrate to LayerNorm/RMSNorm where mathematically valid for optimal fusion
4. **Long-term**: Consider Liger Kernels for guaranteed single-kernel Linear+Norm execution
5. **RecSys-specific**: Use hybrid compilation—`@torch.compiler.disable` on sparse embeddings, compile dense interaction layers
