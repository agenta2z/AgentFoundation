# Memory Bandwidth Optimization and Kernel Efficiency for Large-Scale DLRMs

**Executive Summary**: For Deep Learning Recommendation Models (DLRMs) processing **256,000-sample batches**, memory bandwidth—not compute capacity—becomes the dominant performance constraint. Both explicit `.expand()` and implicit broadcasting are semantically equivalent zero-copy operations that use stride-0 semantics; TorchInductor normalizes both patterns into identical optimized Triton kernels. However, the risk of **accidental materialization** (through `.contiguous()`, device transfers, or reshape operations) can instantly consume **500+ MB per tensor** and degrade performance by 33%. The most impactful optimizations combine kernel fusion via `torch.compile` (achieving **1.41x training speedup**), CUDA graphs for launch overhead elimination (**1.7x MLPerf speedup**), and fused attention kernels like FlashAttention that reduce HBM accesses by **up to 9×**.

---

## Table of Contents

1. [The Memory Wall in Large-Batch DLRMs](#1-the-memory-wall-in-large-batch-dlrms)
2. [Broadcasting vs Explicit .expand(): Semantic Equivalence](#2-broadcasting-vs-explicit-expand-semantic-equivalence)
3. [The Materialization Trap: When Views Become Copies](#3-the-materialization-trap-when-views-become-copies)
4. [TorchInductor Compiler Optimization](#4-torchinductor-compiler-optimization)
5. [Kernel Launch Overhead and Fusion](#5-kernel-launch-overhead-and-fusion)
6. [Memory Bandwidth in Attention Mechanisms](#6-memory-bandwidth-in-attention-mechanisms)
7. [SwishLayerNorm Fusion](#7-swishlayernorm-fusion)
8. [Loss Computation Consolidation](#8-loss-computation-consolidation)
9. [CUDA Graphs for Static Workloads](#9-cuda-graphs-for-static-workloads)
10. [Triton Kernels and H100-Specific Optimizations](#10-triton-kernels-and-h100-specific-optimizations)
11. [Control Flow Optimization: Hoisting Strategy](#11-control-flow-optimization-hoisting-strategy)
12. [Profiling and Production Validation](#12-profiling-and-production-validation)

---

## 1. The Memory Wall in Large-Batch DLRMs

### 1.1 The Physics of Batch Size 256K

Processing a batch size of 256,000 samples creates a distinct performance profile dominated by **memory bandwidth utilization**. Unlike dense LLMs or CNNs where compute intensity is often high due to massive matrix multiplications, DLRMs present a heterogeneous workload that places unique stress on the memory hierarchy.

Consider a standard hidden layer dimension ($H$) of 1,024, typical in recommendation architectures. A single intermediate activation tensor for this batch size, stored in bfloat16 (2 bytes per element), consumes:

$$\text{Memory}_{\text{tensor}} = \text{Batch} \times \text{Hidden} \times \text{Sizeof}(\text{bfloat16})$$

$$\text{Memory}_{\text{tensor}} = 256,000 \times 1,024 \times 2 \text{ bytes} \approx 500 \text{ MB}$$

A single tensor occupies half a gigabyte of GPU memory. Any operation that reads this tensor, performs a simple element-wise computation, and writes it back must transfer **1 GB of data** across the memory bus (500 MB read + 500 MB write).

On an NVIDIA A100 GPU with peak memory bandwidth of approximately 1,555 GB/s (SXM4) or 1,935 GB/s (80GB version), this operation has a theoretical minimum latency:

$$\text{Time}_{\text{min}} = \frac{1 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.64 \text{ ms}$$

In reality, efficiency losses due to non-contiguous memory access, DRAM page misses, and protocol overheads reduce effective bandwidth. **If the model accidentally triggers a memory copy, this latency doubles or triples.**

### 1.2 The Arithmetic Intensity Gap

The central challenge in optimizing these kernels is **Arithmetic Intensity** ($AI$), defined as the ratio of floating-point operations performed to bytes accessed from main memory:

$$AI = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

Operations like Matrix Multiplication (GEMM) have high arithmetic intensity because they perform $O(N^3)$ computations on $O(N^2)$ data. Conversely, the operations common in DLRMs—element-wise additions, Swish activations, and LayerNorms—are inherently **memory-bound**:

**Element-wise Addition**: $C = A + B$
- Reads: 2 elements ($A_i, B_i$)
- Ops: 1 addition
- Writes: 1 element ($C_i$)
- $AI \approx 1/3$ FLOPs/element

At batch size 256K, these low-intensity operations saturate the memory bandwidth long before they tax the compute units. The goal of optimization is **not to reduce FLOPs, but to reduce bytes transferred**.

### 1.3 The Hierarchy of Broadcasting Economics

Broadcasting serves as a primary mechanism to artificially inflate Arithmetic Intensity:

**Without Broadcasting (Materialized):**
- Read Tensor A (500 MB)
- Read Tensor B (500 MB - explicitly expanded)
- Write Result (500 MB)
- Total Traffic: **1.5 GB**

**With Broadcasting (Virtual):**
- Read Tensor A (500 MB)
- Read Vector b (2 KB - stays in L2/L1 cache)
- Write Result (500 MB)
- Total Traffic: **1.0 GB**

Broadcasting immediately reduces required memory bandwidth by **33%**. The decision between explicit expand and implicit broadcasting is not merely stylistic—it determines whether you expose the system to the risk of reverting to the "Without Broadcasting" scenario.

---

## 2. Broadcasting vs Explicit .expand(): Semantic Equivalence

### 2.1 The Anatomy of a Tensor View

A PyTorch tensor is composed of two distinct entities:

1. **Storage**: A container holding the raw data pointer (e.g., `float*`) to the physical memory allocation on the GPU
2. **TensorImpl (Metadata)**: A lightweight structure containing the shape (sizes), traversal logic (strides), dtype, and offset from the storage base pointer

### 2.2 Pattern A: Explicit .expand()

```python
expanded = tensor.expand(batch_size, -1, -1)
```

When `.expand()` is invoked, PyTorch performs a **purely metadata operation**. It creates a new TensorImpl that points to the same Storage, manipulating the stride array.

**Stride-0 Semantics**: If a dimension has size $N > 1$ but stride $0$, it indicates that advancing the index in that dimension does not advance the pointer in memory. This is the internal representation of a broadcast.

**Allocation**: No new GPU memory is allocated for the data. The cost is negligible (microseconds of CPU time to allocate the metadata struct).

### 2.3 Pattern B: Implicit Broadcasting

```python
result = tensor + other_tensor
```

When implicit broadcasting occurs, the PyTorch dispatcher (or the Inductor compiler) calculates the effective output shape and the necessary strides on the fly. Conceptually, an ephemeral view is created to align dimensions using the same stride-0 logic.

**Allocation**: Zero additional memory for input expansion.

### 2.4 TorchInductor Treats Both Patterns Identically

TorchInductor aggressively normalizes these patterns during the graph lowering phase:

1. **AOT Autograd**: Captures PyTorch code into an FX graph. Explicit `.expand()` calls are recorded as `aten.expand` nodes; implicit broadcasts in operations like `aten.add` are recorded as `aten.add` nodes with shape mismatch
2. **Decomposition & Lowering**: Inductor lowers these high-level ops to a simpler IR, identifying that `aten.expand` is a View Operation requiring no compute
3. **Kernel Fusion**: Inductor fuses sequences of pointwise operations (like `expand → add → swish`) into a single kernel
4. **Triton Code Generation**: Whether the input was explicitly expanded or implicitly broadcasted, Inductor generates a Triton kernel that utilizes indexing arithmetic

**Conclusion**: In `torch.compile` mode, Pattern A and Pattern B result in **identical binary code**, provided the `.expand()` does not interact with a graph break.

---

## 3. The Materialization Trap: When Views Become Copies

### 3.1 When Zero-Copy Guarantee Breaks

While `.expand()` itself is zero-copy, the resulting tensor is **non-contiguous**. Many PyTorch operations will implicitly trigger a copy:

| Operation Type | Examples | Behavior with Expanded View | Consequence at Batch 256K |
|----------------|----------|----------------------------|---------------------------|
| Contiguity Check | `.contiguous()` | Allocates new memory; copies data | **Catastrophic**: Allocates 500MB, initiates 500MB Read + 500MB Write |
| Reshaping | `.view()` (some cases), `.reshape()` | `.reshape()` may call `.contiguous()` if strides are incompatible | **High Risk**: If stride logic fails, a copy is forced |
| Legacy Kernels | Some `torch.mm` paths, custom C++ extensions | May internally call `x.contiguous()` before processing | **Hidden Latency**: Developer may not see explicit copy in Python code |
| In-Place Ops | `x += y` | Modifying an expanded tensor in-place is illegal/undefined | **Error/Copy**: PyTorch will raise an error or force a clone |
| Cross-Device | `.to(device)`, `.cuda()`, `.cpu()` | Transfers data | **Critical**: Usually copies the underlying storage |

### 3.2 Critical Footgun: Device Transfers

Moving expanded tensors between devices triggers full materialization:

```python
n = int(1e8)
a = torch.randn(n)
a.cuda().expand((n, n))  # Works - expand AFTER moving
a.expand((n, n)).cuda()  # OOM! - 40+ GB materialized during transfer
```

### 3.3 cuBLAS and cuDNN Behavior

- **cuBLAS strided batched GEMM** accepts stride parameters and can handle stride=0 for batch broadcasting
- **cuDNN convolutions and batch norm** typically require contiguous inputs—non-contiguous tensors trigger `CUDNN_STATUS_NOT_SUPPORTED` errors
- **FlashAttention and scaled_dot_product_attention** require `.stride()[-1] == 1` for all inputs

### 3.4 Pattern B Acts as a Safeguard

In contrast to Pattern A, implicit broadcasting handles the expansion logic **inside the element-wise kernel iterator**. The kernel is generated to handle stride-0 access natively. Since the "expanded" tensor never exists as a Python object passed to other functions, it **cannot be accidentally materialized** by a distinct function call.

### 3.5 Memory Footprint Differential

**Scenario 1: Ideal Execution (No Materialization)**

| Pattern | Footprint |
|---------|-----------|
| Pattern A | ≈ 0 (only CPU-side metadata overhead) |
| Pattern B | = 0 |
| **Difference** | Negligible |

**Scenario 2: Accidental Materialization**

| Pattern | Footprint |
|---------|-----------|
| Pattern A (BF16, 256K × 1024 materialized) | **500 MB** allocation + 500 MB Write + 500 MB Read |
| Pattern B | 0 bytes |
| **Difference** | **500 MB per tensor** |

In a complex DLRM forward pass with thousands of operations, even a 1% rate of accidental materialization can lead to **Out-Of-Memory (OOM) errors** or significant throughput degradation.

---

## 4. TorchInductor Compiler Optimization

### 4.1 Five-Phase Optimization Strategy

TorchInductor operates through five distinct phases:

1. **Pre-grad passes**: High-level Torch IR optimization
2. **AOT Autograd**: Forward/backward graph derivation with decomposition
3. **Post-grad passes**: Normalized ATen IR optimizations
4. **Scheduling**: Dependency analysis and fusion decisions
5. **Code generation**: Hardware-specific kernel emission

### 4.2 IR Design: TensorBox and StorageBox Abstractions

TorchInductor's sophisticated IR design distinguishes between views, storage, and computation through TensorBox/StorageBox abstractions, enabling advanced fusion optimizations.

The `ExpandView` IR node represents a view without memory allocation:

```python
@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes):
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))
```

### 4.3 Kernel Selection and Fusion Algorithm

The scheduler's fusion algorithm makes decisions based on memory access patterns:

```python
class SchedulerNode:
    def can_fuse(self, other):
        # Check memory dependencies and indexing compatibility
        if self.has_incompatible_strides(other):
            return False
        return self.estimate_fusion_benefit(other) > threshold
```

**Fusion Strategies:**

- **Vertical fusion**: Producer-consumer chains (e.g., `expand → elementwise`)
- **Horizontal fusion**: Independent parallel operations
- **Loop fusion**: Compatible iteration patterns across operations

### 4.4 Autotuning Configuration Selection

```python
# Example autotuning configurations for large tensors
configs = [
    {'XBLOCK': 64, 'RBLOCK': 8, 'num_warps': 4},
    {'XBLOCK': 128, 'RBLOCK': 4, 'num_warps': 8},
    {'XBLOCK': 32, 'RBLOCK': 16, 'num_warps': 2}
]
```

Benchmarking shows **30% performance improvements** for outer reductions through better heuristic tuning, with **10% of kernels exhibiting over 2x speedups**.

### 4.5 Generated Triton IR

In Triton, broadcasting is implemented via masking and modular arithmetic on pointers, not by data duplication:

```python
@triton.jit
def fused_add_kernel(ptr_x, ptr_bias, ptr_out, n_elements, ...):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load large tensor X (contiguous)
    x = tl.load(ptr_x + offsets, mask=mask)

    # Load small bias (broadcasted)
    # The stride for the bias dimension is effectively 0 in the batch dimension
    bias_index = offsets % BIAS_SIZE  # Simplified broadcasting logic
    bias = tl.load(ptr_bias + bias_index, mask=mask)

    # Compute
    output = x + bias
    tl.store(ptr_out + offsets, output, mask=mask)
```

Because the bias tensor is small (relative to the 256K batch), it resides almost entirely in the GPU's L2 cache (or even L1 cache). The generated machine code (SASS) will issue load instructions that hit the cache **>99% of the time**.

---

## 5. Kernel Launch Overhead and Fusion

### 5.1 The Launch Overhead Problem

Kernel launch overhead of **5-10μs per operation** compounds dramatically in modern PyTorch training. With transformer layers executing 50+ operations per forward pass, reducing kernel counts from hundreds to dozens can deliver **5-15% QPS improvements**.

NVIDIA benchmarks on V100 demonstrate the compounding effect: a simple elementwise kernel executing in **2.9μs** incurs an effective per-kernel time of **9.6μs** when synchronization exposes launch overhead—a **3.3x penalty**.

The overhead stems from three components:
1. **GPU launch latency** (~1-2μs for the GPU to begin execution)
2. **CPU wrapper overhead** (~3-10μs for the CUDA API call)
3. **Implicit synchronizations** triggered by operations like `tensor.item()` or conditional logic

### 5.2 Identifying Launch-Bound Workloads

The diagnostic formula is straightforward:

$$\text{if } \frac{(\text{total\_step\_time} - \sum\text{kernel\_times})}{\text{total\_step\_time}} > 0.3 \text{, workload is launch-bound}$$

In Nsight Systems traces, look for:
- Gaps between consecutive kernel executions in the GPU timeline
- Situations where CPU API time dominates GPU execution time
- Kernel durations under 10μs paired with high total step time

Meta's MAIProf analysis of production models found GPU idle exceeding **50% of training time** in unoptimized configurations, with SM utilization at just 9.1%.

### 5.3 TorchInductor Fusion Performance

TorchInductor achieves **2.27x inference and 1.41x training geometric mean speedup** across 180+ real-world models on A100. The fusion pipeline works through several stages:

1. TorchDynamo captures an FX graph via bytecode transformation
2. AOTAutograd generates forward and backward graphs
3. 191 decomposition passes break complex operations into fuseable primitives
4. The scheduler identifies fusion opportunities

**Most Impactful Fusion Patterns:**

| Pattern | Description |
|---------|-------------|
| **Elementwise sequences** | Chains like `mul → add → relu → sigmoid` collapse into one kernel |
| **Reduction patterns** | Combinations like `(x * y).sum()` fuse into single kernels that avoid intermediate memory writes |
| **Scaled dot-product attention** | PyTorch 2.2+ automatically dispatches to FlashAttention-2, fusing Q×K^T → softmax → dropout → ×V |

### 5.4 Configuration Options

```python
import torch._inductor.config as config
config.max_autotune = True           # Profile multiple kernel configurations
config.epilogue_fusion = True        # Fuse pointwise ops into matmul templates
config.aggressive_fusion = True      # Fuse even without shared memory access
config.triton.cudagraphs = True      # Combine with CUDA graphs for launch reduction

model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

| Mode | Primary Objective | Optimization Strategy | Trade-offs |
|------|-------------------|----------------------|------------|
| **default** | Balanced performance | Standard fusion and overhead reduction | Baseline memory and speed |
| **reduce-overhead** | Minimal latency | CUDA Graphs to bypass Python overhead | Increased memory usage |
| **max-autotune** | Maximum throughput | Triton-based template searching | Longest compilation time |
| **max-autotune-no-cudagraphs** | Throughput without memory spikes | Triton kernels without static graph overhead | High compute, moderate memory |

The `fullgraph=True` parameter forces an error if graph breaks occur, preventing silent fallback to eager mode.

---

## 6. Memory Bandwidth in Attention Mechanisms

### 6.1 Why Attention is Memory-Bound

For attention computation `softmax(QK^T/√d)V`, memory traffic—not compute—limits throughput regardless of batch size. Research shows attention kernels **do not benefit significantly from batching** because performance is constrained by DRAM reads.

Attention involves computing $S = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$, where $M$ is a mask or bias.

**Dimensions**: Batch $B$, Heads $H$, Sequence $L$

**Score Matrix Size**: $B \times H \times L \times L$

### 6.2 The Catastrophe of Materialized Masks

**Without Broadcasting (Materialized Mask)**:

If $M$ is materialized to match the score matrix size (Pattern A + Accidental Copy):

Size of $M$: $256,000 \times H \times L \times L \times 2$ bytes

For $L=128, H=8$: $256,000 \times 8 \times 128^2 \times 2 \approx 67 \text{ GB}$

**Result: Immediate OOM.** It is physically impossible to materialize the mask at this batch size on current hardware.

**With Broadcasting (Implicit/Virtual)**:
- The GPU reads the Score Matrix ($S$) from HBM
- The GPU reads the Mask ($M$) from L2 Cache (since $M$ is small and reused across the batch)
- **Bandwidth Savings**: The "Read" bandwidth for the add operation is dominated solely by $S$. The cost of reading $M$ is effectively zero.

### 6.3 FlashAttention Transforms the Equation

| Metric | Naive Attention | FlashAttention |
|--------|-----------------|----------------|
| HBM accesses | O(N²) | O(N²d²/M) where M=SRAM |
| Memory complexity | O(N²) | O(N) |
| A100 throughput | ~25% theoretical | **50-73%** theoretical |
| Training speedup | baseline | **7.6×** (GPT-2) |
| HBM reduction | baseline | **up to 9×** |

FlashAttention achieves this through:
- **Tiling**: Loading Q, K, V blocks into SRAM
- **Online softmax**: Incremental computation without materializing N×N
- **Recomputation**: Not storing S, P matrices—recompute during backward

### 6.4 Broadcasting in Attention Masks

A broadcast mask `[1, 1, N, N]` uses N² memory while `[B, H, N, N]` pre-expanded uses B×H×N² memory—potentially **2+ GB vs 8 KB** for typical dimensions.

**Critical PyTorch issue (#154363)**: GQA broadcasting without `enable_gqa=True` may dispatch to *less* memory-efficient backends, paradoxically increasing memory usage.

---

## 7. SwishLayerNorm Fusion

### 7.1 Component Operations

The SwishLayerNorm pattern ($\text{LayerNorm}(\text{Swish}(x))$) is a prime candidate for bandwidth optimization:

**Swish (SiLU)**: $y = x \cdot \sigma(x)$ — Element-wise

**LayerNorm**: $\mu = \text{mean}(y)$, $\sigma^2 = \text{var}(y)$, $z = \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ — Requires reduction (mean/var) and element-wise apply

### 7.2 Non-Fused Execution

| Kernel | Operation | Data Movement |
|--------|-----------|---------------|
| Kernel 1 (Swish) | Read $x$, Write $y$ | 500MB + 500MB |
| Kernel 2 (Mean/Var) | Read $y$, Write stats | 500MB + small |
| Kernel 3 (Norm) | Read $y$, Read stats, Write $z$ | 500MB + small + 500MB |
| **Total** | | **~2.5 GB transferred** |

### 7.3 Fused Execution (Inductor/Triton)

Inductor generates a single kernel:

1. **Load**: Read $x$ (500MB) into registers/SRAM
2. **Compute Swish**: Perform $x \cdot \sigma(x)$ in registers
3. **Compute Stats**: Perform Welford's online algorithm in Shared Memory to get $\mu, \sigma^2$ **without writing $y$ to global memory**
4. **Broadcast Parameters**: Load $\gamma, \beta$ (broadcasted from small vectors) into Shared Memory
5. **Normalize**: Apply normalization in registers
6. **Store**: Write $z$ (500MB)

| Metric | Non-Fused | Fused |
|--------|-----------|-------|
| Data Movement | ~2.5 GB | **~1.0 GB** |
| Performance Gain | baseline | **2.5× speedup** |
| Bandwidth Reduction | baseline | **60-70%** |

### 7.4 SwishLayerNorm Implementation

```python
class SwishLayerNorm(nn.Module):
    def __init__(self, input_dims, device=None):
        super().__init__()
        self.norm = nn.Sequential(
            nn.LayerNorm(input_dims, device=device),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return input * self.norm(input)  # Y = X * Sigmoid(LayerNorm(X))
```

cuDNN 9.4+ supports LayerNorm/RMSNorm with activation fusion including Swish.

---

## 8. Loss Computation Consolidation

### 8.1 The Problem: Excessive Kernel Launches

Multi-task training scenarios often spawn excessive kernel launches from separate loss computations. The target reduction from **221 to ~30 kernels** is achievable through systematic batching of similar loss types.

### 8.2 Batched Cross-Entropy Pattern

For cross-entropy losses across multiple classification heads with the same number of classes, concatenating predictions and targets enables single-kernel computation:

```python
@torch.compile(mode="max-autotune", fullgraph=True)
def batched_cross_entropy(preds_list, targets_list, weights=None):
    # Batch all predictions: [total_samples, num_classes]
    preds_batched = torch.cat(preds_list, dim=0)
    targets_batched = torch.cat(targets_list, dim=0)

    # Single kernel for all losses
    loss_per_sample = F.cross_entropy(preds_batched, targets_batched, reduction='none')

    # Apply per-task weighting if needed
    if weights is not None:
        idx = 0
        weighted_sum = 0.0
        for i, pred in enumerate(preds_list):
            n = len(pred)
            weighted_sum += weights[i] * loss_per_sample[idx:idx+n].mean()
            idx += n
        return weighted_sum
    return loss_per_sample.mean()
```

### 8.3 Dynamic Loss Balancing

**Uncertainty-based weighting** (Kendall et al.) learns task-specific precision parameters:

$$\text{total\_loss} = \sum_i \left( e^{-\log\sigma_i} \cdot \mathcal{L}_i + \log\sigma_i \right)$$

**RLW (Random Loss Weighting)** normalizes each loss by its detached value:

$$\text{total\_loss} = \sum_i \frac{\mathcal{L}_i}{\text{detach}(\mathcal{L}_i)}$$

This achieves automatic magnitude balancing without learned parameters.

### 8.4 Gradient Accumulation Constraints

When using mixed precision, a **single GradScaler must serve all losses**—using separate scalers corrupts the gradient accumulation math. Scale the combined loss once before backward, then call `scaler.step()` and `scaler.update()` only on accumulation boundaries.

---

## 9. CUDA Graphs for Static Workloads

### 9.1 Mechanism and Performance

CUDA graphs record sequences of GPU operations during capture and replay them with a single `cudaGraphLaunch` call, eliminating Python/C++/driver overhead entirely.

| Metric | With CUDA Graphs |
|--------|------------------|
| MLPerf v1.0 Mask R-CNN | **~1.7× speedup** |
| MLPerf v1.0 BERT at scale | ~1.12× speedup |
| Per-kernel overhead reduction | 230% → 17% |

### 9.2 Capture Process

```python
# Warmup on side stream (critical for cuBLAS/cuDNN initialization)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):  # Minimum 3 iterations; 11 for DDP
        optimizer.zero_grad(set_to_none=True)  # set_to_none is cleaner for graphs
        with torch.autocast('cuda', torch.float16):
            y = model(static_input)
            loss = loss_fn(y, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)
    static_loss = loss_fn(static_output, static_target)
    scaler.scale(static_loss).backward()
```

### 9.3 Critical Constraints

- **No dynamic tensor shapes**: Batch size and sequence length must be fixed
- **No CPU-GPU synchronization points**
- **No data-dependent control flow**
- **No memory allocation** via `cudaMalloc`

For training loops with variable-length sequences: pre-capture multiple graphs for different shape buckets or pad to maximum length with attention masks.

### 9.4 Memory Implications

NVIDIA allocates **~64 KB per kernel** in graph overhead (reduced in CUDA 12.4+). For a model with 100 kernel launches across 10 different batch sizes, this totals ~64 MB of graph overhead.

Pool sharing via `torch.cuda.graph_pool_handle()` allows multiple graphs to share memory.

`torch.compile(mode="reduce-overhead")` provides a higher-level alternative that manages CUDA graphs automatically, including re-recording for new shapes through CUDA Graph Trees.

---

## 10. Triton Kernels and H100-Specific Optimizations

### 10.1 Triton Performance Characteristics

Triton's block-based programming model enables near-expert-level GPU performance with significantly reduced development time:

| Metric | CUDA | Triton |
|--------|------|--------|
| Lines for GEMM kernel | 100-500+ | **25-50** |
| Performance vs hand-tuned | 100% | **80-95%** |

### 10.2 H100 Tensor Memory Accelerator (TMA)

TMA provides a dedicated hardware unit for async bulk data transfer between global and shared memory, bypassing the register file entirely:

```python
@triton.jit
def kernel_with_tma(a_desc_ptr, block_m, block_k, ...):
    offs_am = tl.program_id(0) * block_m
    offs_k = 0
    # TMA descriptor handles address calculation and transfer
    a = tl._experimental_descriptor_load(
        a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv
    )
```

| Metric | Without TMA | With TMA |
|--------|-------------|----------|
| Memory throughput | 910 GB/s | **1.45 TB/s** |
| **Improvement** | baseline | **59%** |

### 10.3 Warp Specialization

Warp specialization divides warps into producer (data movement) and consumer (compute) roles, automatically enabled via autotune parameters:

```python
num_consumer_groups=2
num_buffers_warp_spec=3
```

This delivers **10-15% speedup** on FlashAttention and FP8 GEMM kernels. The Tawa automatic warp specialization framework achieves **1.21× speedup** over baseline Triton.

### 10.4 Integration with PyTorch Training

```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::fused_gelu", mutates_args={})
def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    wrap_triton(gelu_kernel)[(triton.cdiv(n, 1024),)](x, out, n, BLOCK_SIZE=1024)
    return out

@fused_gelu.register_autograd
def backward(ctx, grad):
    x, = ctx.saved_tensors
    return grad * gelu_backward_impl(x)  # Uses another triton_op
```

---

## 11. Control Flow Optimization: Hoisting Strategy

### 11.1 Graph Breaks in Conditionals

When `torch.compile` traces conditional code, it evaluates the condition:

**Static Condition**: If `bf16_training` is a global constant or hyperparameter known at compile time, Dynamo traces only the taken branch—no runtime graph break.

**Dynamic Condition**: If `bf16_training` varies at runtime, Dynamo cannot predict the path and inserts a Graph Break, compiling code before the `if` and inside the branches separately.

```python
# Before: Code duplication and potential graph breaks
if bf16_training:
    result = func(tensor.unsqueeze(1).to(bf16))
else:
    result = func(tensor.unsqueeze(1))
```

### 11.2 Systematic Hoisting

**Strategy**: Extract Loop-Invariant (or Branch-Invariant) operations—a classic compiler optimization called Code Motion.

**Identification Steps**:

1. **AST Analysis**: Let $S_{true}$ be the set of operations in the if block, $S_{false}$ be the set in the else block. Identify the longest common prefix $P = S_{true} \cap S_{false}$ starting from input variables.
2. **Dependency Check**: Verify common operations do not depend on variables defined only within specific branch logic.
3. **Side-Effect Check**: Ensure operations are pure (no printing, no global state mutation).

### 11.3 Refactored Pattern

```python
# Step 1: Hoist the common view operation
tensor_view = tensor.unsqueeze(1)

# Step 2: Handle the divergent dtype logic
# Option A: Use Autocast (Preferred for mixed precision)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16_training):
    result = func(tensor_view)

# Option B: Explicit Functional Control Flow (if manual control needed)
target_dtype = torch.bfloat16 if bf16_training else tensor.dtype
tensor_ready = tensor_view.to(target_dtype)
result = func(tensor_ready)
```

### 11.4 Autocast-Safe Caching Pattern

Pre-computed float32 tensors inside autocast regions cause dtype mismatches:

```python
class AutocastSafeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1024, 1024))
        self._precomputed = {}

    def _get_cached(self, dtype, device):
        key = (dtype, device)
        if key not in self._precomputed:
            self._precomputed[key] = self.weight.to(dtype=dtype, device=device)
        return self._precomputed[key]

    def forward(self, x):
        # Automatically uses correct dtype inside/outside autocast
        w = self._get_cached(x.dtype, x.device)
        return x @ w
```

**Critical autocast rules:**
- Never call `.half()` or `.bfloat16()` manually inside autocast—it handles casting automatically
- In-place operations don't autocast (`a.addmm_()` won't work; use `a.addmm()`)
- Store buffers in float32 and let autocast convert on-the-fly

### 11.5 Using torch.fx for Systematic Identification

```python
import torch.fx as fx

def find_hoistable_ops(model, example_input):
    traced = fx.symbolic_trace(model)
    hoistable = []
    for node in traced.graph.nodes:
        if node.target in [torch.unsqueeze, torch.expand]:
            # Check if inputs are constants/parameters
            hoistable.append(node)
    return hoistable
```

### 11.6 Validating with Profiling Tools

```bash
TORCH_LOGS="graph_breaks,output_code" python train.py
```

**Before**: Log entries like `Graph break: if bf16_training`

**After**: Continuous trace or graph including `torch.ops.aten.unsqueeze` followed by casting logic, fused into a single Triton kernel

---

## 12. Profiling and Production Validation

### 12.1 Profiling Strategy

**Detect Materialization Events:**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)
```

Look for "Allocations" that match the size of your batch (500 MB). If you see an allocation immediately followed by `aten::copy_` or `aten::contiguous`, you have confirmed a materialization event.

**Analyze Bandwidth**: Use Nsight Compute to look at "DRAM Throughput". If throughput is saturated but FLOPs are low during element-wise phases, you are strictly bandwidth-bound.

### 12.2 Estimated Costs Table

| Metric | Pattern A (Materialized) | Pattern B (Broadcasted) | Difference |
|--------|-------------------------|------------------------|------------|
| Allocation Size | 500 MB | 0 MB | +500 MB |
| HBM Read | 1000 MB (A + B_exp) | 500 MB (A) | +500 MB |
| HBM Write | 500 MB | 500 MB | 0 MB |
| L2 Cache Hit Rate | Low (Thrashing) | High (99% for B) | **Critical** |
| Latency (A100) | ~1.0 ms | ~0.65 ms | **~35% Slower** |

### 12.3 Production Benchmarks

**DeepSpeed Deep Fusion:**

| Fusion Pattern | Speedup |
|----------------|---------|
| Input LayerNorm + QKV GEMM + bias adds | **1.5×** over unfused cuBLAS |
| Attention with implicit matrix transformation | **2.9×** |
| Intermediate FF + LayerNorm + bias + residual + GELU | **3.0×** |
| MoE kernel optimization | **>6× reduction** in MoE-related latency |

**LinkedIn's Liger Kernel (Triton-based):**

| Metric | Improvement |
|--------|-------------|
| Multi-GPU training throughput | **20% improvement** |
| Memory reduction | **60%** from fused operations |

Includes fused RMSNorm, RoPE, SwiGLU, and FusedLinearCrossEntropy.

**Internal Production Measurements:**

| Optimization | Impact |
|--------------|--------|
| CUDA synchronization point removal | **12% training QPS boost** |
| Algorithmic optimization in CMSL models | **40% QPS increase** |
| 2× faster performance with fused kernels | B200 hardware |
| OmniFM v3 | **5% E2E improvement** |

### 12.4 Measurement Methodology

1. Use **CUDA events** for accurate timing (not Python `time.time()`)
2. Warm up for **10+ iterations** before measurement
3. Run **50+ iterations** taking the median
4. Profile with **Nsight Systems** to visualize kernel timeline

**Key metrics to track:**
- Kernel count per training step
- Achieved SM occupancy (target >50%)
- Memory bandwidth utilization (H100 theoretical: 3.35 TB/s HBM3)
- End-to-end throughput in samples/second

---

## Appendix: Quick Reference Answers

### RQ1: Does TorchInductor handle broadcasting differently than explicit .expand()?

**No.** TorchInductor aggressively normalizes these patterns during graph lowering. Both create ExpandView IR nodes with stride-0 semantics and generate identical Triton kernels. The expand becomes just a stride-based view with no data copy.

### RQ2: When does .expand() allocate memory vs remain a view?

`.expand()` itself **always** remains a view and never allocates memory. Allocation only occurs if you subsequently perform operations requiring contiguous memory:
- `.contiguous()` on expanded tensor → **500 MB allocation**
- Device transfers (`.cuda()`, `.cpu()`) → Full materialization
- `.reshape()` when view impossible → Forces contiguous copy
- `.clone()` → Explicit copy

### RQ3: How does broadcasting affect memory bandwidth in attention computation?

Broadcasting **dramatically reduces** memory bandwidth. A broadcast mask `[1, 1, N, N]` uses N² memory while `[B, H, N, N]` pre-expanded uses B×H×N² memory. For 256K batch with L=128, H=8: materialized mask would be **~67 GB** (immediate OOM) vs **~8 KB** with broadcasting. FlashAttention further reduces HBM accesses by **up to 9×**.

### RQ4: What's the memory footprint difference for batch_size=256K between the two patterns?

**Ideal execution**: Both patterns have **0 additional memory footprint** (only result tensor allocated).

**Accidental materialization (risk case)**: Pattern A with materialization costs **500 MB per tensor** vs 0 for Pattern B. Compounding across 1000 layers: 10+ GB worst case with severe fragmentation.

### RQ5: How to systematically identify and hoist common operations out of conditionals?

1. **AST Analysis**: Identify common prefix $P = S_{true} \cap S_{false}$
2. **Dependency Check**: Verify operations don't depend on branch-specific variables
3. **Side-Effect Check**: Ensure operations are pure
4. **Refactor**: Hoist to before conditional, use autocast for dtype handling
5. **Validate**: Use `TORCH_LOGS="graph_breaks"` to confirm graph continuity

### RQ6: What performance improvements can be expected?

| Optimization | Expected Impact |
|--------------|-----------------|
| Broadcasting over materialized expand | **33% bandwidth savings** |
| torch.compile (max-autotune) | **1.41× training speedup** |
| CUDA graphs | **1.7× speedup** (static workloads) |
| SwishLayerNorm fusion | **2.5× kernel speedup**, 60-70% bandwidth reduction |
| Loss consolidation (221→30 kernels) | **5-15% QPS improvement** |
| FlashAttention | **7.6× training speedup**, 9× HBM reduction |
| TMA on H100 | **59% memory throughput improvement** |

---

## References

1. NVIDIA Hopper Architecture In-Depth: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
2. Computing GPU memory bandwidth with Deep Learning Benchmarks - Paperspace Blog
3. TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
4. torch.Tensor.expand — PyTorch Documentation: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html
5. Inductor notes – Ian's Blog: https://ianbarber.blog/2024/01/16/inductor-notes/
6. Working with Graph Breaks — PyTorch Documentation: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html
7. Out of the box acceleration and memory savings of decoder models with PyTorch 2.0: https://pytorch.org/blog/out-of-the-box-acceleration/
8. Liger-Kernel: Efficient Triton Kernels for LLM Training - OpenReview
9. Understanding GPU Memory 1: Visualizing All Allocations over Time - PyTorch: https://pytorch.org/blog/understanding-gpu-memory-1/
10. FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
11. DeepSpeed Deep Fusion Documentation
12. MLPerf Training v1.0 Results
13. Stack Overflow - PyTorch broadcast vs expand memory usage
14. PyTorch Forums - Expand() memory savings discussion
