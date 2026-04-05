# Memory Bandwidth Optimization & Kernel Efficiency for Large-Scale DLRMs (Q4 Synthesis)

**Document Version**: 2.0
**Date**: 2026-01-30
**Author**: CSML Optimization Team
**Status**: Proposal (Revised)

---

## Executive Summary

This proposal synthesizes findings from Q4 deep research on **memory bandwidth optimization, kernel fusion, and launch overhead reduction** for Deep Learning Recommendation Models (DLRMs) processing large batches (256K samples). The research reveals a **paradigm shift**: at this scale, **memory bandwidth—not compute capacity—is the dominant performance constraint**.

### Critical Insight: The Memory Wall

At batch size 256K with hidden dimension 1024, a single intermediate tensor in bfloat16 consumes **~500 MB**. Any element-wise operation triggers **1 GB+ of memory traffic**. This fundamentally changes optimization priorities:

| Traditional Focus | Required Focus at 256K Batch |
|------------------|------------------------------|
| Reduce FLOPs | **Reduce bytes transferred** |
| Compute optimization | **Memory bandwidth optimization** |
| Kernel speed | **Kernel fusion to eliminate intermediates** |

### Quantified Impact Summary

| Optimization Category | Expected Impact | Confidence | Effort |
|-----------------------|-----------------|------------|--------|
| **Prevent accidental materialization** | **33% bandwidth savings** | High | Low |
| **torch.compile (max-autotune)** | **1.41× training speedup** | High | Low |
| **CUDA graphs** | **1.7× speedup** (static workloads) | High | Medium |
| **SwishLayerNorm fusion** | **2.5× kernel speedup, 60-70% bandwidth reduction** | High | Medium |
| **FlashAttention** | **7.6× training speedup, 9× HBM reduction** | High | Verified |
| **Loss consolidation (221→30 kernels)** | **5-15% QPS** | Already Applied | N/A |
| **H100 TMA optimization** | **59% memory throughput gain** | Medium | High |

### V2 Optimizations Already Applied

The V2 codebase has achieved significant improvements:

| Optimization | Impact | Status |
|--------------|--------|--------|
| Kernel fusion (221→30 kernels) | 87% kernel count reduction | ✅ Applied |
| CUDA sync point removal | 12% train, 20% eval | ✅ Applied |
| NumCandidatesInfo pattern | 2-8% QPS | ✅ Applied |
| PT2 framework-level compilation | Significant | ✅ Applied |

**Remaining Opportunity**: The research identifies **memory bandwidth efficiency** as the next frontier, with particular focus on preventing accidental tensor materialization.

---

## Table of Contents

1. [The Physics of Memory Bandwidth at Scale](#1-the-physics-of-memory-bandwidth-at-scale)
2. [Critical Risk: The Materialization Trap](#2-critical-risk-the-materialization-trap)
3. [TorchInductor Optimization Deep Dive](#3-torchinductor-optimization-deep-dive)
4. [Control Flow Optimization: Hoisting Strategy](#4-control-flow-optimization-hoisting-strategy)
5. [Phase 0: Low-Hanging Fruits (Highest Priority)](#5-phase-0-low-hanging-fruits-highest-priority)
6. [Phase 1: Kernel Fusion Opportunities](#6-phase-1-kernel-fusion-opportunities)
7. [Phase 2: CUDA Graphs Integration](#7-phase-2-cuda-graphs-integration)
8. [Phase 3: Attention Optimization](#8-phase-3-attention-optimization)
9. [Phase 4: Advanced Triton Optimizations](#9-phase-4-advanced-triton-optimizations)
10. [Hardware-Specific Considerations](#10-hardware-specific-considerations)
11. [Profiling & Validation Methodology](#11-profiling--validation-methodology)
12. [Prioritized Implementation Matrix](#12-prioritized-implementation-matrix)
13. [Recommended Execution Timeline](#13-recommended-execution-timeline)

---

## 1. The Physics of Memory Bandwidth at Scale

### 1.1 Why Batch Size 256K Changes Everything

At batch size 256K, a single hidden layer tensor (H=1024, bfloat16) occupies:

$$\text{Memory} = 256,000 \times 1,024 \times 2 \text{ bytes} \approx 500 \text{ MB}$$

Any read-modify-write operation transfers **1 GB** across the memory bus. On A100 with 1,555 GB/s peak bandwidth:

$$\text{Theoretical Minimum Latency} = \frac{1 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.64 \text{ ms}$$

**Critical Implication**: If code accidentally triggers a memory copy, latency doubles or triples.

### 1.2 Arithmetic Intensity: The Governing Metric

**Arithmetic Intensity (AI)** = FLOPs / Bytes Accessed

| Operation Type | AI | Bottleneck |
|---------------|-----|------------|
| Matrix Multiplication (GEMM) | High (~100s) | Compute-bound |
| Element-wise ops (add, mul) | ~1/3 | **Memory-bound** |
| Swish activation | ~2 | **Memory-bound** |
| LayerNorm | ~5-10 | **Memory-bound** |

**Key Insight**: At batch 256K, most DLRM operations are **strictly memory-bound**. The goal is **not to reduce FLOPs, but to reduce bytes transferred**.

### 1.3 The Economics of Broadcasting

**Without Broadcasting (Materialized)**:
- Read Tensor A: 500 MB
- Read Tensor B (expanded): 500 MB
- Write Result: 500 MB
- **Total: 1.5 GB**

**With Broadcasting (Virtual)**:
- Read Tensor A: 500 MB
- Read Vector b: ~2 KB (fits in L2 cache)
- Write Result: 500 MB
- **Total: ~1.0 GB**

Broadcasting provides **33% bandwidth savings** by keeping small tensors in cache.

---

## 2. Critical Risk: The Materialization Trap

### 2.1 Understanding Tensor Views vs. Copies

PyTorch tensors consist of:
1. **Storage**: Raw data pointer to GPU memory
2. **TensorImpl**: Metadata (shape, strides, dtype, offset)

Both `.expand()` and implicit broadcasting use **stride-0 semantics**—they create views without copying data. However, this zero-copy guarantee can break.

### 2.2 Operations That Trigger Materialization

| Operation | Trigger | Impact at 256K Batch |
|-----------|---------|----------------------|
| `.contiguous()` | Explicit call | **500 MB allocation + 1 GB transfer** |
| `.reshape()` | When strides incompatible | **Silent copy** |
| Device transfer | `.to(device)`, `.cuda()`, `.cpu()` | **Full materialization** |
| In-place ops | `x += y` on expanded tensor | **Error or clone** |
| Legacy kernels | Some cuBLAS/custom extensions | **Hidden `.contiguous()` call** |

### 2.3 cuBLAS/cuDNN/FlashAttention Stride Requirements

**Critical library-specific constraints** that affect expanded tensor handling:

| Library | Requirement | Consequence |
|---------|-------------|-------------|
| **cuBLAS strided batched GEMM** | Accepts stride=0 for batch broadcasting | ✅ Compatible with expanded tensors |
| **cuDNN** (convolutions, batch norm) | **Requires contiguous inputs** | ❌ Triggers `CUDNN_STATUS_NOT_SUPPORTED` or implicit `.contiguous()` |
| **FlashAttention / SDPA** | **Requires `.stride()[-1] == 1` for all inputs** | ❌ May fail or force copy on non-contiguous tensors |

**Practical Implication**: When using FlashAttention with masks, ensure the mask tensor has contiguous last dimension. Broadcasting masks via `[1, 1, N, N]` shape with proper strides works, but arbitrary reshaping may trigger copies.

### 2.4 The Device Transfer Footgun

**This is a critical pattern to audit:**

```python
# SAFE: Expand AFTER moving to device
tensor.cuda().expand((batch_size, -1, -1))

# DANGEROUS: Expand BEFORE moving - triggers full materialization!
tensor.expand((batch_size, -1, -1)).cuda()  # OOM with large expansion!
```

For batch 256K with 1024-dim tensor, the dangerous pattern attempts to materialize **500 MB per tensor**.

### 2.5 Pattern B (Implicit Broadcasting) as Safeguard

**Pattern A** (Explicit expand):
```python
expanded = tensor.expand(batch_size, -1, -1)
result = expanded + other_tensor  # Risk: expanded can be passed to unsafe functions
```

**Pattern B** (Implicit broadcasting):
```python
result = tensor + other_tensor  # Safe: expansion happens inside kernel
```

**Critical Difference**: With Pattern B, the "expanded" tensor never exists as a Python object that could be passed to `.contiguous()`, device transfers, or legacy kernels. The expansion is handled **inside the fused kernel**.

**Recommendation**: **Prefer implicit broadcasting (Pattern B) unless explicit expand is required for clarity**. Pattern A and B generate identical TorchInductor code, but Pattern A exposes the code to accidental materialization.

### 2.6 Codebase Audit Checklist

| Item | Action | Risk Level |
|------|--------|------------|
| ☐ Search for `.expand()` followed by `.to()` | Audit all occurrences | **Critical** |
| ☐ Search for `.expand()` followed by `.contiguous()` | Likely bug | **Critical** |
| ☐ Search for `.expand()` followed by `.reshape()` | May trigger copy | **High** |
| ☐ Verify custom CUDA/Triton kernels handle stride-0 | May force contiguous | **Medium** |
| ☐ Check cross-device transfers | Ensure expand happens post-transfer | **Critical** |
| ☐ Verify FlashAttention inputs have `.stride()[-1] == 1` | May trigger copy or error | **High** |
| ☐ Check cuDNN operations for non-contiguous inputs | Triggers `CUDNN_STATUS_NOT_SUPPORTED` | **High** |

---

## 3. TorchInductor Optimization Deep Dive

### 3.1 How TorchInductor Normalizes Broadcasting

TorchInductor's five-phase optimization:

1. **Pre-grad passes**: High-level IR optimization
2. **AOT Autograd**: Forward/backward graph derivation
3. **Post-grad passes**: Normalized ATen IR optimizations
4. **Scheduling**: Dependency analysis and fusion decisions
5. **Code generation**: Hardware-specific Triton kernel emission

**Key Finding**: TorchInductor treats `.expand()` as an `ExpandView` IR node with **zero memory allocation**. Both explicit expand and implicit broadcasting generate **identical Triton kernels**.

### 3.2 Generated Triton Kernel Pattern

```python
@triton.jit
def fused_add_kernel(ptr_x, ptr_bias, ptr_out, n_elements, BIAS_SIZE, ...):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load large tensor (contiguous)
    x = tl.load(ptr_x + offsets, mask=mask)

    # Load small bias (broadcasted via modular arithmetic)
    bias_index = offsets % BIAS_SIZE  # Stride-0 emulation
    bias = tl.load(ptr_bias + bias_index, mask=mask)

    # Compute and store
    output = x + bias
    tl.store(ptr_out + offsets, output, mask=mask)
```

The bias tensor, being small, resides in L2/L1 cache with **>99% cache hit rate**.

### 3.3 Fusion Patterns That Eliminate Memory Traffic

| Pattern | Description | Memory Savings |
|---------|-------------|----------------|
| **Elementwise sequences** | `mul → add → relu → sigmoid` → single kernel | Eliminates all intermediates |
| **Reduction patterns** | `(x * y).sum()` → fused kernel | No intermediate tensor |
| **GEMM epilogue** | `Linear + bias + activation` → fused | Activation is "free" |
| **Scaled dot-product attention** | Q×K^T → softmax → dropout → ×V | FlashAttention fusion |

### 3.4 Graph Breaks: The Silent Performance Killer

Graph breaks force TorchInductor to emit separate compilation units, **destroying fusion opportunities**.

**Common Graph Break Causes**:
- Data-dependent control flow (`if tensor.sum() > threshold`)
- `.item()`, `.tolist()` calls
- Print statements in hot path
- Unsupported operations

**Detection**:
```bash
TORCH_LOGS="graph_breaks,output_code" python train.py
```

**Target**: Zero graph breaks in the critical forward/backward path.

---

## 4. Control Flow Optimization: Hoisting Strategy

### 4.1 Graph Breaks in Conditionals

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

### 4.2 Systematic Hoisting Technique

**Strategy**: Extract Loop-Invariant (or Branch-Invariant) operations—a classic compiler optimization called Code Motion.

**Identification Steps**:

1. **AST Analysis**: Let $S_{true}$ be the set of operations in the if block, $S_{false}$ be the set in the else block. Identify the longest common prefix $P = S_{true} \cap S_{false}$ starting from input variables.
2. **Dependency Check**: Verify common operations do not depend on variables defined only within specific branch logic.
3. **Side-Effect Check**: Ensure operations are pure (no printing, no global state mutation).

### 4.3 Refactored Pattern

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

### 4.4 Autocast-Safe Caching Pattern

**Problem**: Pre-computed float32 tensors inside autocast regions cause dtype mismatches. This is a critical pattern for mixed-precision training.

```python
class AutocastSafeModule(nn.Module):
    """Cache tensors per dtype to handle autocast correctly."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1024, 1024))
        self._precomputed: dict[tuple, torch.Tensor] = {}

    def _get_cached(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        if key not in self._precomputed:
            self._precomputed[key] = self.weight.to(dtype=dtype, device=device)
        return self._precomputed[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Automatically uses correct dtype inside/outside autocast
        w = self._get_cached(x.dtype, x.device)
        return x @ w
```

**Critical autocast rules:**
- Never call `.half()` or `.bfloat16()` manually inside autocast—it handles casting automatically
- In-place operations don't autocast (`a.addmm_()` won't work; use `a.addmm()`)
- Store buffers in float32 and let autocast convert on-the-fly

### 4.5 Using torch.fx for Systematic Identification

```python
import torch.fx as fx

def find_hoistable_ops(model: nn.Module, example_input: torch.Tensor) -> list:
    """Automatically identify operations that can be hoisted out of conditionals."""
    traced = fx.symbolic_trace(model)
    hoistable = []
    for node in traced.graph.nodes:
        if node.target in [torch.unsqueeze, torch.expand, torch.Tensor.view]:
            # Check if inputs are constants/parameters (not dependent on branch)
            hoistable.append(node)
    return hoistable
```

### 4.6 Validating Graph Continuity

```bash
TORCH_LOGS="graph_breaks,output_code" python train.py
```

**Before**: Log entries like `Graph break: if bf16_training`

**After**: Continuous trace or graph including `torch.ops.aten.unsqueeze` followed by casting logic, fused into a single Triton kernel

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 3/5 | Requires code refactoring |
| **Complexity** | 2/5 | Standard compiler optimization pattern |
| **Success Probability** | **High** | Eliminates graph breaks |
| **Risk Level** | **Low** | No numerical changes |

**Expected Impact**: Eliminates graph breaks, enables fusion
**Effort**: 0.5-1 day per module

---

## 5. Phase 0: Low-Hanging Fruits (Highest Priority)

### ⚠️ Before complex optimizations, audit for quick wins!

### 5.1 TorchInductor Configuration Optimization

**Current State**: V2 uses framework-level PT2 but may not have optimal inductor settings.

**Recommended Configuration**:
```python
import torch._inductor.config as config

# Enable aggressive fusion
config.max_autotune = True              # Profile multiple kernel configurations
config.epilogue_fusion = True           # Fuse bias+activation into GEMM
config.aggressive_fusion = True         # Fuse even without shared memory benefit
config.coordinate_descent_tuning = True # Better kernel selection

# Memory optimization
config.triton.cudagraphs = True         # Combine with CUDA graphs
```

**⚠️ IMPORTANT NOTES**:
- `epilogue_fusion` is **explicitly set to False** in some inference configs. **Investigate why before enabling**—there may be a historical regression.
- `max_autotune` is True for inference but **False for training PT2Config**. Enable for training.

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 5/5 | Config changes only |
| **Complexity** | 1/5 | Standard PyTorch options |
| **Success Probability** | **Medium-High** | 5-10% marginal on already-compiled models |
| **Risk Level** | **Low** | Can be toggled off |

**Expected Impact**: 5-10% marginal improvement
**Effort**: 1-2 hours

---

### 5.2 Materialization Audit

**Problem**: Accidental materialization can instantly degrade performance by 33%+ per tensor.

**Action**: Systematic codebase search for dangerous patterns.

```bash
# Search for expand followed by device transfer
grep -rn "\.expand.*\.to\(" --include="*.py"
grep -rn "\.expand.*\.cuda\(" --include="*.py"
grep -rn "\.expand.*\.cpu\(" --include="*.py"

# Search for expand followed by contiguous
grep -rn "\.expand.*\.contiguous\(" --include="*.py"

# Search for expand followed by reshape
grep -rn "\.expand.*\.reshape\(" --include="*.py"
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 5/5 | Search and review |
| **Complexity** | 1/5 | Pattern matching |
| **Success Probability** | **High** | Will identify issues if present |
| **Risk Level** | **None** | Audit only |

**Expected Impact**: Prevents 33% per-tensor degradation
**Effort**: 2-4 hours

---

### 5.3 Kernel Count Profiling

**Problem**: V2 achieved 221→30 kernels for loss. What's the **total training step kernel count**?

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_training_step(model, sample_input):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            _ = model(sample_input)

    # Count CUDA kernels
    cuda_events = [e for e in prof.events() if e.device_type == 1]
    print(f"Total CUDA kernels: {len(cuda_events)}")

    # Identify launch-bound workload
    total_cuda_time = sum(e.cuda_time for e in cuda_events)
    total_step_time = prof.profiler.total_average().self_cuda_time_total

    launch_overhead_ratio = (total_step_time - total_cuda_time) / total_step_time
    if launch_overhead_ratio > 0.3:
        print(f"⚠️ LAUNCH-BOUND: {launch_overhead_ratio:.1%} overhead")

    return prof
```

**Key Metric**: If `(total_step_time - sum(kernel_times)) / total_step_time > 0.3`, the workload is **launch-bound**.

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 5/5 | Diagnostic only |
| **Complexity** | 1/5 | Standard profiling |
| **Success Probability** | **Very High** | Identifies opportunities |
| **Risk Level** | **None** | No production impact |

**Expected Impact**: Identifies 5-15% additional optimization opportunities
**Effort**: 2-4 hours

---

### 5.4 SwishLayerNorm Fusion Verification

**Problem**: SwishLayerNorm (`Y = X * Sigmoid(LayerNorm(X))`) involves multiple memory-bound operations that should be fused.

**Non-Fused Execution** (worst case):
| Kernel | Data Movement |
|--------|---------------|
| LayerNorm (mean/var) | 500MB read |
| LayerNorm (normalize) | 500MB read + 500MB write |
| Sigmoid | 500MB read + 500MB write |
| Multiply | 1000MB read + 500MB write |
| **Total** | **~3.5 GB transferred** |

**Fused Execution** (optimal):
| Single Kernel | Data Movement |
|---------------|---------------|
| Load X, compute all in registers, store Y | **~1.0 GB transferred** |

**Verification Steps**:
1. Profile SwishLayerNorm forward pass
2. Count separate kernels (should be 1 fused kernel, not 3-4)
3. If unfused, check if `torch.compile` is applied to the module

```python
# Verify fusion
with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = swish_layer_norm(input)

# Should show 1 kernel, not 3-4
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 4/5 | Verification + potential config change |
| **Complexity** | 2/5 | May need model restructuring |
| **Success Probability** | **High** | 2.5× speedup for fused SwishLayerNorm |
| **Risk Level** | **Low** | Well-tested optimization |

**Expected Impact**: 2.5× kernel speedup, 60-70% bandwidth reduction
**Effort**: 0.5-1 day

---

## 6. Phase 1: Kernel Fusion Opportunities

### 6.1 Further Loss Batching (30→15 kernels)

**Current State**: V2 achieved 221→30 kernels via fusion.

**Opportunity**: Group losses by type and compute in batched single-kernel calls.

```python
@torch.compile(mode="max-autotune", fullgraph=True)
def batched_multi_task_loss(preds_dict, targets_dict, weights_dict):
    """Batch all same-type losses into single kernel calls."""

    # Group by loss type
    bce_preds, bce_targets, bce_weights = [], [], []
    mse_preds, mse_targets, mse_weights = [], [], []

    for task_name, pred in preds_dict.items():
        if task_name.endswith('_bce'):
            bce_preds.append(pred)
            bce_targets.append(targets_dict[task_name])
            bce_weights.append(weights_dict[task_name])
        elif task_name.endswith('_mse'):
            mse_preds.append(pred)
            mse_targets.append(targets_dict[task_name])
            mse_weights.append(weights_dict[task_name])

    total_loss = 0.0

    # Single kernel for all BCE losses
    if bce_preds:
        bce_batched = torch.cat(bce_preds)
        targets_batched = torch.cat(bce_targets)
        loss_per_sample = F.binary_cross_entropy_with_logits(
            bce_batched, targets_batched, reduction='none'
        )
        # Apply per-task weighting (stays in GPU)
        idx = 0
        for i, pred in enumerate(bce_preds):
            n = len(pred)
            total_loss += bce_weights[i] * loss_per_sample[idx:idx+n].mean()
            idx += n

    return total_loss
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 3/5 | Requires loss function refactoring |
| **Complexity** | 3/5 | Must maintain task-specific behavior |
| **Success Probability** | **Medium-High** | Diminishing returns from 30 kernels |
| **Risk Level** | **Medium** | Numerical precision must match |

**Expected Impact**: 2-5% additional from 30→15 kernels
**Effort**: 2-3 days

---

### 6.2 Dynamic Loss Balancing Fusion

**Technique**: Fuse uncertainty-based weighting (Kendall et al.) or RLW normalization.

```python
@torch.compile(fullgraph=True)
def uncertainty_weighted_loss(losses: torch.Tensor, log_vars: torch.Tensor):
    """
    Fused: total_loss = sum(exp(-log_var[i]) * loss[i] + log_var[i])

    Fuses into 3 kernels instead of 2N+1:
    1. exp(-log_vars) -> precision
    2. precision * losses + log_vars -> weighted (fused mul+add)
    3. weighted.sum() -> total (reduction)
    """
    precision = torch.exp(-log_vars)
    weighted = precision * losses + log_vars
    return weighted.sum()

@torch.compile(fullgraph=True)
def rlw_weighted_loss(losses: torch.Tensor):
    """
    RLW: sum(loss / loss.detach())
    Auto-balances magnitudes without learned parameters.
    """
    return (losses / losses.detach()).sum()
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 3/5 | Requires loss balancing changes |
| **Success Probability** | **Medium** | Depends on current approach |
| **Risk Level** | **Medium** | May affect convergence |

**Expected Impact**: 1-3% + better gradient balancing
**Effort**: 2 days

### 6.3 GradScaler Constraint for Multi-Task Loss

**⚠️ CRITICAL**: When using mixed precision with multiple losses, a **single GradScaler must serve all losses**—using separate scalers corrupts the gradient accumulation math.

```python
# CORRECT: Single GradScaler for all losses
scaler = torch.cuda.amp.GradScaler()

# Combine losses FIRST, then scale once
combined_loss = task1_loss + task2_loss + task3_loss
scaler.scale(combined_loss).backward()

# Step and update only on accumulation boundaries
if (step + 1) % accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# INCORRECT: Multiple scalers (CORRUPTS GRADIENTS!)
scaler1 = torch.cuda.amp.GradScaler()
scaler2 = torch.cuda.amp.GradScaler()
scaler1.scale(task1_loss).backward()
scaler2.scale(task2_loss).backward()  # WRONG!
```

**Key Rules**:
1. Compute all losses with `torch.autocast` enabled
2. Sum/combine losses into single tensor
3. Scale the combined loss once
4. Call `scaler.step()` and `scaler.update()` only at accumulation boundaries

---

## 7. Phase 2: CUDA Graphs Integration

### 7.1 The Launch Overhead Problem

Kernel launch overhead: **5-10μs per operation**. With 50+ operations per layer:

| Kernel Duration | Launch Overhead | GPU Utilization |
|-----------------|-----------------|-----------------|
| 10 ms | 5 μs | **99.95%** |
| 100 μs | 5 μs | **95.2%** |
| 10 μs | 5 μs | **66.7%** |
| 1 μs | 5 μs | **16.7%** |

Meta's MAIProf found GPU idle exceeding **50% of training time** in unoptimized configurations.

### 7.2 torch.compile with reduce-overhead Mode

```python
# Full model compile with automatic CUDA graph management
model = torch.compile(model, mode="reduce-overhead")

# Or selective for static portions
@torch.compile(mode="reduce-overhead", fullgraph=True)
def forward_with_graphs(self, inputs, num_candidates_info):
    # NumCandidatesInfo provides static shapes - perfect for graphing
    ...
```

**V2 Compatibility**: NumCandidatesInfo pattern provides static shapes, enabling graph capture.

### 7.3 Manual CUDA Graph Capture for Hot Paths

```python
def create_training_step_graph(model, loss_fn, optimizer, sample_input, sample_target):
    """Capture entire training step as CUDA graph."""

    # Warmup (critical for cuBLAS/cuDNN initialization)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(11):  # 11 iterations for DDP
            optimizer.zero_grad(set_to_none=True)
            output = model(sample_input)
            loss = loss_fn(output, sample_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # Create static placeholders
    static_input = sample_input.clone()
    static_target = sample_target.clone()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_output = model(static_input)
        static_loss = loss_fn(static_output, static_target)
        static_loss.backward()
        optimizer.step()

    def replay(inputs, targets):
        static_input.copy_(inputs)
        static_target.copy_(targets)
        g.replay()
        return static_loss.clone()

    return replay
```

**MLPerf Results**: ~1.7× speedup for Mask R-CNN, ~1.12× for BERT at scale.

### 7.4 CUDA Graph Constraints

| Constraint | Description | Workaround |
|------------|-------------|------------|
| Static shapes | Batch size must be fixed | Pre-capture multiple graphs for shape buckets |
| No CPU-GPU sync | No `.item()`, no conditionals | Move all logic to GPU |
| No cudaMalloc | No dynamic allocation | Pre-allocate all buffers |
| Memory overhead | ~64 KB per kernel | Use `graph_pool_handle()` for sharing |

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 2/5 | May need to resolve graph breaks |
| **Complexity** | 4/5 | Dynamic control flow causes breaks |
| **Success Probability** | **Medium-High** | 1.3×-3× documented |
| **Risk Level** | **Medium** | May conflict with existing PT2 |

**Expected Impact**: 10-30% for launch-bound portions
**Effort**: 1-2 weeks

---

## 8. Phase 3: Attention Optimization

### 8.1 The Attention Memory Wall

For attention `softmax(QK^T/√d)V` at batch 256K:

**Score Matrix Size**: B × H × L × L

For B=256K, H=8, L=128:
- Materialized score matrix: **~67 GB** → **Immediate OOM**
- With broadcasting: **~8 KB** mask + streaming computation

### 8.2 HSTU Attention Architecture

**⚠️ IMPORTANT**: HSTU uses **custom Hammer Triton kernels**, NOT standard SDPA:

```python
# HSTU uses custom kernels
from hammer.v2.ops.triton.template.triton_hstu_attention import triton_hstu_mha
```

**However**, SDPA is used in other attention modules (e.g., cross-attention):
```python
# pytorch_modules_roo.py - SDPA IS used here
attn_output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
)
```

**Verification Strategy**:
1. Profile Hammer kernel performance for HSTU
2. Verify FlashAttention dispatch for SDPA-based cross-attention
3. Check `enable_gqa=True` for grouped-query attention

### 8.3 Critical: PyTorch Issue #154363 for GQA

**⚠️ PyTorch Issue #154363**: GQA (Grouped-Query Attention) broadcasting **without `enable_gqa=True`** may dispatch to *less* memory-efficient backends, paradoxically **increasing** memory usage.

```python
# WRONG: GQA without enable_gqa flag - may use inefficient backend
output = F.scaled_dot_product_attention(
    query,  # [B, num_heads, seq, head_dim]
    key,    # [B, num_kv_heads, seq, head_dim]  # num_kv_heads < num_heads
    value,
)

# CORRECT: Explicitly enable GQA for proper backend selection
output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    enable_gqa=True,  # Required for GQA memory efficiency!
)
```

**Action**: Audit all SDPA calls where `num_kv_heads != num_heads` and add `enable_gqa=True`.

### 8.4 FlashAttention Benefits

| Metric | Naive Attention | FlashAttention |
|--------|-----------------|----------------|
| HBM accesses | O(N²) | O(N²d²/M) |
| Memory complexity | O(N²) | O(N) |
| Throughput (A100) | ~25% theoretical | **50-73%** |
| Training speedup | baseline | **7.6×** (GPT-2) |
| HBM reduction | baseline | **up to 9×** |

### 8.5 FlexAttention for Custom Masks

If HSTU has custom masking patterns incompatible with FlashAttention:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def hstu_mask_mod(b, h, q_idx, kv_idx):
    """Custom HSTU masking logic - computed on-the-fly, not materialized."""
    return kv_idx <= q_idx  # Example: causal mask

block_mask = create_block_mask(
    mask_mod=hstu_mask_mod,
    B=batch_size, H=num_heads, Q_LEN=q_len, KV_LEN=kv_len,
    device=device,
)

# No [seq_len, seq_len] allocation!
output = flex_attention(query, key, value, block_mask=block_mask)
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 2/5 | Requires attention refactoring |
| **Complexity** | 3/5 | Custom mask function definition |
| **Success Probability** | **Medium-High** | Memory + compute savings |
| **Risk Level** | **Medium** | Numerical validation needed |

**Expected Impact**: Memory savings + 5-10% compute
**Effort**: 3-5 days

---

## 9. Phase 4: Advanced Triton Optimizations

### 9.1 Fused RMSNorm + Activation

**Single-pass kernel** avoiding second memory read:

```python
@triton.jit
def fused_rmsnorm_gelu_kernel(x_ptr, weight_ptr, out_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Single read
    x = tl.load(x_ptr + row_idx * N + cols, mask=cols < N)

    # RMS in registers
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    x_norm = x / rms

    # Weight and activation in registers
    w = tl.load(weight_ptr + cols, mask=cols < N)
    x_scaled = x_norm * w
    out = x_scaled * 0.5 * (1.0 + tl.math.erf(x_scaled / 1.41421356))

    # Single write
    tl.store(out_ptr + row_idx * N + cols, out, mask=cols < N)
```

**Liger Kernel Results**: 20% throughput improvement, 60% memory reduction.

### 9.2 H100 TMA (Tensor Memory Accelerator)

TMA provides dedicated hardware for async bulk data transfer:

```python
@triton.jit
def kernel_with_tma(a_desc_ptr, block_m, block_k, ...):
    offs_am = tl.program_id(0) * block_m
    # TMA descriptor handles address calculation and transfer
    a = tl._experimental_descriptor_load(
        a_desc_ptr, [offs_am, 0], [block_m, block_k], tl.float16
    )
```

| Metric | Without TMA | With TMA |
|--------|-------------|----------|
| Memory throughput | 910 GB/s | **1.45 TB/s** |
| **Improvement** | baseline | **59%** |

**Warp Specialization**: Divides warps into producer (data movement) and consumer (compute) roles. **10-15% speedup** on FlashAttention and FP8 GEMM.

The **Tawa automatic warp specialization framework** achieves **1.21× speedup** over baseline Triton kernels by automatically applying producer-consumer patterns.

```python
# Warp specialization autotune parameters
num_consumer_groups = 2
num_buffers_warp_spec = 3
```

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| **Easiness** | 1/5 | Experimental API, H100-specific |
| **Complexity** | 5/5 | Advanced hardware optimization |
| **Success Probability** | **Medium** | API still experimental |
| **Risk Level** | **High** | H100-only |

**Expected Impact**: 10-15% for memory-bound kernels on H100
**Effort**: 2-3 weeks

---

## 10. Hardware-Specific Considerations

### 10.1 H100 (GRANDTETON)

| Characteristic | Implication |
|----------------|-------------|
| 3.35 TB/s HBM bandwidth | Higher ceiling for memory-bound ops |
| TMA available | 59% memory throughput improvement potential |
| Warp specialization | 10-15% via producer-consumer pattern |
| Faster compute | Sync points relatively cheaper |

**Strategy**: Focus on **kernel fusion** over launch elimination.

### 10.2 B200 (GRANDTETON_B200)

| Characteristic | Implication |
|----------------|-------------|
| Even faster compute | Launch overhead relatively higher impact |
| Improved memory bandwidth | Same fusion techniques apply |

**Strategy**: Focus on **both fusion AND launch elimination**. CUDA graphs more valuable.

**Key Insight**: Faster GPUs make sync points and launch overhead **relatively more expensive**. B200 will benefit more from CUDA graphs than H100.

### 10.3 Mixed Fleet Configuration

```python
def get_optimal_config():
    device_name = torch.cuda.get_device_name()

    if "H100" in device_name:
        return {
            "use_tma": True,
            "warp_specialization": True,
            "cuda_graphs_priority": "medium",
            "focus": "kernel_fusion",
        }
    elif "B200" in device_name:
        return {
            "use_tma": True,
            "warp_specialization": True,
            "cuda_graphs_priority": "high",
            "focus": "launch_elimination",
        }
    else:  # A100 and earlier
        return {
            "use_tma": False,
            "warp_specialization": False,
            "cuda_graphs_priority": "high",
            "focus": "cuda_graphs",
        }
```

---

## 11. Profiling & Validation Methodology

### 11.1 Detecting Materialization Events

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)

# Look for 500MB allocations followed by aten::copy_ or aten::contiguous
```

### 11.2 Memory Bandwidth Analysis

Use Nsight Compute:
- Check "DRAM Throughput"
- If throughput saturated but FLOPs low → **strictly bandwidth-bound**

### 11.3 Cost Comparison Table

| Metric | Pattern A (Materialized) | Pattern B (Broadcasted) |
|--------|-------------------------|------------------------|
| Allocation | 500 MB | 0 MB |
| HBM Read | 1000 MB | 500 MB |
| HBM Write | 500 MB | 500 MB |
| L2 Cache Hit | Low (thrashing) | High (99%) |
| Latency (A100) | ~1.0 ms | ~0.65 ms |
| **Difference** | **~35% Slower** | Baseline |

### 11.4 DeepSpeed Deep Fusion Production Benchmarks

Reference benchmarks from DeepSpeed Deep Fusion for transformer optimization:

| Fusion Pattern | Speedup |
|----------------|---------|
| Input LayerNorm + QKV GEMM + bias adds | **1.5×** over unfused cuBLAS |
| Attention with implicit matrix transformation | **2.9×** |
| Intermediate FF + LayerNorm + bias + residual + GELU | **3.0×** |
| MoE kernel optimization | **>6× reduction** in MoE-related latency |

**LinkedIn's Liger Kernel (Triton-based)**:

| Metric | Improvement |
|--------|-------------|
| Multi-GPU training throughput | **20% improvement** |
| Memory reduction | **60%** from fused operations |

Includes fused RMSNorm, RoPE, SwiGLU, and FusedLinearCrossEntropy.

### 11.5 Measurement Best Practices

1. Use **CUDA events** for timing (not `time.time()`)
2. Warm up for **10+ iterations**
3. Run **50+ iterations**, take median
4. Profile with **Nsight Systems** for timeline visualization

**Key Metrics**:
- Kernel count per training step
- SM occupancy (target >50%)
- Memory bandwidth utilization
- Samples/second throughput

---

## 12. Prioritized Implementation Matrix

| Priority | Optimization | QPS Gain | Easiness | Success Prob. | Risk | Effort |
|----------|--------------|----------|----------|---------------|------|--------|
| **P0** 🍎 | Materialization audit | Prevents 33%+ degradation | 5/5 | Very High | None | 2-4h |
| **P0** 🍎 | Inductor config optimization | 5-10% | 5/5 | Med-High | Low | 1-2h |
| **P0** 🍎 | Kernel count profiling | Identifies issues | 5/5 | Very High | None | 2-4h |
| **P0** 🍎 | SwishLayerNorm fusion verification | 2.5× kernel speedup | 4/5 | High | Low | 0.5d |
| **P1** | HSTU/SDPA attention verification | Variable | 3/5 | Medium | Low | 3-5d |
| **P1** | Further loss batching (30→15) | 2-5% | 3/5 | Med-High | Medium | 2-3d |
| **P1** | reduce-overhead mode | 10-30% | 2/5 | Med-High | Medium | 1-2w |
| **P2** | Manual CUDA graphs | 20-50% | 2/5 | Medium | Medium | 1w |
| **P2** | FlexAttention masks | 5-10% | 2/5 | Med-High | Medium | 3-5d |
| **P3** | Fused RMSNorm+activation (Triton) | 5-10% | 1/5 | High | Medium | 1-2w |
| **P3** | H100 TMA optimization | 10-15% | 1/5 | Medium | High | 2-3w |

🍎 = Low-hanging fruit - do these FIRST!

---

## 13. Recommended Execution Timeline

### Week 1: Low-Hanging Fruits & Diagnostics

| Day | Task | Expected Outcome |
|-----|------|------------------|
| Day 1 AM | **Materialization audit** (search for dangerous patterns) | Identify risk areas |
| Day 1 PM | **Kernel count profiling** | Total kernel count, launch-bound assessment |
| Day 2 | **Inductor config optimization** | 5-10% potential |
| Day 3 | **SwishLayerNorm fusion verification** | Confirm 2.5× speedup or identify gaps |
| Day 4-5 | Address issues found in profiling | Variable |

### Week 2: CUDA Graphs Preparation

| Day | Task | Expected Outcome |
|-----|------|------------------|
| Day 1-2 | Test `reduce-overhead` mode | Identify graph breaks |
| Day 3-4 | Resolve graph breaks or identify static portions | Enable CUDA graphs |
| Day 5 | Benchmark with graphs enabled | 10-30% for launch-bound |

### Week 3: Attention & Loss Optimization

| Day | Task | Expected Outcome |
|-----|------|------------------|
| Day 1-2 | HSTU/SDPA attention verification | Understand current state |
| Day 3-4 | Further loss batching (30→15 kernels) | 2-5% additional |
| Day 5 | Evaluate FlexAttention for custom masks | Memory + 5-10% |

### Week 4+: Advanced Optimizations

| Week | Task | Expected Outcome |
|------|------|------------------|
| Week 4 | Manual CUDA graphs for hot paths | 20-50% for captured portions |
| Week 5-6 | Custom Triton kernels (if justified) | 5-10% per pattern |
| Week 7+ | H100 TMA optimization (if justified) | 10-15% for memory-bound |

---

## Appendix A: Diagnostic Commands

### Materialization Pattern Search

```bash
# Critical: expand before device transfer
grep -rn "\.expand.*\.to\|\.expand.*\.cuda\|\.expand.*\.cpu" --include="*.py"

# High risk: expand before contiguous
grep -rn "\.expand.*\.contiguous" --include="*.py"

# Medium risk: expand before reshape
grep -rn "\.expand.*\.reshape" --include="*.py"
```

### Enable Inductor Logging

```python
import logging
torch._logging.set_logs(inductor=logging.DEBUG)

# Or via environment variable
# TORCH_LOGS="graph_breaks,output_code" python train.py
```

### Kernel Count Analysis

```python
def count_and_categorize_kernels(prof):
    """Analyze kernel distribution from profiler output."""
    cuda_events = [e for e in prof.events() if e.device_type == 1]

    # Categorize by operation type
    categories = {
        'elementwise': 0,
        'reduction': 0,
        'gemm': 0,
        'attention': 0,
        'other': 0
    }

    for e in cuda_events:
        name = e.name.lower()
        if any(x in name for x in ['add', 'mul', 'relu', 'sigmoid', 'gelu']):
            categories['elementwise'] += 1
        elif any(x in name for x in ['sum', 'mean', 'norm', 'softmax']):
            categories['reduction'] += 1
        elif any(x in name for x in ['gemm', 'mm', 'linear']):
            categories['gemm'] += 1
        elif any(x in name for x in ['attention', 'flash', 'sdpa']):
            categories['attention'] += 1
        else:
            categories['other'] += 1

    return categories
```

---

## Appendix B: Files to Modify

### P0 - Low-Hanging Fruits

| File | Change |
|------|--------|
| Training config / model init | Add inductor configuration settings |
| All files with `.expand()` | Audit for materialization risks |
| SwishLayerNorm module | Verify fusion, potentially restructure |

### P1 - Medium Effort

| File | Change |
|------|--------|
| Loss computation modules | Further batching optimization |
| Model forward pass | Test `mode="reduce-overhead"` |
| HSTU attention modules | Verify Hammer kernel settings |

### P2-P3 - High Effort

| File | Change |
|------|--------|
| New file: `triton_kernels.py` | Custom fused kernels |
| HSTU attention | FlexAttention integration |
| Training loop | CUDA graph capture points |

---

## Appendix C: Expected Total Impact

| Phase | Optimizations | Cumulative Impact |
|-------|---------------|-------------------|
| **Baseline (V2)** | Kernel fusion, PT2, NumCandidatesInfo | ~65-75% over V1 |
| **+ Phase 0** | Materialization audit, Inductor config, SwishLayerNorm | +15-30% |
| **+ Phase 1** | Loss batching, reduce-overhead | +10-25% |
| **+ Phase 2** | CUDA graphs, attention optimization | +10-20% |
| **+ Phase 3** | Custom Triton, TMA | +10-20% |
| **Total Potential** | All optimizations | **~50-80% over V2** |

**Note**: Gains are not strictly additive. Profile first to identify actual bottlenecks.

---

## Appendix D: Critical Thinking & Limitations

### What Could Go Wrong

1. **Inductor config changes**: `epilogue_fusion=True` was explicitly disabled in some configs—there may be historical regressions. **Test thoroughly before enabling.**

2. **CUDA graphs with dynamic shapes**: NumCandidatesInfo helps but edge cases with variable sequence lengths may cause graph re-compilation overhead.

3. **Custom Triton kernels**: Numerical precision differences possible. **Validate against PyTorch reference implementations.**

4. **H100 TMA**: Experimental API may change. Code may break on future PyTorch/Triton versions.

5. **Mixed-precision interactions**: Accidental materialization combined with autocast can cause unexpected dtype conversions.

### Assumptions Made

1. Batch size 256K is representative of production workloads
2. Hidden dimension ~1024 is typical
3. V2 codebase already uses bfloat16 where applicable
4. NumCandidatesInfo provides sufficiently static shapes for CUDA graphs

### Areas Requiring Further Investigation

1. **Why is `epilogue_fusion` disabled?** Historical context needed.
2. **What is Hammer kernel performance vs. FlashAttention?** Profile comparison needed.
3. **What is the actual kernel count per training step?** Profiling needed.
4. **Are there existing materialization bugs?** Audit needed.

---

## Appendix E: References

1. **Q4 Merged Research**: `result_Q4_merged.md` (source of most findings)
2. **NVIDIA Hopper Architecture**: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
3. **TorchInductor Deep Dive**: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
4. **FlashAttention Paper**: https://arxiv.org/abs/2205.14135
5. **Liger Kernel**: https://github.com/linkedin/liger-kernel
6. **PyTorch CUDA Graphs Blog**: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
7. **FlexAttention Blog**: https://pytorch.org/blog/flexattention/

---

*Document Version: 2.0*
*Revised: 2026-01-30*
*Based on: Q4 Deep Research (Merged), Memory Bandwidth Optimization Focus*
*Key Insight: At batch 256K, memory bandwidth—not compute—is the dominant constraint*
