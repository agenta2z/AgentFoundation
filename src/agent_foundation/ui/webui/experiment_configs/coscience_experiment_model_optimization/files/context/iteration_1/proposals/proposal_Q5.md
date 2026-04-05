# Proposal Q5: TorchRec Training Pipeline & Mixed-Precision Optimization

## Executive Summary

This proposal synthesizes comprehensive research on TorchRec pipeline optimization and presents a **risk-stratified, evidence-based** set of recommendations for achieving the targeted **5-10% aggregate QPS improvement**. Building on deep analysis of PyTorch internals, we identify critical **silent performance regressions** that may be degrading current training throughput—most notably the **SDPA Float32 Trap** and **redundant explicit casting overhead**.

> **🎯 Critical Insight from Research:**
> FSDP analysis reveals that **autocast as a strategy** incurs **130 `_to_copy` calls** vs only **5 for FSDP**—a **26× difference**. This comparison is between two different mixed-precision strategies (autocast vs FSDP's approach), not about explicit vs implicit casting within autocast. Removing redundant explicit casts provides **marginal CPU-side overhead reduction** (Python dispatch + kernel launch latency), but the primary value is code cleanliness and avoiding unnecessary operations.

> **⚠️ Important Caveats:**
> - Impact estimates derive from PyTorch internals analysis and benchmark studies; actual gains depend on model-specific compute/memory ratios
> - The MTML ROO model uses **DPP (Data Platform Pipeline)** infrastructure—parameter names differ from standard PyTorch DataLoader
> - Some optimizations (SDD Lite, `gradient_as_bucket_view`) are **already implemented** per commit `f220c6d5`

### Target Model Specification

| Attribute | Value |
|-----------|-------|
| **Target Model** | Main Feed MTML ROO |
| **Config File** | `models/main_feed_mtml/conf/mast_roo_trainer_config.py` |
| **DPP `client_thread_count`** | 24 |
| **DPP `prefetch_queue_size`** | 32 |
| **Hardware** | NVIDIA A100 (SM80, Ampere architecture) |

---

## Key Research Findings

### Critical Performance Insights

| Finding | Impact | Source |
|---------|--------|--------|
| Explicit `.to(bfloat16)` in autocast is **redundant** | Marginal CPU overhead (Python dispatch + kernel launch) | Section 6.1-6.2 |
| SDPA silently falls back to O(N²) Math kernel on float32 | **2×+ performance degradation** | PyTorch SDPA dispatcher |
| Triton kernels are **opaque to autocast** | Potential garbage data without errors | Triton compilation barrier |
| FlashAttention requires bf16/fp16, head_dim ≤256, contiguous last dim | Silent Math fallback if unmet | FlashAttention-2 constraints |
| SDD Pipeline overlaps All-to-All with compute | **20-40% QPS improvement** | TorchRec documentation |
| Cache dtype **before** autocast blocks | **Output** dtype changes; **input dtype unchanged** | Section 7.2 |
| `torch.compile` can break autocast with Triton kernels | GEMM stays fp32 instead of bf16 | Section 10.6 |
| CUDA grid limit of **65,535 blocks** | `batch_size × num_heads` constraint | Section 9.4 |
| Sample packing forces Memory-Efficient fallback | 2× slower than FlashAttention | Section 8.4 |

### A100 Architecture Context

Understanding **WHY** these optimizations matter at the hardware level:

| Format | Sign | Exponent | Mantissa | Dynamic Range | Precision |
|--------|------|----------|----------|---------------|-----------|
| **FP32** | 1 bit | 8 bits | 23 bits | ~1e-38 to 3e38 | High |
| **FP16** | 1 bit | 5 bits | 10 bits | ~6e-5 to 6e4 | Medium |
| **BF16** | 1 bit | 8 bits | 7 bits | ~1e-38 to 3e38 | Low |

**Key Insight**: BF16 shares the **same dynamic range as FP32** (8-bit exponent), eliminating the need for loss scaling. However, **accumulation ops (Softmax, LayerNorm) must run in FP32** to preserve numerical stability.

**Tensor Core Dispatch**: A100 optimal performance requires Tensor Cores. Float32 inputs → CUDA Cores (slower). The primary role of `torch.autocast` is ensuring data reaches MMA-heavy operations in bf16/fp16 to unlock Tensor Core usage.

### TorchRec Pipeline Comparison

| Pipeline | Memory Overhead | Expected Gain | Use Case |
|----------|-----------------|---------------|----------|
| **Base** | 1× batch | Baseline | Debugging |
| **SDD** | ~3× batch | 20-40% QPS | Production multi-GPU |
| **SDD Lite** | ~1.01× batch | 4-5% QPS | Memory-constrained ✅ **Current** |
| **Prefetch SDD** | ~4× batch + cache | Larger models | UVM-cached embeddings |
| **Fused SDD** | ~3× batch | +5-10% QPS | Heavy optimizers (Shampoo) |

### Communication Architecture (for Advanced Scaling)

#### LazyAwaitable Enables Deferred Execution

TorchRec uses `LazyAwaitable` types to delay result computation as long as possible. Operations return awaitable handles immediately, with actual computation/communication triggered only when results are needed. This decouples data production from consumption, enabling maximum overlap.

#### Sharding Strategy Affects Communication Patterns

| Strategy | Communication Pattern | Best For |
|----------|----------------------|----------|
| **Table-wise (TW)** | All-to-All to owning device | Few large tables |
| **Row-wise (RW)** | Row-based routing | Load balancing large tables |
| **Column-wise (CW)** | Concat after All-to-All | Wide embedding dimensions |
| **Grid (2D)** | Complex multi-stage | Very large tables at scale |

#### 2D Sparse Parallelism for 1000+ GPU Scale

Meta's **DMPCollection** implements 2D parallelism combining model and data parallelism:

```python
# Process group topology (2 nodes, 4 GPUs each)
Sharding Groups: [0,2,4,6] and [1,3,5,7]  # Model parallel
Replica Groups:  [0,1], [2,3], [4,5], [6,7]  # Data parallel
```

**Key optimizations**: Replica ranks placed on same node for high-bandwidth intra-node AllReduce; sharding over smaller rank groups reduces All-to-All latency. Critically, 2D parallel synchronizes **weights, not gradients**, enabling the fused optimizer optimization.

### Already Implemented (No Action Required)

| Optimization | Status | Impact |
|--------------|--------|--------|
| SDD Lite Pipeline | ✅ Done | +4-5% QPS, +1% memory |
| Inplace Data Copy | ✅ Done | -3% peak memory |
| Pinned Memory (`pin_memory=True`) | ✅ Done | 16% faster transfers |
| `gradient_as_bucket_view=True` | ✅ Done | ~4GB savings |
| NumCandidatesInfo sync consolidation | ✅ Done | Single GPU→CPU sync |

---

## Production Priority Order

Based on research Section 14.7, the recommended priority for production deployment:

| Priority | Optimization | Expected Impact |
|----------|--------------|-----------------|
| **1** | SDD pipeline | 20-40% QPS improvement |
| **2** | Pinned memory with prefetching | 2-4% improvement |
| **3** | Fused optimizer | Memory efficiency |
| **4** | Buffer pre-allocation | Prevents fragmentation |
| **5** | Mixed precision | 50% memory savings |

Monitor via PyTorch Profiler for remaining bottlenecks in your specific hardware/model configuration.

---

## Assessment Framework

| Criteria | Description |
|----------|-------------|
| **Technical Analysis** | Mechanism of improvement, underlying PyTorch/CUDA behavior |
| **Easiness** | 1-5 scale (5=trivial config change) |
| **Complexity** | 1-5 scale (1=simple, 5=architectural change) |
| **Risk Level** | Low/Medium/High—potential for correctness regressions |
| **Success Estimation** | % likelihood of achieving expected gains |
| **Verification Method** | How to validate the optimization |
| **🍎 Low-Hanging Fruit?** | Easiness ≥4, Complexity ≤2, Risk=Low, Success ≥80% |

---

## 1. Phase 1: Critical Optimizations 🔴 (High Impact, Week 1)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 🔴1 | **Remove redundant explicit casts in autocast** | **4/5** | **2/5** | Low | 90% | Reduce kernel launches | 1 day |
| 🔴2 | **SDPA backend verification & guard** | **4/5** | **2/5** | Medium | 85% | Prevent 2× slowdown | 1 day |
| 🍎3 | `zero_grad(set_to_none=True)` | **5/5** | **1/5** | Low | 95% | Minor speedup | 0.5 day |
| 🍎4 | Cached dtype pattern implementation | **4/5** | **2/5** | Low | 85% | Correct precision handling | 1 day |

**Total Phase 1 Effort: ~3.5 days**

---

### 🔴 CRITICAL-1: Remove Redundant Explicit Casts in Autocast Blocks

#### Technical Analysis

**The Core Issue:** Explicit `.to(bfloat16)` calls inside `torch.autocast` blocks are **functionally redundant**—the autocast dispatcher handles casting automatically with weight caching.

**Research Evidence:**
```python
# From autocast_mode.cpp
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
    if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
        return arg.to(to_type);  // Cast only if needed
    }
    return arg;  // NO CAST if already correct dtype
}
```

**Hidden Costs of Explicit Casting:**
1. **Python Interpreter Overhead**: Each `to()` call requires dispatch, argument parsing
2. **Kernel Launch Latency**: 3-10μs per launch, accumulates in tight loops
3. **Redundant Work**: If input is already bfloat16, autocast does nothing; explicit cast still launches Python dispatch

> **Note on the 26× Statistic:** Research Section 6.3 shows autocast incurs **130 `_to_copy` calls** vs **5 for FSDP**—but this compares **two different mixed-precision strategies** (autocast vs FSDP's approach), NOT explicit vs implicit casting within autocast. Removing redundant explicit casts provides marginal CPU overhead reduction, not 26× improvement.

**When Explicit Casts ARE Needed (Exceptions):**
- **SDPA inputs** (see CRITICAL-2)—guard rails against Math fallback
- **Triton kernel inputs** (see Phase 2)—opaque to autocast
- **Non-eligible custom ops** not on autocast op list

#### Implementation

**Step 1: Identify redundant casts**
```bash
# Search pattern in codebase
grep -rn "\.to(torch\.bfloat16)" --include="*.py" | grep -i "autocast"
grep -rn "\.to(dtype=torch\.bfloat16)" --include="*.py"
```

**Step 2: Audit and remove**
```python
# BEFORE (redundant)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = x.to(torch.bfloat16)  # ← REMOVE: autocast handles this
    output = model(x)

# AFTER (correct)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)  # Autocast casts automatically for eligible ops
```

**Critical Exception—KEEP casts for:**
```python
# KEEP: Guard rail for SDPA FlashAttention dispatch
with torch.autocast('cuda', dtype=torch.bfloat16):
    q = q.to(torch.bfloat16)  # KEEP—ensures FlashAttention
    k = k.to(torch.bfloat16)  # KEEP
    v = v.to(torch.bfloat16)  # KEEP
    attn = F.scaled_dot_product_attention(q, k, v)
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5**—Search and remove pattern |
| Complexity | **2/5**—Must identify exceptions (SDPA, Triton) |
| Risk Level | **Low**—Removing redundant code |
| Dependencies | None |
| Success Estimation | **90%** |
| Expected Impact | Reduced kernel launches, lower CPU overhead |
| Implementation Effort | **1 day** |
| Verification | Profile kernel launch counts before/after |

#### Verification Protocol

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Before optimization
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        x = x.to(torch.bfloat16)  # Redundant
        output = model(x)

# Count _to_copy operations
to_copy_count = sum(1 for e in prof.key_averages() if '_to_copy' in e.key)
print(f"_to_copy calls: {to_copy_count}")
```

---

### 🔴 CRITICAL-2: SDPA Backend Verification & FlashAttention Guards

#### Technical Analysis

**The Float32 Trap:** PyTorch's `F.scaled_dot_product_attention` selects backends with strict priority ordering. If inputs are **float32**, FlashAttention is **silently skipped**:

| Priority | Backend | Float32 Support | Memory | Speed |
|----------|---------|-----------------|--------|-------|
| 1 | FlashAttention | ❌ **No** | O(N) | Fastest |
| 2 | Memory-Efficient | ✅ Yes | O(N) | Medium |
| 3 | **Math (Fallback)** | ✅ Yes | **O(N²)** | **2×+ slower** |

**Common Fallback Triggers:**
- Float32 inputs (most common)
- Mixed dtypes (Q, K, V must match)
- Custom attention masks (`is_causal=True` required for Flash)
- Head dimension not divisible by 8 or >256
- Non-contiguous last dimension (after `transpose`)
- **Sample Packing Problem**: Flash Attention doesn't support block-wise causal attention masks needed for sequence packing, forcing fallback to Memory-Efficient kernel (2× slower)

**FlashAttention-2 Constraints (A100):**

| Constraint | Requirement |
|------------|-------------|
| **Dtype** | fp16 or bf16 only |
| **Head dim** | Multiple of 8, ≤256 |
| **The "80" Problem** | Models using `head_dim=80` may fail on earlier FlashAttention versions (pre-2.4.x). **Workaround:** Use `head_dim=64` or `head_dim=128` if model architecture allows, or update to FlashAttention ≥2.4. |
| **Contiguity** | Last dim stride == 1 |
| **Mask** | Only `is_causal=True`, no custom masks |
| **Architecture** | SM80+ (A100, H100) |
| **CUDA Grid Limit** | `batch_size × num_heads` ≤ 65,535 blocks (**silent failure**) |

> **⚠️ CRITICAL: CUDA Grid Limit is a Silent Failure Mode**
>
> If `batch_size × num_heads > 65,535`, FlashAttention silently degrades or produces incorrect results without raising an error. For large batch training (e.g., 256K batch sizes), verify this constraint is met:
> ```python
> assert batch_size * num_heads <= 65535, f"CUDA grid limit exceeded: {batch_size} × {num_heads} = {batch_size * num_heads}"
> ```

> **⚠️ PRODUCTION WARNING: Sample Packing Limitation**
>
> **Sample packing** (combining multiple sequences into one to maximize GPU utilization) requires **block-wise causal attention masks**. FlashAttention does **NOT** support this—it forces fallback to the Memory-Efficient kernel, which is **2× slower**.
>
> **Impact for production training:**
> - If your training uses sample packing for efficiency, FlashAttention cannot be used
> - You will be limited to Memory-Efficient attention (still O(N) memory, but slower)
> - Consider whether the packing efficiency gains outweigh the 2× attention slowdown
>
> **Current workaround:** None for FlashAttention. Either disable sample packing or accept Memory-Efficient kernel performance.

#### Implementation

**Step 1: Add backend verification function**
```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPAParams
import torch.nn.attention as attn

def verify_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Verify FlashAttention constraints are met."""
    params = SDPAParams(q, k, v, attn_mask=None, dropout=0.0, is_causal=False)

    can_use = torch.backends.cuda.can_use_flash_attention(params, debug=True)
    if not can_use:
        print(f"FlashAttention REJECTED:")
        print(f"  dtype: {q.dtype} (need fp16/bf16)")
        print(f"  head_dim: {q.shape[-1]} (need ≤256, multiple of 8)")
        print(f"  contiguous: {q.stride(-1) == 1}")
        # Check CUDA grid limit
        batch_size, num_heads = q.shape[0], q.shape[1]
        if batch_size * num_heads > 65535:
            print(f"  CUDA grid limit exceeded: {batch_size} × {num_heads} = {batch_size * num_heads} > 65535")
    return can_use

def enforce_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Force FlashAttention, raise if constraints not met."""
    with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)
```

**Step 2: Add dtype guard before SDPA calls**
```python
def attention_with_flash_guard(q, k, v, is_causal=False):
    """SDPA wrapper ensuring FlashAttention dispatch."""
    # Ensure bf16 for FlashAttention compatibility
    target_dtype = torch.bfloat16

    # Cast if needed (this IS necessary for SDPA, unlike generic ops)
    if q.dtype != target_dtype:
        q = q.to(target_dtype)
    if k.dtype != target_dtype:
        k = k.to(target_dtype)
    if v.dtype != target_dtype:
        v = v.to(target_dtype)

    # Ensure contiguity (transpose can break this)
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()

    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
```

**Step 3: Add runtime verification in debug mode**
```python
# In training config or debug flag
VERIFY_FLASH_ATTENTION = True  # Enable during testing

def attention_forward(q, k, v, is_causal=False):
    if VERIFY_FLASH_ATTENTION:
        if not verify_flash_attention(q, k, v):
            raise RuntimeError(
                "FlashAttention constraints not met! "
                "Performance will degrade significantly."
            )

    return attention_with_flash_guard(q, k, v, is_causal)
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5**—Wrapper functions |
| Complexity | **2/5**—Identify all SDPA call sites |
| Risk Level | **Medium**—Changes attention behavior |
| Dependencies | None |
| Success Estimation | **85%** |
| Expected Impact | Prevent 2×+ slowdown from Math fallback |
| Implementation Effort | **1 day** |
| Verification | Use `sdpa_kernel` context manager to force backend |

#### Verification Protocol

```python
# Method 1: Force backend and check for error
def test_flash_attention_availability(q, k, v):
    try:
        with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
            _ = F.scaled_dot_product_attention(q, k, v)
        print("✅ FlashAttention available")
    except RuntimeError as e:
        print(f"❌ FlashAttention rejected: {e}")

# Method 2: Nsight profiling for kernel names
# FlashAttention: pytorch_flash::flash_fwd_kernel
# Memory-Efficient: fmha_*, efficient_attention_*
# Math fallback: aten::bmm, aten::softmax (SEPARATE softmax = Math)
```

---

### 🍎 LHF-3: `zero_grad(set_to_none=True)`

#### Technical Analysis

Standard `zero_grad()` writes zeros to gradient memory. `set_to_none=True` uses assignment instead—faster and avoids memory operations.

```python
# Standard: Allocates and writes zeros
optimizer.zero_grad()

# Optimized: Just sets reference to None
optimizer.zero_grad(set_to_none=True)
```

**From Research:** "Uses assignment instead of memory-writing zeroes—faster and avoids unnecessary memory operations"

#### Implementation

```python
# Locate optimizer.zero_grad() in training loop
# Add set_to_none=True parameter

# Before
optimizer.zero_grad()

# After
optimizer.zero_grad(set_to_none=True)
```

**Target File:** Training framework / `train_pipeline.py`

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **5/5**—Single parameter |
| Complexity | **1/5**—No restructuring |
| Risk Level | **Low**—Standard PyTorch API |
| Success Estimation | **95%** |
| Expected Impact | Minor speedup |
| Implementation Effort | **0.5 day** |

---

### 🍎 LHF-4: Cached Dtype Pattern Implementation

#### Technical Analysis

**The Problem:** Autocast **output** tensors may have different dtypes than inputs, but **input tensors remain unchanged**:

```python
x = torch.randn(10, 10, device='cuda')  # float32
with torch.autocast(device_type='cuda', dtype=torch.float16):
    y = torch.mm(x, x)
    print(y.dtype)  # torch.float16 — OUTPUT CHANGED!
    print(x.dtype)  # torch.float32 — INPUT UNCHANGED (autocast doesn't modify tensor .dtype)
```

**Clarification:** Autocast does **NOT** modify input tensor `.dtype` attributes—accessing `tensor.dtype` always returns the actual storage dtype. What changes is the **output tensor's dtype** from operations inside the block.

**Why Cache BEFORE Autocast:**
1. **Module Contracts**: Users expect `f(x)` to return same dtype as input
2. **Gradient Safety**: Backward pass uses forward dtype; mixed precision in gradients → underflow
3. **Prevent confusion**: Caching outside makes intent clear and avoids accessing output dtype by mistake

#### Implementation

```python
def forward(self, seq_embeddings, ...):
    # Cache dtype BEFORE autocast block (captures input dtype)
    input_dtype = seq_embeddings.dtype

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16_training):
        out = self.cross_attn(seq_embeddings, ...)
        # NOTE: seq_embeddings.dtype is STILL float32 here
        # BUT out.dtype is bfloat16 (output was demoted)

    # Restore original dtype for API consistency
    out = out.to(input_dtype)

    return out
```

**Anti-Pattern (AVOID):**
```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    out = cross_attn(seq_embeddings, ...)
    # NOTE: seq_embeddings.dtype is STILL float32 here (input unchanged)
    # BUT: out.dtype is bfloat16 (output was demoted)

# MISTAKE: Using output dtype for restoration (common confusion)
out = out.to(out.dtype)  # ← No-op! out.dtype is already bfloat16

# CORRECT: Cache INPUT dtype BEFORE autocast, then restore
```

**Key Clarification:** Autocast does **NOT** modify the input tensor's `.dtype` attribute—`seq_embeddings.dtype` remains float32 throughout. What changes is the **output tensor's dtype** (`out.dtype` becomes bfloat16). The anti-pattern is using the output dtype for restoration instead of caching the input dtype beforehand.

**Best Practices:**
1. Cache at module boundary (e.g., `forward` of Transformer Block)
2. Avoid "cast thrashing"—don't nest pattern deeply
3. Consider keeping BF16 until final loss computation in memory-constrained scenarios

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5**—Add caching at boundaries |
| Complexity | **2/5**—Identify correct boundaries |
| Risk Level | **Low**—Preserves API contracts |
| Success Estimation | **85%** |
| Expected Impact | Correct precision handling, prevents gradient issues |
| Implementation Effort | **1 day** |

---

## 2. Phase 2: Medium Priority (Week 2-3)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 5 | Triton kernel `custom_fwd` integration | 3/5 | 3/5 | Medium | 75% | Correctness + speed | 2-3 days |
| 6 | DPP parameter tuning | 4/5 | 2/5 | Low | 80% | Data loading | 1 day |
| 7 | Buffer pre-allocation warmup | 4/5 | 2/5 | Low | 80% | Prevent OOM | 1 day |
| 8 | Post-accumulate gradient hooks | 3/5 | 3/5 | Medium | 70% | ~20% peak memory | 3-4 days |

**Total Phase 2 Effort: ~8-10 days**

---

### Proposal 5: Triton Kernel `custom_fwd` Integration

#### Technical Analysis

**Critical Issue:** Triton kernels are **opaque to autocast**. The dispatcher cannot inspect JIT-compiled PTX code.

**Danger Scenario:**
```python
x = torch.randn(1024, 1024, device='cuda')  # float32

with torch.autocast(device_type='cuda', dtype=torch.float16):
    # Triton kernel receives float32—autocast doesn't cast non-PyTorch ops!
    output = my_triton_kernel_wrapper(x)  # Sees float32
```

**Data Reinterpretation Risk:**
- Triton pointers treat memory as raw bytes
- If kernel expects float16 (16-bit) but receives float32 (32-bit)
- Kernel reads lower 16 bits as element 0, upper 16 bits as element 1
- **Result: Garbage data without errors**

**torch.compile Interaction Issue (NEW):**
```python
# torch.compile can break autocast with Triton kernels
# Some GEMM kernels stay in fp32 instead of being cast to bf16
# Particularly affects aten::bmm operations
```

**Solution for torch.compile:**
```python
# Use this config for exact eager behavior with compiled Triton
torch._inductor.config.emulate_precision_casts = True
```

#### Implementation

```python
from torch.amp import custom_fwd, custom_bwd

class TritonOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.bfloat16)
    def forward(ctx, x):
        """
        Guarantees:
        1. 'x' is cast to bfloat16 BEFORE execution
        2. Autocast is disabled inside body (no double-casting)
        """
        return my_triton_kernel(x)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        # Handle backward with same precision
        return my_triton_kernel_backward(grad_output)

# Usage
output = TritonOp.apply(x)  # Safe with autocast
```

**Alternative for torch.compile compatibility (PyTorch 2.6+):**
```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::my_op", mutates_args={})
def my_op(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    wrap_triton(my_kernel)[grid](x, out, x.numel())
    return out
```

**Testing with Mixed Precision:**
```python
# Test Triton kernels thoroughly with mixed precision enabled
# Compiled Triton kernels show numerical variations due to:
# - Different reduction orders
# - Cast elision optimizations

# For debugging, enable exact eager behavior:
torch._inductor.config.emulate_precision_casts = True
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **3/5**—Requires identifying all Triton kernels |
| Complexity | **3/5**—autograd.Function pattern |
| Risk Level | **Medium**—Changes kernel invocation |
| Dependencies | Triton kernel inventory |
| Success Estimation | **75%** |
| Expected Impact | Correctness + consistent BF16 performance |
| Implementation Effort | **2-3 days** |

---

### Proposal 6: DPP Parameter Tuning + Data Prefetcher Pattern

#### Technical Analysis

MTML ROO uses **DPP (Data Platform Pipeline)**, not standard PyTorch DataLoader:

| PyTorch DataLoader | DPP Equivalent | Current Value |
|-------------------|----------------|---------------|
| `prefetch_factor` | `prefetch_queue_size` | 32 |
| `num_workers` | `client_thread_count` | 24 |

**Research Finding:** Optimal `prefetch_factor` is 2-4 for standard DataLoader. `prefetch_queue_size=32` may cause memory pressure.

#### Data Prefetcher Pattern (Reference)

For standard DataLoader scenarios, the data prefetcher pattern achieves **45% more batches per second**:

```python
class DataPrefetcher:
    """Overlap CPU→GPU transfer with computation using separate CUDA stream."""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

# Usage
prefetcher = DataPrefetcher(train_loader, device)
batch = prefetcher.next()
while batch is not None:
    output = model(batch)
    batch = prefetcher.next()
```

**Note:** DPP may already implement similar prefetching internally. Verify before adding custom prefetcher.

#### Implementation

```python
# In mast_roo_trainer_config.py
def _dataloader_config(self) -> DataLoaderConfig:
    return DisgDataLoaderConfig(
        client_thread_count=16,    # Test: 12, 16, 20 (current: 24)
        prefetch_queue_size=16,    # Test: 8, 16, 24 (current: 32)
        pin_memory=True,
    )
```

**Benchmark Matrix:**

| `client_thread_count` | `prefetch_queue_size` | Test Priority |
|----------------------|----------------------|---------------|
| 16 | 16 | High |
| 20 | 16 | Medium |
| 24 | 16 | Low |
| 16 | 24 | Low |

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5**—Config change |
| Complexity | **2/5**—Requires benchmarking |
| Risk Level | **Low**—Config only |
| Success Estimation | **80%** |
| Expected Impact | Reduced memory pressure, potentially faster loading |
| Implementation Effort | **1 day** |

---

### Proposal 7: Buffer Pre-allocation Warmup

#### Technical Analysis

Variable-length sequences cause memory fragmentation:

```
Iteration 1: Allocate [====Batch 1====]
Iteration 2: Allocate [==Batch 2==] (smaller)
Free Batch 1: [    hole    ][==Batch 2==]
Iteration 3: Need [======Batch 3======] but hole too small!
             → cudaMalloc → Fragmentation → OOM
```

**Solution:** Pre-allocate max-size buffers during warmup.

#### Implementation

```python
def preallocate_buffers(model, dataloader, device):
    """Warmup with max-size batch to cache memory allocations."""
    print("Pre-allocating GPU memory buffers...")

    sample_batch = next(iter(dataloader))

    with torch.no_grad():
        if hasattr(sample_batch, 'to'):
            sample_batch = sample_batch.to(device)

        output = model(sample_batch)

        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.sum().backward()

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print(f"Peak memory after warmup: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Usage
preallocate_buffers(model, train_loader, device)
# Then start training
```

**For KeyedJaggedTensor inputs:** Ensure warmup batch has maximum sequence lengths, maximum sparsity density, all feature keys populated.

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5**—Add warmup function |
| Complexity | **2/5**—Need correct batch structure |
| Risk Level | **Low**—Read-only warmup |
| Success Estimation | **80%** |
| Expected Impact | Prevents OOM from fragmentation |
| Implementation Effort | **1 day** |

---

### Proposal 8: Post-Accumulate Gradient Hooks for Dense Layers

#### Technical Analysis

Standard flow stores all gradients during backward:
```
Memory: [grad_L][grad_{L-1}][...][grad_1]  ← All stored!
```

With post-accumulate hooks, gradients are applied immediately:
```
grad_L → APPLY → free → grad_{L-1} → APPLY → free
Memory: Only one gradient at a time
```

**Research Finding:** Peak memory reduced from ~6GB to ~4.8GB for ViT-L-16 (~20% reduction)

**Note:** TorchRec's fused optimizer already does this for embeddings. This targets **dense layers** (MLPs, transformers).

#### Implementation

```python
def configure_post_accumulate_hooks(model, optimizer_cls, optimizer_kwargs):
    optimizer_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            optimizer_dict[param] = optimizer_cls([param], **optimizer_kwargs)

    def make_hook(param):
        def hook(grad):
            optimizer_dict[param].step()
            optimizer_dict[param].zero_grad(set_to_none=True)
            return grad
        return hook

    handles = []
    for param in model.parameters():
        if param.requires_grad:
            handle = param.register_post_accumulate_grad_hook(make_hook(param))
            handles.append(handle)

    return optimizer_dict, handles
```

**Caveats:**
- Not compatible with gradient accumulation
- Learning rate schedulers need adaptation
- Mixed precision scaling needs care

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **3/5**—Hook registration |
| Complexity | **3/5**—Per-parameter optimizer management |
| Risk Level | **Medium**—Changes training dynamics |
| Success Estimation | **70%** |
| Expected Impact | ~20% peak memory reduction |
| Implementation Effort | **3-4 days** |

---

## 3. Phase 3: Advanced Optimizations (Week 4+)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 9 | Fused SDD for optimizer overlap | 3/5 | 3/5 | Medium | 70% | +5-10% QPS | 3-5 days |
| 10 | Prefetch SDD for UVM-Cached Embeddings | 2/5 | 4/5 | Medium | 60% | Larger models | 7-10 days |
| 11 | 2D Sparse Parallelism | 1/5 | 5/5 | High | 50% | 1000+ GPU scale | 15+ days |

---

### Proposal 9: Fused SDD for Optimizer Overlap

**When valuable:** Computationally expensive optimizers (Shampoo, LAMB)

```python
pipeline = TrainPipelineSparseDist(
    model=model,
    optimizer=optimizer,
    device=device,
    apply_optimizer_in_backward=True,  # Fused SDD
)
```

**Dependency:** TorchRec v1.3.0+

---

### Proposal 10: Prefetch SDD for UVM-Cached Embeddings

**When valuable:** Embedding tables exceed GPU HBM capacity

4-stage pipeline: Cache Prefetch → H2D Transfer → Input Dist → Compute

**Research Finding:** Single-A100 full DLRM training possible (28 min vs 4 min on 8×A100)

---

### Proposal 11: 2D Sparse Parallelism

**When valuable:** Scaling to 1000+ GPUs

```
Sharding Groups (Model Parallel): [0,2,4,6], [1,3,5,7]
Replica Groups (Data Parallel):   [0,1], [2,3], [4,5], [6,7]
```

Benefits: Intra-node high-bandwidth AllReduce, reduced All-to-All latency

---

## 4. H100/H200 Future Considerations

### FlashAttention-3 Transition

| GPU | Architecture | FlashAttention Version | Key Technology |
|-----|--------------|------------------------|----------------|
| **A100** | Ampere (SM80) | FlashAttention-2 | `cp.async` |
| **H100** | Hopper (SM90) | FlashAttention-3 | TMA + Warp Specialization |

**Action Item:** When migrating to H100, verify FA3 dispatch instead of FA2.

### FP8 Precision Opportunity

H100/H200 introduces native FP8 Tensor Cores:
- **A100**: autocast manages float32 ↔ bfloat16
- **H100**: FP8 quantization (via TransformerEngine or torchao) for additional gains

**Note:** FP8 requires delayed scaling strategies—different workflow from BF16 autocast.

### Memory Bandwidth vs Capacity

| GPU | Memory | Capacity | Bandwidth |
|-----|--------|----------|-----------|
| **A100** | HBM2e | 80GB | ~2.0 TB/s |
| **H100** | HBM3 | 80GB | ~3.35 TB/s |
| **H200** | HBM3e | 141GB | ~4.8 TB/s |

**Impact:** H200's 141GB allows brute-force larger batches; some memory optimizations become less critical.

---

## 5. Complete Optimized Training Loop

Reference implementation combining all optimizations (from Research Section 14.5):

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
import torch.distributed as dist

# Initialize NCCL backend
dist.init_process_group(backend="nccl")

# Optimized DataLoader
sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
train_loader = DataLoader(
    dataset,
    batch_size=128,
    sampler=sampler,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True
)

# Distributed model with sharded embeddings
model = DistributedModelParallel(
    module=recommendation_model,
    device=torch.device("cuda"),
)

# Pipelined training (SDD)
pipeline = TrainPipelineSparseDist(
    model=model,
    optimizer=optimizer,
    device=device,
)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Critical for proper shuffling
    data_iter = iter(train_loader)
    for _ in range(len(train_loader)):
        pipeline.progress(data_iter)
```

---

## 6. Complete Verification Protocols

### 6.1 SDPA Backend Verification

```python
import torch.nn.attention as attn

# Method 1: Force and verify
def verify_sdpa_backend(q, k, v):
    backends = [
        (attn.SDPBackend.FLASH_ATTENTION, "FlashAttention"),
        (attn.SDPBackend.EFFICIENT_ATTENTION, "Memory-Efficient"),
        (attn.SDPBackend.MATH, "Math (Fallback)"),
    ]

    for backend, name in backends:
        try:
            with attn.sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v)
            print(f"✅ {name}: Available")
        except RuntimeError:
            print(f"❌ {name}: Not available")
```

### 6.2 Cast Overhead Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Training step
    ...

# Analyze cast operations
for event in prof.key_averages():
    if '_to_copy' in event.key or 'aten::to' in event.key:
        print(f"{event.key}: {event.count} calls, {event.cuda_time_total/1000:.2f}ms")
```

### 6.3 Kernel Name Patterns (Nsight)

| Pattern | Backend |
|---------|---------|
| `pytorch_flash::flash_fwd_kernel` | FlashAttention |
| `fmha_*`, `efficient_attention_*` | Memory-Efficient |
| `aten::bmm`, `aten::softmax` | Math (Fallback)—PROBLEM! |

### 6.4 Memory Validation

```python
torch.cuda.reset_peak_memory_stats()
# Training epoch
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 6.5 Throughput Validation

```python
import time
start = time.time()
for i, batch in enumerate(dataloader):
    if i >= 100: break
    # Training step
qps = 100 * batch_size / (time.time() - start)
print(f"QPS: {qps:.2f}")
```

---

## 7. Implementation Roadmap

### Recommended Execution Order

```
Week 1 (Critical Optimizations):
  Day 1:        🔴 CRITICAL-1: Remove redundant explicit casts       [1 day]
  Day 2:        🔴 CRITICAL-2: SDPA backend verification             [1 day]
  Day 3 AM:     🍎 LHF-3: zero_grad(set_to_none=True)               [0.5 day]
  Day 3 PM:     🍎 LHF-4: Cached dtype pattern                       [0.5 day]
  Day 4:        Validate all Phase 1 changes                         [0.5 day]

Week 2-3 (Medium Priority):
  Days 5-7:     Proposal 5: Triton kernel custom_fwd                 [2-3 days]
  Days 8-9:     Proposal 6-7: DPP tuning + Buffer pre-allocation     [2 days]
  Days 10-13:   Proposal 8: Post-accumulate gradient hooks           [3-4 days]
  Day 14:       Validate Phase 2, compare baseline                   [1 day]

Week 4+ (As Needed):
  Proposal 9:   Fused SDD (if using Shampoo/LAMB)
  Proposal 10:  Prefetch SDD (if embeddings exceed HBM)
  Proposal 11:  2D Sparse Parallelism (if scaling 1000+ GPUs)
```

---

## 8. Complete Prioritization Matrix

### Tier 1: Critical Optimizations 🔴

| # | Proposal | Impact | Risk | Success | Effort |
|---|----------|--------|------|---------|--------|
| 🔴1 | Remove redundant casts | Reduce kernel launches | Low | 90% | 1d |
| 🔴2 | SDPA backend verification | Prevent 2× slowdown | Medium | 85% | 1d |
| 🍎3 | `zero_grad(set_to_none=True)` | Minor speedup | Low | 95% | 0.5d |
| 🍎4 | Cached dtype pattern | Correct precision | Low | 85% | 1d |

**Total: ~3.5 days**

### Tier 2: Medium Priority

| # | Proposal | Impact | Risk | Success | Effort |
|---|----------|--------|------|---------|--------|
| 5 | Triton `custom_fwd` | Correctness + speed | Medium | 75% | 2-3d |
| 6 | DPP parameter tuning | Data loading | Low | 80% | 1d |
| 7 | Buffer pre-allocation | Prevent OOM | Low | 80% | 1d |
| 8 | Post-accumulate hooks | ~20% memory | Medium | 70% | 3-4d |

**Total: ~8-10 days**

### Tier 3: Advanced

| # | Proposal | Impact | Risk | Success | Effort |
|---|----------|--------|------|---------|--------|
| 9 | Fused SDD | +5-10% QPS | Medium | 70% | 3-5d |
| 10 | Prefetch SDD (UVM) | Larger models | Medium | 60% | 7-10d |
| 11 | 2D Parallelism | 1000+ GPU scale | High | 50% | 15+d |

**Total: 30+ days**

---

## 9. Appendix A: Quick Implementation Checklist

### Phase 1 Checklist

```
□ CRITICAL-1: Remove redundant explicit casts
  □ Search for .to(torch.bfloat16) in autocast blocks
  □ Remove casts before standard layers (Linear, Conv, etc.)
  □ KEEP casts before SDPA calls (guard rail)
  □ KEEP casts before Triton kernels
  □ Profile _to_copy count before/after

□ CRITICAL-2: SDPA backend verification
  □ Add verify_flash_attention() function
  □ Wrap SDPA calls with dtype/contiguity guards
  □ Test with sdpa_kernel context manager
  □ Add runtime verification flag for testing
  □ Check CUDA grid limit (batch_size × num_heads ≤ 65,535)
  □ Check head_dim (multiple of 8, ≤256, avoid 80 on older FA)

□ LHF-3: zero_grad(set_to_none=True)
  □ Locate optimizer.zero_grad() calls
  □ Add set_to_none=True parameter
  □ Verify no code depends on grad being zero tensor

□ LHF-4: Cached dtype pattern
  □ Identify modules with autocast blocks
  □ Add input_dtype caching BEFORE autocast
  □ Add output.to(input_dtype) AFTER autocast
  □ Avoid nesting pattern too deeply
```

### Phase 2 Checklist

```
□ Proposal 5: Triton kernel custom_fwd
  □ Inventory all Triton kernels in codebase
  □ Wrap with custom_fwd decorator
  □ If using torch.compile, set emulate_precision_casts = True
  □ Test mixed precision thoroughly

□ Proposal 6: DPP tuning
  □ Benchmark current settings
  □ Test reduced prefetch_queue_size (16 vs 32)
  □ Test reduced client_thread_count (16 vs 24)

□ Proposal 7: Buffer pre-allocation
  □ Add warmup function before training
  □ Ensure warmup batch has max sizes
```

---

## 10. Appendix B: Memory Optimization Summary

| Technique | Estimated Impact | Status |
|-----------|------------------|--------|
| `gradient_as_bucket_view=True` | ~4GB savings | ✅ Done |
| Inplace data copy | -3% peak | ✅ Done |
| SDD Lite | +4-5% QPS, +1% memory | ✅ Done |
| Remove redundant casts | Reduced kernel overhead | **TODO Phase 1** |
| SDPA FlashAttention guards | Prevent 2× slowdown | **TODO Phase 1** |
| Buffer pre-allocation | Prevent OOM | **TODO Phase 2** |
| Post-accumulate hooks | ~20% peak reduction | **TODO Phase 2** |

---

## 11. Appendix C: Optimization Impact Summary

| Issue | Impact | Solution |
|-------|--------|----------|
| Redundant explicit casts | 26× more kernel launches | Remove `.to(dtype)` in autocast blocks |
| SDPA backend fallbacks | 2×+ performance loss | Verify Flash Attention compatibility |
| Triton kernel conflicts | Numerical inconsistencies | Test mixed precision thoroughly |
| torch.compile + Triton | GEMM stays fp32 | Use `emulate_precision_casts = True` |
| Dtype access timing | Runtime inconsistencies | Cache dtype before autocast |
| Sample packing | 2× slower (Memory-Efficient) | Aware of limitation |
| CUDA grid limit | Silent failures | Check batch_size × num_heads ≤ 65,535 |

---

## 12. Appendix D: References

### Research Sources
- PyTorch Mixed Precision Blog: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
- PyTorch SDPA Documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- FlashAttention-2 Paper (Dao et al., 2022)
- TorchRec TrainPipelineSparseDist API Documentation
- NVIDIA Ampere Architecture Documentation

### Codebase References
- Commit `f220c6d5`: SDD Lite + Inplace copy implementation
- File `num_candidates_info.py`: NumCandidatesInfo pattern
- File `mast_roo_trainer_config.py`: DPP configuration

---

*Document generated: 2026-01-30*
*Research sources: Comprehensive TorchRec Pipeline and Mixed-Precision Training Research (Q5)*
*Focus: Evidence-based optimizations with verification protocols*
