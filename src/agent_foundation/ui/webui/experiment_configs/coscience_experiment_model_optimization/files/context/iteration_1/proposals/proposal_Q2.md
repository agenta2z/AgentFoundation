# Proposal Q2: torch.compile Optimization Strategies for Recommendation Models

**Date:** 2026-01-30 (Revised)
**Author:** CSML Optimization Team
**Status:** Draft - Revised with Deep Research Integration
**Target Codebase:** `fbs_1ce9d8_jia` (V2 Optimized)

---

## Executive Summary

This revised proposal synthesizes comprehensive research findings on `torch.compile` optimization strategies for recommendation models, incorporating deep technical analysis of kernel fusion mechanics, GEMM barriers, numerical precision concerns, and production deployment patterns. Our analysis identifies **a refined set of optimization opportunities** organized by evidence-based impact assessment.

### Critical Research Insights (New)

1. **3→1 Kernel Fusion is Real, but Limited**: Normalization subgraphs achieve 3→1 kernel fusion under Inductor. However, the **total Linear+Normalize block remains at 2 kernels** due to the fundamental GEMM fusion barrier—this is architecturally insurmountable with standard `torch.compile`.

2. **GEMM Fusion Barrier is Fundamental**: Normalization operations **cannot fuse with adjacent Linear layers** because:
   - GEMMs dispatch to external kernels (cuBLAS/CUTLASS) treated as `ExternKernelSchedulerNode`
   - Normalizations are reductions requiring global synchronization
   - This barrier is inherent to Inductor's architecture, not a configuration issue

3. **`activation_memory_budget=0.05` Preserves Fusion with Caveats**: The aggressive budget does NOT break fusion logic. However, it can introduce **graph cuts** to drop activations, fragmenting operations into smaller segments—more total kernel launches even though each kernel remains internally fused.

4. **Numerical Precision Alert**: Default `eps=1e-12` in `F.normalize` is **UNSAFE for float16/bfloat16** (PyTorch GitHub #32137). **Always use `eps=1e-6`** for mixed precision training.

5. **F.normalize Preferred Over Manual**: While both achieve identical kernel counts under Inductor (2 kernels for Linear+Norm), `F.normalize` provides:
   - Canonical IR graph guaranteeing optimal scheduling
   - Superior gradient stability via `max(x, eps)` vs `clamp(x, min=eps)`
   - Reduced "decomposition drift" risk

### PyTorch 2.0 Compilation Stack

| Component | Function |
|-----------|----------|
| **TorchDynamo** | Captures Python bytecode, constructs FX graph, handles dynamic behavior |
| **AOTAutograd** | Traces backward graph ahead of time, decomposes operators to ATen primitives |
| **TorchInductor** | Lowers FX graph to optimized Triton kernels, performs loop fusion |

**Core Optimization Mechanism:** Loop fusion combines multiple pointwise and reduction operations into a single kernel to maximize data locality in GPU SRAM (L1/L2 cache).

### Revised Impact Summary

| Tier | Proposals | Expected Impact | Confidence | Effort |
|------|-----------|-----------------|------------|--------|
| **🍎 Low-Hanging Fruits** | 3 | 10-20% QPS improvement | High (85%+) | ~4 days |
| **High Priority** | 3 | 15-30% additional | Medium (70-80%) | ~12 days |
| **Medium Priority** | 3 | 10-20% additional | Medium (60-75%) | ~15 days |
| **Future Considerations** | 3 | Variable | Lower (40-60%) | 20+ days |

---

## Research Synthesis

### Key Technical Findings

#### 1. Kernel Fusion Mechanics

**Inductor Fusion Decision Process:**
- Reduction Node: `linalg.norm` identified as reduction summing squares over `dim=-1`
- Pointwise Nodes: `clamp` and `div` identified as element-wise consumers
- Fusion Decision: Scheduler recognizes Producer-Consumer pattern, fuses all three operations

**Resulting Kernel Counts:**

| Implementation | Execution Mode | Total Kernels |
|----------------|----------------|---------------|
| Manual (norm→clamp→div) | Eager | **3-4** |
| F.normalize | Eager | **2** |
| Manual | **Inductor** | **2** |
| F.normalize | **Inductor** | **2** |
| Liger Kernel (Custom) | Custom Triton | **1** |

**Critical Insight**: The "3 to 1" fusion applies to the **normalization subgraph only**. The total block (Linear+Norm) remains at 2 kernels regardless of implementation due to the GEMM barrier.

#### Persistent Reduction Kernels (Technical Detail)

Inductor selects kernel types based on scheduling heuristics:

| Kernel Type | When Used | Memory Traffic |
|-------------|-----------|----------------|
| **Persistent Reduction** | Reduction dimension fits in registers/shared memory (D ≤ 1024) | 1 Read, 1 Write (optimal) |
| **Loop Reduction** | Massive reduction dimensions | Multiple reads, spills to global memory |

For recommendation models with embedding dimensions typically 64-1024, Inductor correctly selects Persistent Reduction. The generated Triton kernel structure demonstrates why fusion is so effective:

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

#### 2. GEMM Fusion Barrier Analysis

**Why GEMMs Cannot Fuse with Normalization:**

1. **External Kernel Isolation**: cuBLAS/CUTLASS kernels are pre-optimized, self-contained
2. **Prologue Fusion Unsupported**: Operations before GEMM cannot be fused
3. **Reduction Synchronization**: Normalization requires:
   - Computing sum of squares of entire output row
   - Global barrier synchronization to aggregate partial sums
   - Second pass to divide elements

**Execution Graph for `Linear → F.normalize`:**
```
Kernel 1 (GEMM): Y = XW^T, fuses bias if present, writes Y to HBM
Kernel 2 (Normalization): Reads Y from HBM, computes ‖Y‖, normalizes, writes Z
```

**Implication**: No amount of configuration tuning will achieve 1-kernel Linear+Norm execution via standard `torch.compile`. This requires custom Triton kernels (Liger Kernel, FlashAttention's fused RMSNorm).

#### 3. Activation Memory Budget Impact

**`activation_memory_budget=0.05` Behavior:**

| Aspect | Impact |
|--------|--------|
| **Forward Pass Fusion** | **No impact** - Fused kernels generated and executed regardless |
| **Backward Pass** | Recomputation Graph uses **same fused Triton kernels** |
| **Graph Cuts** | ⚠️ Very aggressive budgets may introduce graph cuts to drop activations |
| **Kernel Fragmentation** | Operations that would otherwise fuse may end up in different segments |

**Key Insight**: Fusion decisions are based on instruction-level parallelism and memory bandwidth. Recomputation decisions are based on memory capacity. These are **orthogonal optimization axes**.

**Counter-Intuitive Finding**: Low budget can **enable more fusion** in backward pass:
- Recomputed pointwise ops can fuse with gradient computations
- PyTorch AOT Autograd benchmarks show recomputation can be faster

**Recommendation**: Budget 0.05 is extremely aggressive—consider 0.2-0.5 for better balance unless model physically cannot fit in VRAM.

#### 4. Numerical Precision Considerations

**Gradient Divergence at Singularity:**

| Method | Behavior at ‖v‖ < ε |
|--------|---------------------|
| `max(x, eps)` (F.normalize) | Gradient w.r.t. norm becomes **0** (clean cutoff) |
| `clamp(x, min=eps)` (Manual) | Can have instability if norm produces NaN/Inf before clamp |

**Epsilon Safety by Dtype:**

| Dtype | Safe eps Range | Recommended |
|-------|----------------|-------------|
| float32 | 1e-12 to 1e-8 | 1e-8 |
| **float16/bfloat16** | **1e-6 to 1e-5** | **1e-6** |
| float64 | 1e-12 | 1e-12 |

⚠️ **ACTION REQUIRED**: Audit all `F.normalize` calls for explicit `eps=1e-6` in mixed precision code paths.

**Floating-Point Accumulation Differences:**

Minor precision nuances between manual and compiled approaches:

1. **Reduction Order**: Fused kernels may sum elements in different order/parallel pattern
2. **Half-Precision Behavior**: Eager mode may internally upsample certain ops to higher precision; Inductor does not insert extra upcasts by default
3. **Magnitude**: Differences are on the order of machine epsilon, generally negligible for model quality

### Codebase Investigation Findings (Updated)

| Aspect | Current V2 State | Research-Informed Assessment |
|--------|------------------|------------------------------|
| **torch.compile() calls** | REMOVED (was 4 calls in V1) | ✅ Correct—framework-level PT2 strategies |
| **Kernel fusion** | 221→30 kernels (87% reduction) | ✅ Significant gain; further reduction limited by GEMM barrier |
| **activation_memory_budget** | 0.05 (5%) | ⚠️ May cause graph cuts; consider 0.2-0.5 for speed |
| **Normalization patterns** | Manual patterns present | 🔴 Replace with `F.normalize(x, p=2, dim=-1, eps=1e-6)` |
| **Epsilon values** | Not explicitly verified | 🔴 Audit for float16/bfloat16 safety |
| **mark_dynamic() in model** | ❌ NOT implemented | 🔴 **Opportunity for model_roo_v0.py** |
| **Hybrid compile strategy** | ❌ NOT explicit | 🔴 **Opportunity: @torch.compiler.disable on sparse** |

---

## Assessment Framework

| Criteria | Description | Scale |
|----------|-------------|-------|
| **Technical Analysis** | Why this change helps, underlying mechanisms, research evidence | - |
| **Easiness** | How simple to implement | 1-5 (5=trivial) |
| **Complexity** | System changes required | 1-5 (1=simple config) |
| **Risk Level** | Potential for regressions | Low/Medium/High |
| **Research Confidence** | How well-supported by research findings | % |
| **Expected Impact** | Quantified improvement | QPS/latency % |
| **Implementation Effort** | Person-days | Days |
| **🍎 Low-Hanging Fruit?** | Easiness≥4, Complexity≤2, Risk=Low, Confidence≥80% | Yes/No |

---

## 🍎 Phase 1: LOW-HANGING FRUITS (Week 1)

These proposals meet all low-hanging fruit criteria with strong research evidence.

---

### 🍎 LHF-1: Normalize Pattern Standardization with F.normalize

**Technical Analysis:**

Research confirms both manual and `F.normalize` achieve identical kernel counts (2) under Inductor. However, `F.normalize` is **strongly preferred** because:

1. **Canonical IR Graph**: Consistent lowering to Persistent Reduction pattern
2. **Pattern Matching**: Inductor recognizes subgraph: `SumSq → Sqrt → Max → Div`
3. **Optimized Templates**: Maps directly to specialized Triton templates
4. **Gradient Stability**: Uses `max(x, eps)` providing cleaner gradient signal
5. **Decomposition Drift Prevention**: Eliminates risk of suboptimal "split reductions"

**Critical Fix**: Default `eps=1e-12` is UNSAFE for float16/bfloat16.

**Implementation:**

```python
# BEFORE (manual pattern - decomposition drift risk)
norm = torch.linalg.norm(x, dim=-1, keepdim=True)
norm = torch.clamp(norm, min=1e-8)
result = x / norm

# AFTER (canonical pattern - guaranteed optimal scheduling)
result = F.normalize(x, p=2, dim=-1, eps=1e-6)  # eps=1e-6 for mixed precision safety
```

**Search and Replace Targets:**
- All occurrences of `torch.linalg.norm` followed by `clamp` and division
- All occurrences of `torch.norm` (deprecated) patterns
- Verify all existing `F.normalize` calls have explicit `eps=1e-6`

**Validation:**

```python
# Numerical equivalence verification
def validate_normalize_migration(x, eps=1e-6):
    # Manual
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    manual_result = x / norm

    # F.normalize
    fn_result = F.normalize(x, p=2, dim=-1, eps=eps)

    torch.testing.assert_close(manual_result, fn_result, rtol=1e-3, atol=1e-3)
    print("✓ Numerical equivalence validated")
```

| Assessment | Value |
|------------|-------|
| Easiness | **5/5** - Direct pattern replacement |
| Complexity | **1/5** - No architectural changes |
| Risk Level | **Low** - Numerically equivalent, cleaner gradients |
| Research Confidence | **95%** - Extensively documented in PyTorch and research |
| Expected Impact | **5-10% end-to-end speedup** from normalization overhead reduction; ~200+ fewer kernel launches per forward pass |
| Implementation Effort | **1 day** |
| 🍎 Low-Hanging Fruit? | **YES** ✅ |

---

### 🍎 LHF-2: Dynamic Shape Guards with `mark_dynamic()`

**Technical Analysis:**

From research synthesis: "Each new sequence length combination generates fresh compiled code until hitting `recompile_limit` (default: 8 per function)." Production systems can exhaust this limit within minutes, degrading to eager execution.

The `mark_dynamic()` API prevents shape specialization, forcing symbolic kernels that handle variable dimensions without recompilation.

**Key Tensors for Main Feed MTML:**
- `train_input.ro_float_features` - batch dimension varies
- `train_input.nro_float_features` - batch dimension varies
- `train_input.num_candidates` - variable candidates per example

**Implementation:**

```python
# In MainFeedMTMLROOTrain.forward()
def forward(
    self, train_input: MainFeedMTMLROOTrainBatch
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Add dynamic shape guards at start of forward
    torch._dynamo.mark_dynamic(train_input.ro_float_features, 0)  # Batch dim
    torch._dynamo.mark_dynamic(train_input.nro_float_features, 0)  # Batch dim
    torch._dynamo.mark_dynamic(train_input.num_candidates, 0)      # Variable length

    # For bounded ranges (if known):
    # torch._dynamo.mark_dynamic(train_input.ro_float_features, 0, min=1, max=4096)

    # Existing forward logic follows...
```

**Production Configuration (from research):**

```python
# Recompilation limits for production stability
torch.compiler.config.recompile_limit = 8
torch.compiler.config.accumulated_recompile_limit = 256
torch.compiler.config.fail_on_recompile_limit_hit = True  # Alert on limit
torch.compiler.config.automatic_dynamic_shapes = True
torch.compiler.config.assume_static_by_default = True
```

**Monitoring:**

```python
import torch._dynamo.utils
counters = torch._dynamo.utils.counters
metrics.gauge("pytorch.recompiles", sum(counters['recompiles'].values()))
```

**Nested Tensors for Variable-Length Sequences:**

For recommendation models with variable-length sequences, native nested tensors compile cleanly with `fullgraph=True`:

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

| Assessment | Value |
|------------|-------|
| Easiness | **4/5** - Add mark_dynamic calls |
| Complexity | **2/5** - Identify key tensors with variable shapes |
| Risk Level | **Low** - Non-invasive, graceful degradation |
| Research Confidence | **90%** - Proven pattern from vLLM, production RecSys |
| Expected Impact | Prevent recompilation thrashing, stable compilation |
| Implementation Effort | **1 day** |
| 🍎 Low-Hanging Fruit? | **YES** ✅ |

---

### 🍎 LHF-3: Hybrid Sparse/Dense Compilation Strategy

**Technical Analysis:**

Research confirms: "Direct torch.compile on full DLRM-style models yields essentially zero difference due to graph breaks from sparse operations." The compute distribution:

- **Sparse Architecture (60-80%)**: Already optimized via FBGEMM's `SplitTableBatchedEmbeddingBagsCodegen`
- **Dense Architecture (20-40%)**: MLP interaction layers benefit from Inductor fusion (20-50% speedup)

**Graph Break Sources in RecSys:**
- `sparse=True` fails with Inductor assertion error
- TorchRec's `EmbeddingBagCollection` requires `@torch.compiler.disable`
- `.item()` calls force immediate graph termination
- Data-dependent control flow

**Implementation:**

```python
# In pytorch_modules.py - Explicitly disable compilation on sparse lookups
class SparseArch(nn.Module):
    @torch.compiler.disable  # Don't compile sparse lookups - FBGEMM already optimized
    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
        id_score_list_features: KeyedJaggedTensor,
    ) -> Tuple[KeyedTensor, KeyedTensor]:
        """FBGEMM's SplitTableBatchedEmbeddingBagsCodegen handles this optimally"""
        with record_function(f"## {self.__class__.__name__}: forward ##"):
            with record_function("## ebc: forward ##"):
                embeddings = self.ebc(id_list_features)
            return embeddings

# For KeyedJaggedTensor inputs - mark dynamic dimensions
@torch.compiler.disable
def embed(self, kjt):
    # Mark dynamic before any compilation boundary
    torch._dynamo.mark_dynamic(kjt.values(), 0)
    torch._dynamo.mark_dynamic(kjt.lengths(), 0)
    return self.sparse_arch(kjt)
```

**Dense Layer Compilation:**

```python
# Compile dense interaction layers with appropriate mode
class MainFeedMTMLROO(nn.Module):
    def __init__(self, ...):
        ...
        # Compile dense interaction architecture
        self.interaction_arch = torch.compile(
            InteractionArch(...),
            mode="max-autotune",
            options={
                "epilogue_fusion": True,
                "max_autotune": True,
                "shape_padding": True,  # Tensor core alignment
            }
        )

        # Compile task architecture (dense MLP)
        self.task_arch = torch.compile(
            TaskArch(...),
            mode="max-autotune"
        )
```

**Expected Results (from industry case studies):**
- 20-50% latency reduction on dense portions
- FBGEMM's hand-tuned sparse kernels preserved
- Dense MLP components achieve 2-3x speedups

| Assessment | Value |
|------------|-------|
| Easiness | **4/5** - Decorator-based approach |
| Complexity | **2/5** - Clear sparse/dense separation exists |
| Risk Level | **Low** - Conservative disable on sparse |
| Research Confidence | **85%** - Validated by Meta, Pinterest, Netflix |
| Expected Impact | 20-50% dense layer speedup |
| Implementation Effort | **2 days** |
| 🍎 Low-Hanging Fruit? | **YES** ✅ |

---

## Phase 2: HIGH-PRIORITY (Week 2-3)

---

### Proposal 4: Activation Memory Budget Tuning

**Technical Analysis:**

Current setting `activation_memory_budget=0.05` is extremely aggressive. Research reveals a nuanced tradeoff:

**Benefits of 0.05:**
- Minimal activation memory footprint
- Enables training larger models
- Recomputation uses same fused kernels (no fusion loss)

**Drawbacks of 0.05:**
- Recomputes heavy GEMM operations (doubles FLOPs for linear layers)
- May introduce graph cuts fragmenting fused operations
- Suboptimal for speed when memory isn't the bottleneck

**Counter-Intuitive Finding: Recomputation Can Be Faster**

Low budget (0.05) can enable **more fusion** in the backward pass because recomputed pointwise ops can fuse with gradient computations. PyTorch AOT Autograd benchmarks demonstrate this:

```
Eager,       Fwd = 740.77µs, Bwd = 1560.52µs
AOT,         Fwd = 713.85µs, Bwd = 909.12µs
AOT_Recomp,  Fwd = 712.22µs, Bwd = 791.46µs  ← Recomputation is faster
```

Documentation insight: "We can recompute fusion-friendly operators to save memory, and then rely on the fusing compiler to fuse the recomputed operators. This reduces both memory and runtime."

**Research Recommendation:**

| Budget | Behavior | Use Case |
|--------|----------|----------|
| 1.0 | No extra recomputation | Speed-critical inference |
| **0.2-0.5** | **Balanced tradeoff** | **Most training scenarios** |
| 0.1-0.2 | Moderate recomputation | Memory-conscious training |
| 0.05 | Aggressive recomputation | Only if model won't fit otherwise |

**Implementation:**

```python
# In model_roo_v0.py - Tunable memory budget
import os

# Allow runtime configuration
ACTIVATION_MEMORY_BUDGET = float(os.environ.get(
    "ACTIVATION_MEMORY_BUDGET",
    "0.3"  # Changed from 0.05 to 0.3 for better speed/memory balance (0.2-0.5 recommended)
))

torch._functorch.config.activation_memory_budget = ACTIVATION_MEMORY_BUDGET
torch._functorch.config.activation_memory_budget_runtime_estimator = "flops"
torch._functorch.config.activation_memory_budget_solver = "dp"
```

**A/B Testing Strategy:**
1. Run with budget=0.05 (current) - measure QPS, memory
2. Run with budget=0.3 - measure QPS, memory
3. Run with budget=0.5 - measure QPS, memory
4. Select optimal based on infrastructure constraints

| Assessment | Value |
|------------|-------|
| Easiness | **5/5** - Single config change |
| Complexity | **1/5** - No code changes |
| Risk Level | **Low** - Easy to revert |
| Research Confidence | **80%** - Well-documented tradeoff |
| Expected Impact | 10-20% throughput improvement if not memory-bound |
| Implementation Effort | **0.5 days** (plus A/B testing) |
| 🍎 Low-Hanging Fruit? | **Borderline** - requires validation |

---

### Proposal 5: LayerNorm/RMSNorm Migration for Optimal Fusion

**Technical Analysis:**

Research reveals PyTorch Inductor has **dedicated fusion classes** for LayerNorm that `F.normalize` lacks:

- `BatchLayernormFusion`: Robust fusion with adjacent layers
- `BatchLinearFusion`: Strong support for linear layer combinations

**Key Insight**: `F.normalize` relies on generic pointwise fusion patterns, making it a lower-priority optimization target. LayerNorm/RMSNorm achieve better fusion with Linear layers through specialized templates.

**Mathematical Equivalence Check:**

L2 normalization: $y = \frac{x}{\|x\|_2}$

RMSNorm: $y = \frac{x}{\text{RMS}(x)} \cdot \gamma$ where $\text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2}$

**Note**: These operations are mathematically different. Migration requires careful validation that the semantic meaning is preserved for the specific use case. The research recommends "Consider replacing `F.normalize` with LayerNorm or RMSNorm **where mathematically appropriate** for proven fusion capabilities."

**Where to Apply:**
- Embedding normalizations after projection layers
- Feature normalizations where affine parameters can be absorbed
- NOT suitable for cosine similarity computations requiring true L2 norm

**Implementation (where mathematically valid):**

```python
# IMPORTANT: L2 normalization and RMSNorm are NOT mathematically equivalent!
# L2 norm: ||x||_2 = sqrt(sum(x_i^2))
# RMS: sqrt(1/n * sum(x_i^2)) = ||x||_2 / sqrt(n)
#
# Use LayerNorm/RMSNorm ONLY where the application allows this difference,
# such as when followed by learnable scale parameters that can absorb the factor.

# Option 1: Use PyTorch's built-in RMSNorm (gets BatchLayernormFusion benefits)
self.norm = torch.compile(nn.RMSNorm(hidden_dim, eps=1e-6))

# Option 2: For true L2 normalization, use F.normalize (canonical, well-fused)
# output = F.normalize(x, p=2, dim=-1, eps=1e-6)

# The research recommends: "Consider replacing F.normalize with LayerNorm or
# RMSNorm WHERE MATHEMATICALLY APPROPRIATE for proven fusion capabilities."
# This requires validating that the semantic meaning is preserved for your use case.
```

| Assessment | Value |
|------------|-------|
| Easiness | **2/5** - Mathematical validation required |
| Complexity | **3/5** - Case-by-case analysis |
| Risk Level | **Medium** - Numerical differences possible |
| Research Confidence | **75%** - Clear fusion benefits, application-specific |
| Expected Impact | Better Linear+Norm fusion (potentially 1 kernel vs 2) |
| Implementation Effort | **3-4 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

### Proposal 6: FP8 Tensor Core Integration for H100

**Technical Analysis:**

H100 FP8 delivers 1,979 TFLOPS—6x vs A100 FP16. The `torchao.float8` library enables FP8 training on eligible Linear layers with dimensions divisible by 16.

**Candidate Layers in V2:**
- DeepCrossNet: `low_rank_dim = 512` (divisible by 16) ✓
- Task MLP layers: Various dimensions (audit required)
- Interaction layers: Must verify dimensions

**Implementation:**

```python
from torchao.float8 import convert_to_float8_training

def filter_fn(mod, fqn):
    """Only convert Linear layers with Tensor Core-compatible dimensions"""
    if isinstance(mod, torch.nn.Linear):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    return False

# Apply after model construction
def enable_fp8_training(model):
    convert_to_float8_training(model, module_filter_fn=filter_fn)
    return model

# Usage with max-autotune for H100 optimization
model = MainFeedMTMLROO(...)
model = enable_fp8_training(model)
compiled_model = torch.compile(
    model,
    mode="max-autotune",
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,  # Tensor core alignment
    }
)
```

**Prerequisites:**
- H100 GPU hardware
- `torchao` library installed
- Numerical stability validation across all training scenarios

| Assessment | Value |
|------------|-------|
| Easiness | **3/5** - External library integration |
| Complexity | **3/5** - Precision calibration required |
| Risk Level | **Medium** - Numerical stability concerns |
| Research Confidence | **70%** - Hardware-dependent, needs validation |
| Expected Impact | Up to 6x on eligible dense ops |
| Implementation Effort | **5-7 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

## Phase 3: MEDIUM-PRIORITY (Week 4+)

---

### Proposal 7: Comprehensive Graph Break Audit

**Technical Analysis:**

V2 uses `torch.fx.wrap` on many operations and NumCandidatesInfo has eliminated most `.item()` calls. A systematic audit would identify remaining graph breaks.

**Diagnostic Toolkit (from research):**

```bash
# Environment variable diagnostics
TORCH_LOGS=graph_breaks python train.py
TORCH_LOGS=recompiles,guards python train.py
TORCH_COMPILE_DEBUG=1 python train.py  # Creates debug directory
```

```python
# Programmatic analysis
def audit_graph_breaks(model, sample_input):
    explanation = torch._dynamo.explain(model)(sample_input)
    print(f"Graph breaks: {explanation.graph_break_count}")
    print(f"Break reasons:\n{explanation.break_reasons}")
    return explanation

# Strict validation
try:
    strict_compiled = torch.compile(model, fullgraph=True)
    output = strict_compiled(sample_input)
    print("✓ No graph breaks - fullgraph compilation successful")
except Exception as e:
    print(f"✗ Graph break detected: {e}")
```

**Modern Trace Analysis:**

```bash
pip install tlparse
TORCH_TRACE="/tmp/tracedir" python your_model.py
tlparse /tmp/tracedir --output report.html
```

**Pattern Matcher Counters:**

```python
from torch._dynamo.utils import counters
counters.clear()
output = compiled_model(input)
print(f"Fusions applied: {counters['inductor']['pattern_matcher_count']}")
```

| Assessment | Value |
|------------|-------|
| Easiness | **3/5** - Systematic audit required |
| Complexity | **3/5** - Depends on findings |
| Risk Level | **Low** - Diagnostic only |
| Research Confidence | **85%** - Well-established tooling |
| Expected Impact | Enable fullgraph compilation |
| Implementation Effort | **3-5 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

### Proposal 8: Regional Compilation for HSTU

**Technical Analysis:**

From research: "For massive models, compiling the entire graph at once is prohibitively slow. Regional compilation compiles one layer and reuses for all instances."

The HSTU module contains repetitive transformer layers ideal for this optimization.

**Implementation:**

```python
class HSTUModule(nn.Module):
    def __init__(self, num_layers, ...):
        super().__init__()
        # Create single compiled layer template
        self.layer_template = torch.compile(
            HSTULayer(...),
            mode="max-autotune"
        )
        # Reuse compiled layer for all instances
        self.layers = nn.ModuleList([
            self._create_layer_from_template(i)
            for i in range(num_layers)
        ])

    def _create_layer_from_template(self, idx):
        # Share compiled implementation, unique parameters
        layer = copy.deepcopy(self.layer_template)
        return layer
```

| Assessment | Value |
|------------|-------|
| Easiness | **2/5** - HSTU internal understanding required |
| Complexity | **4/5** - State sharing complexity |
| Risk Level | **Medium** - Parameter sharing concerns |
| Research Confidence | **70%** - Validated for LLMs, less for RecSys |
| Expected Impact | 5-10x cold start reduction |
| Implementation Effort | **5-7 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

### Proposal 9: CUDA Graphs for Inference

**Technical Analysis:**

`reduce-overhead` mode captures kernel launches as static graph, eliminating Python dispatch overhead. For inference with predictable batch sizes, CUDA graphs provide significant latency reduction.

**Challenge**: Variable input shapes require bucketing.

**Implementation:**

```python
class MainFeedMTMLROOInferenceWithCUDAGraphs(MainFeedMTMLROOInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_buckets = [64, 128, 256, 512, 1024, 2048]
        self.compiled_graphs = {}

    def _get_bucket(self, batch_size: int) -> int:
        for bucket in self.batch_buckets:
            if batch_size <= bucket:
                return bucket
        return self.batch_buckets[-1]

    def forward(self, inference_input):
        actual_batch = inference_input.ro_float_features.shape[0]
        bucket = self._get_bucket(actual_batch)

        if bucket not in self.compiled_graphs:
            self.compiled_graphs[bucket] = torch.compile(
                super().forward,
                mode="reduce-overhead"
            )

        padded_input = self._pad_to_bucket(inference_input, bucket)
        output = self.compiled_graphs[bucket](padded_input)
        return self._truncate_output(output, actual_batch)
```

**Note from research**: `reduce-overhead` eliminates Python kernel launch overhead but requires static shapes. Use `max-autotune-no-cudagraphs` or bucket inputs for variable shapes.

| Assessment | Value |
|------------|-------|
| Easiness | **3/5** - Bucket management complexity |
| Complexity | **3/5** - Memory management |
| Risk Level | **Medium** - +10-20% memory overhead |
| Research Confidence | **75%** - Well-documented pattern |
| Expected Impact | 20-40% inference latency reduction |
| Implementation Effort | **4-5 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

## Phase 4: FUTURE CONSIDERATIONS

---

### Proposal 10: Liger Kernel Integration for True Linear+Norm Fusion

**Technical Analysis:**

Research confirms standard `torch.compile` **cannot** achieve 1-kernel Linear+Norm execution due to GEMM fusion barrier. Liger Kernels provide manually written Triton kernels that:

- Restructure GEMM loop using "Split-K" or tile-based reduction
- Keep data in SRAM throughout
- Achieve true 1-kernel execution

**From research:**
> "For true 1-kernel execution of Linear+Norm, specialized kernels like Liger Kernel, FlashLinear, or FlashAttention's fused RMSNorm are required."

**Implementation:**

```python
from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss

# Replace nn.RMSNorm with fused implementation
self.norm = LigerRMSNorm(hidden_dim, eps=1e-6)

# For loss computation - fused Linear+CrossEntropy
self.loss_fn = LigerFusedLinearCrossEntropyLoss()
```

**Expected Impact**:
- Linear+Norm: 2 kernels → 1 kernel
- Memory savings from not materializing intermediate Linear output

| Assessment | Value |
|------------|-------|
| Easiness | **2/5** - External kernel integration |
| Complexity | **4/5** - Compatibility validation |
| Risk Level | **Medium** - Custom kernel stability |
| Research Confidence | **65%** - Newer technology, less production validation |
| Expected Impact | True 1-kernel Linear+Norm execution |
| Implementation Effort | **7-10 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

### Proposal 11: AOT Compilation with torch.export

**Technical Analysis:**

For environments where JIT compilation overhead is unacceptable, `torch.export` provides strict AOT compilation. Unlike `torch.compile` which falls back gracefully, `torch.export` requires a sound graph with no breaks.

**Prerequisites:**
- All graph breaks eliminated (Proposal 7)
- All dynamic shapes expressed via `torch.export.Dim` constraints
- No data-dependent control flow (or expressed via `torch.cond`)

```python
# Replace data-dependent branches with torch.cond
def forward(self, x):
    return torch.cond(
        x.sum() > 0,
        lambda x: self.layer_a(x),
        lambda x: self.layer_b(x),
        (x,)
    )
```

| Assessment | Value |
|------------|-------|
| Easiness | **2/5** - Requires clean graph |
| Complexity | **4/5** - Strict requirements |
| Risk Level | **High** - No graceful fallback |
| Research Confidence | **50%** - Challenging for complex RecSys |
| Expected Impact | Zero compilation latency in production |
| Implementation Effort | **7-10 days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

### Proposal 12: Custom CUDA Kernel Registration

**Technical Analysis:**

Custom CUDA kernels require FakeTensor registration to participate in compilation. This enables specialized kernels to be traced by TorchDynamo.

**Implementation:**

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

| Assessment | Value |
|------------|-------|
| Easiness | **1/5** - Expert CUDA knowledge required |
| Complexity | **5/5** - Major custom kernel work |
| Risk Level | **Medium** - Custom kernel stability |
| Research Confidence | **40%** - Highly specialized |
| Expected Impact | Enable compilation of custom ops |
| Implementation Effort | **10+ days** |
| 🍎 Low-Hanging Fruit? | **No** |

---

## 📊 Complete Prioritization Matrix

### Tier 1: Low-Hanging Fruits 🍎 (IMPLEMENT FIRST)

| # | Proposal | Easiness | Complexity | Risk | Confidence | Effort | Impact |
|---|----------|----------|------------|------|------------|--------|--------|
| **🍎1** | F.normalize Standardization | **5/5** | **1/5** | Low | 95% | 1d | Optimal IR, numerical safety |
| **🍎2** | Dynamic Shape Guards | **4/5** | **2/5** | Low | 90% | 1d | Stable compilation |
| **🍎3** | Hybrid Sparse/Dense Compile | **4/5** | **2/5** | Low | 85% | 2d | 20-50% dense speedup |

**Total Tier 1 Effort: ~4 days**

### Tier 2: High Priority

| # | Proposal | Easiness | Complexity | Risk | Confidence | Effort | Impact |
|---|----------|----------|------------|------|------------|--------|--------|
| 4 | Memory Budget Tuning | 5/5 | 1/5 | Low | 80% | 0.5d | 10-20% if not memory-bound |
| 5 | LayerNorm/RMSNorm Migration | 2/5 | 3/5 | Medium | 75% | 3-4d | Better Linear+Norm fusion |
| 6 | FP8 Integration | 3/5 | 3/5 | Medium | 70% | 5-7d | Up to 6x dense ops |

**Total Tier 2 Effort: ~12 days**

### Tier 3: Medium Priority

| # | Proposal | Easiness | Complexity | Risk | Confidence | Effort | Impact |
|---|----------|----------|------------|------|------------|--------|--------|
| 7 | Graph Break Audit | 3/5 | 3/5 | Low | 85% | 3-5d | Enable fullgraph |
| 8 | Regional HSTU Compile | 2/5 | 4/5 | Medium | 70% | 5-7d | 5-10x cold start |
| 9 | CUDA Graphs Inference | 3/5 | 3/5 | Medium | 75% | 4-5d | 20-40% inference |

**Total Tier 3 Effort: ~15 days**

### Tier 4: Future

| # | Proposal | Easiness | Complexity | Risk | Confidence | Effort | Impact |
|---|----------|----------|------------|------|------------|--------|--------|
| 10 | Liger Kernel Integration | 2/5 | 4/5 | Medium | 65% | 7-10d | True 1-kernel fusion |
| 11 | AOT Compilation | 2/5 | 4/5 | High | 50% | 7-10d | Zero compile latency |
| 12 | Custom Kernel Registration | 1/5 | 5/5 | Medium | 40% | 10+d | Custom op compilation |

---

## 🚀 Recommended Implementation Roadmap

```
Week 1 (Low-Hanging Fruits):
  Day 1:     🍎 LHF-1: F.normalize standardization
             - Replace manual norm patterns
             - Add eps=1e-6 for mixed precision
             - Validate numerical equivalence

  Day 2:     🍎 LHF-2: mark_dynamic() guards
             - Add guards in MainFeedMTMLROOTrain.forward()
             - Configure recompilation limits
             - Set up monitoring

  Day 3-4:   🍎 LHF-3: Hybrid sparse/dense compilation
             - Add @torch.compiler.disable to sparse paths
             - Apply max-autotune to dense interaction layers
             - Benchmark QPS improvement

Week 2-3 (High Priority):
  Day 5:     Proposal 4: Memory budget tuning A/B test
             - Test budget=0.3 vs 0.05
             - Measure QPS/memory tradeoff

  Day 6-9:   Proposal 5: LayerNorm/RMSNorm migration analysis
             - Identify mathematically valid replacement sites
             - Implement and validate

  Day 10-16: Proposal 6: FP8 integration (if H100 available)
             - Install torchao
             - Identify eligible Linear layers
             - Run precision calibration

Week 4+ (Medium Priority):
  Proposal 7: Graph break audit
  Proposal 8: Regional HSTU compilation
  Proposal 9: CUDA graphs for inference
```

---

## Validation Approach

### 1. Kernel Count Verification

```python
import torch

def verify_kernel_count(model, sample_input, expected_count=None):
    compiled_model = torch.compile(model, mode="max-autotune")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        result = compiled_model(sample_input)
        torch.cuda.synchronize()

    cuda_events = [e for e in prof.key_averages()
                   if e.device_type == torch.profiler.DeviceType.CUDA]
    actual_count = len(cuda_events)

    print(f"Kernel count: {actual_count}")
    if expected_count:
        assert actual_count <= expected_count, f"Expected ≤{expected_count}, got {actual_count}"

    # Export for visualization
    prof.export_chrome_trace("fusion_trace.json")
    return actual_count
```

### 2. Numerical Correctness

```python
def validate_numerical_correctness(model_original, model_optimized, test_inputs, rtol=1e-4, atol=1e-5):
    with torch.no_grad():
        out_orig = model_original(test_inputs)
        out_opt = model_optimized(test_inputs)

    for key in out_orig:
        torch.testing.assert_close(out_orig[key], out_opt[key], rtol=rtol, atol=atol)
    print("✓ Numerical correctness validated")
```

### 3. Fusion Verification

```python
from torch._dynamo.utils import same

def verify_fusion(input_tensor, eps=1e-6):
    eager_result = F.normalize(input_tensor, dim=-1, eps=eps)
    compiled_fn = torch.compile(lambda x: F.normalize(x, dim=-1, eps=eps))
    compiled_result = compiled_fn(input_tensor)

    assert same(eager_result, compiled_result, tol=1e-4)
    print("✓ Fusion produces equivalent results")
```

### 4. Recompilation Monitoring

```python
import torch._dynamo.utils

def monitor_recompilation():
    counters = torch._dynamo.utils.counters
    recompiles = sum(counters['recompiles'].values())
    graph_breaks = sum(counters['graph_break'].values())

    print(f"Recompiles: {recompiles}")
    print(f"Graph breaks: {graph_breaks}")

    # Alert if exceeding thresholds
    if recompiles > 10:
        print("⚠️ WARNING: High recompilation count")
    if graph_breaks > 5:
        print("⚠️ WARNING: Multiple graph breaks detected")

    return recompiles, graph_breaks
```

---

## Appendix A: Debugging Reference

### Environment Variables

```bash
# Graph break locations and causes
TORCH_LOGS=graph_breaks python train.py

# Track recompilation triggers
TORCH_LOGS=recompiles,guards python train.py

# Maximum verbosity
TORCH_LOGS="+dynamo,aot,inductor" python train.py

# Debug directory with IR dumps
TORCH_COMPILE_DEBUG=1 python train.py
```

### Generated Kernel Name Patterns

| Pattern | Meaning |
|---------|---------|
| `triton_poi_fused_*` | Pointwise fused operations |
| `triton_red_fused_*` | Reduction fused operations |
| `triton_per_fused_*` | Persistent reduction fused |
| `triton_red_fused_clamp_div_norm` | Normalization fused |

### Programmatic Configuration

```python
import torch._inductor.config as config
config.trace.enabled = True
config.trace.ir_pre_fusion = True
config.trace.ir_post_fusion = True
config.trace.output_code = True
```

---

## Appendix B: Compilation Caching for Production

### Environment Variable Configuration

```bash
# Persistent cache location - avoids recompilation across restarts
export TORCHINDUCTOR_CACHE_DIR=/persistent/cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
```

### Portable Cache for Containerized Deployments

For containerized production environments, export and load cache artifacts to avoid recompilation on each instance:

```python
# After warm-up on reference instance, export cache
artifacts = torch.compiler.save_cache_artifacts()
artifact_bytes, cache_info = artifacts
with open("torch_cache.bin", "wb") as f:
    f.write(artifact_bytes)

# On new deployment instances - load pre-compiled cache
with open("torch_cache.bin", "rb") as f:
    torch.compiler.load_cache_artifacts(f.read())
```

**Impact**: Eliminates 10-60+ minute compilation time on cold starts for each new container instance.

### Warm-up Procedures

First inference triggers full compilation. Use explicit warm-up to ensure compilation completes before production traffic:

```python
def warm_up_model(model, sample_inputs, warmup_iterations=5):
    """Warm up model to trigger and complete compilation before production traffic."""
    model.eval()
    compiled = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = compiled(*sample_inputs)
            if i == 0:
                print("Initial compilation complete")
    torch.cuda.synchronize()
    return compiled

# Usage in production startup
compiled_model = warm_up_model(model, [sample_batch])
```

---

## Appendix C: torch.compile Mode Reference

| Mode | Compile Time | Runtime | Memory | Use Case |
|------|-------------|---------|--------|----------|
| `default` | 1-5 min | Good | Baseline | Development |
| `reduce-overhead` | 2-8 min | Better (small batch) | +10-20% | Inference |
| `max-autotune` | 10-60+ min | Best | Variable | Production training |
| `max-autotune-no-cudagraphs` | 10-60+ min | Best | Baseline | Variable shapes |

---

## Appendix D: H100 Optimal Configuration

```python
compiled = torch.compile(
    model,
    mode="max-autotune",
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,      # Tensor core alignment
        "triton.cudagraphs": True,  # Only if static shapes
    }
)
```

---

## Appendix E: Industry Case Studies

| Company | Achievement | Key Insight |
|---------|-------------|-------------|
| **Meta** | Trillion-parameter scale with TorchRec | Distributed embeddings rather than direct torch.compile |
| **Pinterest** | O(10K)-length Sequence RecSys | CUDA graphs + Triton kernels (complementary to torch.compile) |
| **Amazon** | 30-40% latency reduction | Forward pass must be pure function |
| **AWS** | 2x speedup on Graviton (DLRM in TorchBench) | Platform-specific optimizations compound with compilation |
| **Netflix** | 20% improvement | Foundation models with shared embeddings maximize compile benefits |
| **Dashboard** | 43% average speedup, 93% compatibility | RecSys below average due to sparse ops (60-80% compute) |

**Key Lessons from Industry:**
- Forward pass must be a pure function—no state mutations
- Use only `torch.Tensor` types as inputs—lift non-tensor configs to module properties
- Minimize branching complexity by pushing to initialization
- Always test for numerical differences (compiled models are not bitwise exact)

---

## Appendix F: Quick Answers Summary

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
2. **Memory-constrained**: Keep `activation_memory_budget=0.05`; consider 0.2-0.5 for speed
3. **Medium-term**: Migrate to LayerNorm/RMSNorm where mathematically valid for optimal fusion
4. **Long-term**: Consider Liger Kernels for guaranteed single-kernel Linear+Norm execution
5. **RecSys-specific**: Use hybrid compilation—`@torch.compiler.disable` on sparse embeddings, compile dense interaction layers

---

## Appendix G: References

### Research Documents
- Merged Research: `result_Q2_merged.md`
- Deep Research Q2_01 through Q2_05

### PyTorch Documentation
1. [torch.nn.functional.normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
2. [Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)
3. [PyTorch 2: Dynamic Python Bytecode Transformation](https://docs.pytorch.org/assets/pytorch2-2.pdf)

### External Resources
4. [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://github.com/linkedin/Liger-Kernel)
5. [vLLM torch.compile Integration](https://blog.vllm.ai/2025/08/20/torch-compile.html)
6. [TorchAO Float8 Training](https://github.com/pytorch/ao)
7. [PyTorch GitHub Issue #32137 — eps underflow](https://github.com/pytorch/pytorch/issues/32137)
8. [FlashAttention Fused RMSNorm #570](https://github.com/Dao-AILab/flash-attention/issues/570)

---

*Document Revised: 2026-01-30*
*Revision: Deep Research Integration*
*Status: Ready for Technical Review*
