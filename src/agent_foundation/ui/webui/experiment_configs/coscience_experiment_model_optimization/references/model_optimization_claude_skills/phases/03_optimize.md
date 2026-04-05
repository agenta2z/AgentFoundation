# Phase 3: Optimization Iterations

Iteratively optimize forward and backward passes with profiler-guided changes.

---

<critical-rules ref="MEMORY_NEUTRAL">
**YOU ARE OPTIMIZING FOR MULTI-GPU CLUSTER TRAINING SPEED.**

**Goal**: Maximize forward/backward pass speed.
**Constraint**: Keep memory usage NEUTRAL (do not increase).

If an optimization increases memory:
→ ASK USER before applying (they may accept the trade-off)
→ Do NOT auto-reject, but do NOT auto-apply either
</critical-rules>

**Optimization Priority (START WITH LOW-HANGING FRUIT):**

| Priority | Optimization Type | Why | Effort |
|----------|------------------|-----|--------|
| 1 (FIRST) | **PyTorch glue code / CPU-bound fixes** | Low-hanging fruit: easy wins, low risk, memory-neutral | LOW |
| 2 | **Forward pass speedup** | Directly reduces training iteration time | MEDIUM |
| 3 | **Backward pass speedup** | On critical path, often 2x forward time | MEDIUM |
| 4 | Communication-computation overlap | Hides all-reduce latency in DDP | HIGH |
| 5 (LAST) | Kernel micro-optimization | Diminishing returns, requires deep expertise | HIGH |

**Memory Constraint**: All optimizations should be memory-neutral. If an optimization increases memory, ASK USER for guidance before applying.

---

<section id="LOW_HANGING_FRUIT">
## Low-Hanging Fruit - Do These First

**These optimizations are easy to find, low-risk, and often provide 2-10% speedup.**

### Quick Wins Checklist

Before trace analysis or kernel optimization, scan the code for these patterns:

| Pattern | What to Look For | Fix | Typical Speedup |
|---------|-----------------|-----|-----------------|
| **Inefficient indexing** | `torch.arange() + index_select()` | Use slice notation `tensor[:, start::step, :]` | 1-5% |
| **Redundant dtype conversion** | `.to(dtype=X)` inside `torch.autocast()` | Remove - autocast handles it automatically | 1-3% |
| **Unnecessary tensor creation** | `torch.zeros/ones/arange` in hot paths | Pre-allocate or use views | 1-5% |
| **CPU-GPU sync in loops** | `.item()`, `.cpu()`, `print(tensor)` | Move outside loop or remove | 5-20% |
| **Python loops over tensors** | `for i in range(tensor.size(0))` | Vectorize with torch ops | 10-50% |
| **Redundant clones** | `.clone()` when not needed | Remove if original not mutated | 1-3% |
| **Redundant contiguous** | `.contiguous()` on already contiguous tensor | Remove or check with `.is_contiguous()` | 1-2% |
| **Debug code in hot path** | `assert`, logging, shape checks | Remove or guard with `if self.training` | 1-5% |

### Example Low-Hanging Fruit Fixes

**Before:**
```python
# Inefficient: creates indices tensor, copies data
indices = torch.arange(start, end, step, device=x.device)
result = torch.index_select(x, dim=1, index=indices)
```

**After:**
```python
# Efficient: uses strided view, no allocation
result = x[:, start:end:step, :]
```

---

**Before:**
```python
# Redundant: autocast already handles dtype
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = module(x.to(dtype=torch.bfloat16))
```

**After:**
```python
# Clean: let autocast handle dtype conversion
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = module(x)
```

---

**Before:**
```python
# CPU-GPU sync in loop - very slow!
total = 0
for i in range(batch_size):
    total += tensor[i].item()  # Sync on every iteration!
```

**After:**
```python
# Vectorized - single GPU operation
total = tensor.sum().item()  # Single sync at the end
```

---

<section id="COMMON_MISTAKES">
## Common Mistakes - Avoid These

Before optimizing, review these anti-patterns:

| Anti-Pattern | Why It's Problematic | What To Do |
|--------------|---------------------|------------|
| **Use wrong buck mode (e.g., @//mode/dev-nosan instead of @mode/opt)** | Debug modes add overhead, making measurements invalid | ALWAYS use buck_run_command from state or read benchmark file header |
| **Guess benchmark CLI arguments** | Arguments vary by benchmark | Read benchmark file or run with --help first |
| **Disable gradient checkpointing without asking** | Trades memory for speed | ASK USER first - they may accept or reject |
| **Add caching that increases memory without asking** | Uses more GPU memory | ASK USER first - explain the trade-off |
| **Skip the memory impact check** | May cause OOM at scale | Always check memory impact |
| **Focus only on micro-optimizations** | Miss bigger wins in fwd/bwd | Prioritize glue code/CPU > fwd > bwd > overlap > kernel |
| **Ignore CPU-bounded code** | Python overhead can dominate | Check for `.item()`, loops, unnecessary sync |

**If an optimization increases memory**: Do NOT auto-apply. ASK USER with details:
- How much memory increase?
- What speedup does it provide?
- Let user decide if trade-off is acceptable for their cluster setup.

---

<section id="PREFLIGHT_CHECKLIST">
## Mandatory Pre-Flight Checklist

**YOU MUST COMPLETE ALL ITEMS BEFORE ANY OPTIMIZATION WORK.**

Document the following in your response BEFORE proceeding to optimization:

```
=== PHASE 3 PRE-FLIGHT CHECKLIST ===

1. I understand the priority order: glue code/CPU > fwd > bwd > comm-overlap > kernel
2. I will keep memory NEUTRAL, or ASK USER if an optimization increases memory
3. I have reviewed the "Common Mistakes" section above
4. I will complete the memory impact check for EVERY optimization
5. I will look for CPU-bounded issues (Python loops, .item(), unnecessary sync)

PyTorch Glue Code Review Completed:
- Issue 1: {description} at {location} - {proposed_fix}
- Issue 2: {description} at {location} - {proposed_fix}
- (list all findings or "None found")

Low-hanging fruit identified: {count}
Estimated impact: {high|medium|low}

=== PRE-FLIGHT COMPLETE - READY TO OPTIMIZE ===
```

**If you cannot produce this output, DO NOT proceed to optimization.**

---

## The Memory Impact Check

**MANDATORY: Check memory impact BEFORE applying ANY optimization.**

For each proposed optimization, document:

```
=== OPTIMIZATION CHECK: {optimization_name} ===

1. MEMORY IMPACT: Does this increase peak GPU memory?
   Answer: {Yes|No|Uncertain}
   Details: {explanation}
   If YES → ASK USER before applying (explain trade-off)
   If No/Uncertain → PROCEED

2. EXPECTED SPEEDUP: What improvement do you expect?
   Forward: {X}% faster
   Backward: {X}% faster

3. RISK LEVEL: Could this break numerics or cause instability?
   Answer: {Low|Medium|High}
   Details: {explanation}

DECISION: {PROCEED|ASK_USER|SKIP}
Reason: {explanation}

=== END OPTIMIZATION CHECK ===
```

**Memory trade-off decisions are for the USER to make, not auto-rejected.**

---

## Prerequisites

From Phase 2:
- `benchmark_path`: Path to working benchmark script
- `buck_target`: BUCK target for benchmark
- `buck_mode`: Buck mode (e.g., `@mode/opt`)
- `buck_run_command`: Full buck run command to use
- `gpu_device`: GPU index to use
- `baseline_metrics`: Baseline forward/backward timing
- `trace_json_path`: Path to trace.json generated during Phase 2

**Running benchmarks**: Use `buck_run_command` from state. If not available, read the benchmark file header for the correct command.

## Shared Modules Used

- `shared/gpu_check.md`: GPU availability verification before each benchmark run
- `shared/numerical_check.md`: Numerical equivalence verification after each optimization

---

## Step 0: PyTorch Glue Code Review (MANDATORY)

**Before any trace analysis or kernel optimization, review the Python code.**

Read through the target module's Python code and look for:

| Issue | What to Look For | Fix |
|-------|-----------------|-----|
| Redundant tensor copies | `.clone()`, `.contiguous()`, `.to()` in hot paths | Remove if unnecessary or batch operations |
| Unnecessary CPU-GPU sync | `.item()`, `.cpu()`, `print(tensor)` in forward/backward | Remove or move outside critical path |
| Python overhead in loops | `for` loops over batch/sequence dimensions | Vectorize or use `torch.vmap` |
| Repeated allocations | Tensor creation inside loops or forward() | Pre-allocate and reuse buffers |
| Inefficient indexing | Advanced indexing creating copies | Use views or in-place operations |
| Missing `torch.no_grad()` | Inference code without disabling gradients | Add context manager |
| Suboptimal dtype | Using float64 or unnecessary precision | Use float32/float16/bfloat16 |
| Non-contiguous tensors | Transpose/permute before compute-heavy ops | Add `.contiguous()` before kernels |

**Check PyTorch Configuration:**

```python
# TF32 for Ampere+ GPUs (significant speedup for matmul/conv)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cuDNN autotuning (helps if input sizes are consistent)
torch.backends.cudnn.benchmark = True

# Disable debug features in production
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
```

**Check for Distributed Training Overhead:**

- Gradient bucketing configuration (`bucket_cap_mb` in DDP)
- Gradient compression opportunities
- Overlap of communication with computation
- Unnecessary synchronization points (`torch.cuda.synchronize()`, `dist.barrier()`)

---

## Step 1: Trace Analysis

<action>
Use the trace JSON generated during Phase 2 (stored at `trace_json_path` in state, default: `./profiler_output/trace.json`).
</action>

### How to Analyze trace.json

**Option 1: Use Profiler Summary Output (Fastest)**

The benchmark script prints profiler summaries to stdout. Look for the tables:
```
PROFILER SUMMARY - CUDA Time
================================================================================
[Table sorted by cuda_time_total]

PROFILER SUMMARY - CPU Time
================================================================================
[Table sorted by cpu_time_total]
```

These tables show the top operations already aggregated - use these as the primary source.

**Option 2: Programmatic Analysis**

Parse the trace JSON to extract top operations:
```python
import json

with open("{trace_json_path}", "r") as f:
    trace = json.load(f)

# Filter for CUDA kernel events
cuda_events = [
    e for e in trace["traceEvents"]
    if e.get("cat") == "kernel" or e.get("cat") == "cuda_runtime"
]

# Sort by duration (dur is in microseconds)
sorted_events = sorted(cuda_events, key=lambda x: x.get("dur", 0), reverse=True)

# Print top 10 time-consuming operations
print("Top 10 CUDA operations by duration:")
for i, event in enumerate(sorted_events[:10]):
    name = event.get("name", "unknown")
    dur_ms = event.get("dur", 0) / 1000  # Convert to ms
    print(f"{i+1}. {name}: {dur_ms:.3f} ms")
```

### Document Findings

After analysis, document:
```
Forward Pass Analysis:
1. {op_1}: {time} ms - {source_location}
2. {op_2}: {time} ms - {source_location}
...
Total forward time: {time} ms

Backward Pass Analysis:
1. {op_1}: {time} ms - {source_location}
2. {op_2}: {time} ms - {source_location}
...
Total backward time: {time} ms
```

---

## Step 2: Apply Optimizations (One at a Time)

For each optimization attempt:

1. **Identify target**: Choose one operation to optimize
2. **4-Question Check**: Complete the MANDATORY cluster impact check (see above)
   - If REJECT: skip this optimization, try another
   - If PROCEED: continue to step 3
3. **Plan change**: Document the optimization approach
4. **Implement**: Make the code change
5. **Measure**: Run benchmark
6. **Verify numerics**: Run quick numerical check (see `shared/numerical_check.md`)
   - If FAIL: rollback immediately, skip to step 9
   - If PASS: continue to step 7
7. **Check diminishing returns**: Calculate improvement percentage
   - If improvement < 2% for 3 consecutive attempts, consider stopping
8. **Commit**: Create a commit for this optimization
9. **Record**: Log results in optimization_history (including failures and rejections)

**Commit message format:**
```
[perf] <module>: <optimization description>

- Before: <X> ms forward, <Y> ms backward
- After: <X'> ms forward, <Y'> ms backward
- Speedup: <Z>x
- Memory impact: [neutral/reduced/N/A]
```

---

## GPU Environment Consistency

Between all benchmark runs:
- Use the **same GPU device** (GPU index)
- Verify GPU is **idle** (< 5% utilization) before each run
- Check **similar memory state** before each run
- If conditions differ, results are invalid

---

## Track Progress

Maintain a progress table that includes MEMORY:

```
| Iter | Change Description | Forward (ms) | Backward (ms) | Memory | Cluster OK? | Num Check |
|------|-------------------|--------------|---------------|--------|-------------|-----------|
| 0    | Baseline          | 100.0        | 150.0         | 512 MB | N/A         | N/A       |
| 1    | {description}     | 95.0         | 145.0         | 512 MB | YES         | PASS      |
| 2    | {description}     | 90.0         | 140.0         | 480 MB | YES (memory-)| PASS     |
```

**"Cluster OK?" column must show YES for every applied optimization.**

---

## Rollback Strategy

If an optimization causes:
- Performance regression
- Numerical mismatch
- Instability

**Immediately rollback**:
1. Revert the commit for this optimization (`hg revert` or `sl revert`)
2. Verify benchmark returns to previous metrics
3. Document the failed attempt
4. Try alternative approach

**If optimization increases memory**: Do not auto-rollback. ASK USER if they want to keep it (trade-off decision).

---

## Diminishing Returns Threshold

Stop optimization when improvements become marginal:

- **Forward optimization**: Stop when improvement < 2% for 3 consecutive attempts
- **Backward optimization**: Stop when improvement < 2% for 3 consecutive attempts
- **Overall**: Stop when total improvement < 1% for 3 consecutive attempts

---

## Numerical Equivalence After Each Optimization

Verify numerical equivalence immediately after each optimization:

1. Run quick numerical check (see `shared/numerical_check.md`)
2. Compare optimized output vs baseline with tolerance `rtol=1e-4, atol=1e-4`
3. Compare gradients with same tolerance

```python
# Quick numerical check after optimization
passed, details = quick_numerical_check(
    model_baseline=baseline_model,
    model_optimized=optimized_model,
    inputs=benchmark_inputs,
    rtol=1e-4,
    atol=1e-4,
)

if not passed:
    print(f"FAIL: forward_diff={details['forward_max_diff']:.2e}")
    # Rollback this optimization
else:
    print(f"PASS: forward_diff={details['forward_max_diff']:.2e}")
    # Proceed to commit
```

---

## Phase 3 Outputs

<output-format>
Update workflow state:

```yaml
optimization_history:
  - iteration: 1
    optimization: "{description}"
    status: "{success|failed|rejected_for_cluster}"
    cluster_impact_check:
      memory_delta: "{+/- X MB or neutral}"
      affects_gradient_sync: "{Yes|No}"
      expected_scaling: "{linear|sublinear|none}"
      risk_at_scale: "{low|medium|high}"
    before_ms: {value}
    after_ms: {value}
    speedup: "{X}x"
    equivalent: true
  - iteration: 2
    ...

current_best:
  forward_ms: {value}
  backward_ms: {value}
  total_ms: {value}
  memory_mb: {value}
  speedup_vs_baseline: "{X}x"
```
</output-format>

---

## Completion Checkpoint

```
Phase 3 Complete: Optimization Iterations

Pre-flight checklist: COMPLETED
Cluster impact checks: {count} performed for each optimization

Optimization Summary:
- Total iterations: {count}
- Optimizations applied: {count}
- Optimizations rejected (cluster impact): {count}

Performance Improvement:
| Metric   | Baseline | Optimized | Speedup | Memory |
|----------|----------|-----------|---------|--------|
| Forward  | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
| Backward | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
| Total    | {X} ms   | {Y} ms    | {Z}x    | {M} MB |

Proceeding to Phase 4: Validation...
```

**Proceed to Phase 4 automatically.**

---

## Error Handling

### No Improvement After Multiple Attempts

After 3 optimization attempts with no meaningful improvement, log and proceed:

```
Note: Tried 3 optimization approaches without meaningful improvement:
1. [approach 1]: [result]
2. [approach 2]: [result]
3. [approach 3]: [result]

The code may already be well-optimized. Proceeding to validation with current gains.
```

### Numerical Equivalence Failures

If optimization breaks numerical equivalence:
1. **Automatically rollback** the problematic optimization
2. Log the failure and try a different approach
3. Continue with remaining optimizations

Only stop if ALL optimization approaches fail numerical equivalence after 10 total attempts.
