# Phase 1: Setup

Gather optimization target information and understand the code structure.

## Prerequisites

- User has identified a module/operation to optimize from MAST training
- User has access to the MAST run trace or knows the bottleneck location

## Steps

### Step 1.0: Proactive Code Exploration (Before Prompting User)

<critical-rules>
<constraint type="silent-exploration">Gather all information silently first. Save questions for Step 1.1.</constraint>
</critical-rules>

**1.0.1: Check for Existing State**

Search for existing optimization state files:
```bash
find . -name ".qps_optimization_state.yaml" -type f 2>/dev/null | head -5
```

Store findings for Step 1.1 (save prompts for later).

**1.0.2: Search for Existing Benchmarks**

Before creating a new benchmark, search for existing ones:
```bash
find . -name "benchmark_*.py" -type f 2>/dev/null | head -10
find . -path "*/tests/benchmark*.py" -type f 2>/dev/null | head -10
```

Store findings for Step 1.1.

**1.0.3: If User Provided Partial Info**

If user mentions a module name but not the full path:
```bash
find . -name "*{module_name}*.py" -type f 2>/dev/null | head -10
grep -r "class {ClassName}" --include="*.py" -l 2>/dev/null | head -10
grep -r "def {function_name}" --include="*.py" -l 2>/dev/null | head -10
```

**1.0.4: Explore Related Files**

Once source file is identified, automatically explore:
```bash
find $(dirname {source_path}) -name "test_*.py" -o -name "*_test.py" | head -5
ls -la $(dirname {source_path})/*.py | head -10
grep -r "profiler\|benchmark\|perf" $(dirname {source_path}) --include="*.py" -l | head -5
```

### Step 1.1: Consolidated User Prompt

<decision-tree>
**Present ONE prompt based on exploration results:**

IF existing_state_found AND module_candidates_found:
  → Use Template A (resume + candidates)
ELSE IF existing_state_found:
  → Use Template B (resume only)
ELSE IF module_candidates_found:
  → Use Template C (candidates only)
ELSE:
  → Use Template D (fresh start)
</decision-tree>

**Template A (Resume + Candidates):**
```
I found the following in your codebase:

Existing optimization state at {state_path}:
- Source: {source_path}
- Phase: {current_phase}

Potential matches for "{module_name}":
1. {path1} - contains {class_or_function1} (line {N1})
2. {path2} - contains {class_or_function2} (line {N2})

Please confirm:
1. Resume existing state or start fresh?
2. Which source file/module to optimize?
3. Which entry point?
```

**Template B (Resume Only):**
```
Found existing optimization state at {state_path}:
- Source: {source_path}
- Phase: {current_phase}
- Progress: {progress_summary}

Resume this optimization, or start fresh?
```

**Template C (Candidates Only):**
```
I found potential matches for "{module_name}":
1. {path1} - contains {class_or_function1} (line {N1})
2. {path2} - contains {class_or_function2} (line {N2})

Detected entry points:
1. {entry_point1} (line {M1})
2. {entry_point2} (line {M2})

Please confirm:
1. Which source file/module to optimize? [default: option 1]
2. Which entry point? [default: forward() if present]
```

**Template D (Fresh Start):**
```
Please provide the optimization target:

1. **Source file path**: Absolute path to the Python file
   Example: /data/users/username/fbsource/fbcode/path/to/module.py

2. **Entry point**: Function or class to optimize
   Example: forward, MyModule.compute, attention_forward
```

**Follow-up if incomplete:**
```
I need a bit more information:
- You provided {what_they_gave}, but I also need {what_missing}
- Could you clarify {specific_question}?
```

### Step 1.2: Read and Understand the Code

<action>
Use the Read tool to load the source file.
</action>

Once you have the source path:

1. **Read the source file**:
   - Use `Read("{source_path}")` to read the entire file
   - If the file is large (>500 lines), focus on the entry point and its dependencies

2. **Identify key components**:
   - Entry point function/class location (line numbers)
   - Input parameters and their types
   - Output structure
   - Key computations and operations
   - Dependencies on other modules

3. **Document the structure**:
   ```
   Code Structure Analysis:
   - File: {source_path}
   - Entry point: {function_or_class} at line {N}
   - Input signature: {parameters}
   - Output type: {type}
   - Key operations:
     1. {operation_1} - line {N1}
     2. {operation_2} - line {N2}
   - Dependencies: {list_of_imports}
   ```

### Step 1.3: Identify Optimization Opportunities

Based on code analysis, identify potential optimization targets:

1. **Compute-intensive operations**:
   - Matrix multiplications
   - Attention computations
   - Activation functions
   - Custom CUDA kernels

2. **Memory-intensive operations**:
   - Large tensor allocations
   - Repeated tensor copies
   - Inefficient memory access patterns

3. **Potential optimizations** (per `MEMORY_NEUTRAL` constraint):
   - Operator fusion opportunities
   - Memory layout improvements
   - Batching/parallelization
   - Algorithm changes

Document findings:
```
Optimization Opportunities:
1. {opportunity_1}: {description} - potential impact: {high|medium|low}
2. {opportunity_2}: {description} - potential impact: {high|medium|low}
```

---

## Output Format

<output-format>
After completing Phase 1, produce this summary:

```
Phase 1 Complete: Setup

Summary:
- Source file: {source_path}
- Entry point: {entry_point}
- Key operations identified: {count}
- Optimization opportunities: {count}

Proceeding to Phase 2: Create Benchmark Tool...
```

Update state file with:
```yaml
source_path: "{source_path}"
entry_point: "{entry_point}"
code_understanding: |
  {summary_of_code_structure}
  {key_operations_identified}
  {optimization_opportunities}
```
</output-format>

<phase-transition>Proceed to Phase 2 automatically.</phase-transition>

---

## Error Handling

### Missing Source File → `STOP_NO_SOURCE`

<critical-stop ref="STOP_NO_SOURCE">
The source file at {path} does not exist or is not accessible.

Please verify:
1. The path is correct and absolute
2. You have read access to the file
3. The file extension is correct (.py)

Provide the correct path to continue.
</critical-stop>

### Entry Point Not Found (Auto-recover)

<decision-tree>
IF entry_point not found in source:
  → Search for similar names
  → IF multiple candidates found:
      → Log: "Couldn't find {entry_point}. Available: {list}. Assuming {best_match}."
      → Proceed with best_match
  → ELSE:
      → Ask user for clarification
</decision-tree>

### Complex Dependencies (Log and proceed)

```
Note: The entry point {entry_point} has complex dependencies:
- {dep_1}
- {dep_2}

These will be handled during benchmark creation.
```
# Phase 2: Create Benchmark Tool

Create a standalone benchmark script with profiling capabilities and establish baseline metrics.

## Prerequisites

From Phase 1:
- `source_path`: Path to source file
- `entry_point`: Function/class to benchmark
- `code_understanding`: Summary of code structure

<rule>
<rule>Create benchmark as fresh, standalone code. Never modify the source being tested.</rule>
<rule>When debugging, only modify benchmark tool code, never the source module.</rule>
</rule>

## Steps

### Step 2.1: Verify GPU Availability

<action>
Use Bash to run `nvidia-smi` and find an idle GPU (< 5% utilization).
</action>

Follow the instructions in `shared/gpu_check.md`:
1. Run `nvidia-smi` to check GPU status
2. Find an idle GPU (< 5% utilization)
3. If no idle GPU → trigger `STOP_NO_GPU`

Record the GPU to use:
```yaml
gpu_device: "{gpu_index}"
```

### Step 2.2: Create Benchmark File

<action>
Use the Write tool to create the benchmark script.
</action>

<rule>Before writing the benchmark, output a `<reasoning>` block (see `shared/reasoning_block.md`).</rule>

<section>File Location:</section>
- Create in `tests/` subdirectory under source file's directory
- Naming: `benchmark_{module_name}.py`
- Example: If source is `/path/to/module.py`, create `/path/to/tests/benchmark_module.py`

<action>Use the benchmark template from `shared/benchmark_template.md`:</action>
1. Copy the full Python template from the shared module
2. Customize `create_benchmark_inputs()` based on Phase 1 code analysis
3. Customize `create_model()` to instantiate your target module
4. Adjust command-line arguments as needed

Key customization points:
- **`create_benchmark_inputs()`**: Match the exact input signature of the target module's forward()
- **`create_model()`**: Import and instantiate the target module with correct constructor arguments

### Step 2.3: Create BUCK Target

<action>
Use the Edit or Write tool to update the BUCK file.
</action>

Create or update BUCK file in the `tests/` directory.

<action>Use the BUCK configuration from `shared/benchmark_template.md`:</action>

| Setting | Requirement | Why |
|---------|-------------|-----|
| `keep_gpu_sections = True` | **Mandatory** | Preserves CUDA kernels |
| `deps` includes source target | **Mandatory** | Include BUCK target of source module for transitive deps |
| `@mode/opt` | **Required** | Debug mode adds overhead |
| `gpu_python_binary` | Consider for complex GPU deps | Alternative for heavy GPU usage |

### Step 2.4: Debug and Run Benchmark

<action>
Use Bash to run the benchmark with buck.
</action>

<action>Run command:</action>
```bash
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target}
```

<note>For debugging guidelines, refer to `shared/benchmark_template.md`:</note>
- Focus on mocking input logic in `create_benchmark_inputs()`
- Fix model initialization in `create_model()`
- Update BUCK dependencies as needed

<rule>Max debugging attempts: 10 total across all error types. After 10 failures → trigger `STOP_BENCHMARK_FAILED`.</rule>

### Step 2.4.1: Benchmark Timeout Handling

<rule>
<warning>Benchmarks can hang due to profiler issues, CUDA errors, or infinite loops.</warning>
</rule>

<section>Timeout Policy:</section>
- If benchmark produces no output within 120 seconds, consider it hung
- Kill the process and count as 1 failed attempt toward the 10-attempt limit

<reference>Common Hang Causes and Fixes:</reference>

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Hangs during profiler | Too many profiler iterations | Reduce to 2-3 iterations |
| Hangs at CUDA init | GPU memory exhausted | Reduce batch size |
| Hangs at import | Circular import or slow module load | Check BUCK deps |
| 100% CPU after benchmark | Profiler post-processing | Reduce profiler iterations |

<output>When timeout occurs:</output>
```
Benchmark timed out after 120 seconds.
Attempting fix: {attempted_fix_description}
Attempts remaining: {N}/10
```

<note>If profiler hang suspected: Add `--skip-profile` flag temporarily to verify benchmark runs without profiler.</note>

### Step 2.5: Capture Baseline Metrics and Trace

<action>
Run benchmark multiple times and record metrics.
</action>

Once benchmark runs successfully:

1. Run benchmark multiple times to ensure consistency
2. Verify GPU environment consistency (see `shared/gpu_check.md`)
3. **The benchmark automatically generates `trace.json`** (profiler runs by default)
4. **Check if backward pass was auto-detected as unnecessary** (see output)
5. Record baseline metrics and trace location:

### Interpreting Backward Pass Results

The benchmark tool auto-detects whether backward pass is needed:

| Output | Meaning | Action in Phase 3 |
|--------|---------|-------------------|
| `Backward pass: X.XX ± Y.YY ms` | Backward is active | Optimize both forward and backward |
| `Backward pass: SKIPPED (reason)` | Module doesn't need backward | Skip backward optimization |
| `backward_skipped: True` | Auto-detected as inference-only | Focus on forward only |

Common auto-detection reasons:
- `"All N parameters are frozen (requires_grad=False)"` - Frozen model
- `"No input tensors require gradients"` - Inference inputs
- `"Output tensor does not require grad"` - Detached or no_grad output
- `"Module has no parameters"` - Pure functional module

### Record Baseline Metrics

```yaml
baseline_metrics:
  forward_ms: {mean_value}
  forward_std_ms: {std_value}
  backward_ms: {mean_value}  # 0.0 if skipped
  backward_std_ms: {std_value}  # 0.0 if skipped
  backward_skipped: {true|false}
  backward_skip_reason: "{reason if skipped}"
  total_ms: {forward_plus_backward}
  memory_mb: {value}
trace_json_path: "./profiler_output/trace.json"
```

<note>The trace.json file is required for Phase 3 optimization. If backward was skipped, the trace only contains forward pass profiling.</note>

### Step 2.5.1: Reproducibility Verification

<rule>
<rule>Before proceeding, verify that benchmark results are reproducible and have low variance.</rule>
</rule>

<action>Run 3 Independent Benchmark Executions:</action>

```bash
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run1.log
sleep 5
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run2.log
sleep 5
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run3.log
```

<output>Analyze Variance Across Runs:</output>

```
Run 1: forward={X1} ms, backward={Y1} ms (or SKIPPED)
Run 2: forward={X2} ms, backward={Y2} ms (or SKIPPED)
Run 3: forward={X3} ms, backward={Y3} ms (or SKIPPED)

Mean forward:  {mean} ms
Std forward:   {std} ms
CV forward:    {cv}%

Mean backward: {mean} ms (or N/A if skipped)
Std backward:  {std} ms (or N/A if skipped)
CV backward:   {cv}% (or N/A if skipped)
```

<reference>Reproducibility Criteria:</reference>

| Coefficient of Variation (CV) | Action |
|-------------------------------|--------|
| < 5% | PASS - proceed |
| 5-10% | WARNING - proceed with caution |
| > 10% | FAIL - investigate before proceeding |

<decision-tree>
IF CV > 10%:
  → Check: GPU truly idle (< 5% utilization)?
  → Check: thermal throttling? (`nvidia-smi -q -d TEMPERATURE`)
  → Check: consistent memory state?
  → Check: background processes?
  → Fix issues and re-run
</decision-tree>

### Step 2.6: Commit Benchmark

<action>
Use Bash to commit the benchmark with hg or sl.
</action>

<action>Commit the benchmark tool as a separate commit before starting optimizations.</action>

```bash
# If backward is active:
hg add {benchmark_path} {buck_file}
hg commit -m "[benchmark] Add benchmark for {module_name}

- Target: {entry_point}
- Baseline forward: {forward_ms} ms
- Baseline backward: {backward_ms} ms"

# If backward was skipped:
hg add {benchmark_path} {buck_file}
hg commit -m "[benchmark] Add benchmark for {module_name}

- Target: {entry_point}
- Baseline forward: {forward_ms} ms
- Backward: skipped ({reason})"
```

---

## Output Format

<output>
After completing Phase 2, produce this summary:

```
Phase 2 Complete: Benchmark Created

Summary:
- Benchmark script: {benchmark_path}
- BUCK target: {buck_target}
- GPU used: {gpu_device}
- Commit: {benchmark_commit}
- Trace file: {trace_json_path}

Baseline Metrics:
- Forward:  {forward_ms} ± {forward_std_ms} ms
- Backward: {backward_ms} ± {backward_std_ms} ms (or "SKIPPED: {reason}")
- Total:    {total_ms} ms

Backward Pass Status:
- Required: {YES|NO}
- Reason: {auto-detection reason if skipped}

Reproducibility Check:
- Forward CV:  {forward_cv}% - {PASS|WARNING|FAIL}
- Backward CV: {backward_cv}% - {PASS|WARNING|FAIL|N/A if skipped}
- Overall: {PASS|WARNING|FAIL}

Proceeding to Phase 3: Optimization Iterations...
```

Update state file with:
```yaml
benchmark_path: "{benchmark_path}"
buck_target: "{buck_target}"
buck_mode: "@mode/opt"
buck_run_command: "buck run @mode/opt {buck_target}"
gpu_device: "{gpu_device}"
benchmark_commit: "{commit_hash}"
trace_json_path: "./profiler_output/trace.json"
baseline_metrics:
  forward_ms: {value}
  forward_std_ms: {value}
  backward_ms: {value}  # 0.0 if skipped
  backward_std_ms: {value}  # 0.0 if skipped
  backward_skipped: {true|false}
  backward_skip_reason: "{reason if skipped}"
  memory_mb: {value}
reproducibility:
  forward_cv_percent: {value}
  backward_cv_percent: {value}  # null if skipped
  status: "{PASS|WARNING|FAIL}"
```
</output>

<phase-transition>Proceed to Phase 3 automatically.</phase-transition>

---

## Error Handling

### No Idle GPU → `STOP_NO_GPU`

<action type="stop-no-gpu">
No idle GPU available. Current GPU status:
{nvidia_smi_output}

All GPUs are currently in use. Benchmark results would have high variance.

Please either:
1. Wait for other jobs to complete
2. Free up GPU resources on one of the devices
3. Specify which GPU to wait for

Reply when ready to continue.
</action>

### Benchmark Failed After 10 Attempts → `STOP_BENCHMARK_FAILED`

<action type="stop-benchmark-failed">
Benchmark failed after 10 attempts.

The benchmark is failing with: {error_message}

Attempts made:
1. {attempt_1}: {result}
2. {attempt_2}: {result}
...
10. {attempt_10}: {result}

I need your help to:
1. Verify expected tensor shapes for inputs
2. Confirm how the module is initialized in production
3. Check if there are special input formats required

Please provide guidance to continue.
</action>

### High Variance Results (Log warning, proceed)

```
WARNING: High variance in benchmark results:
- Forward: {mean} ± {std} ms (std = {percent}% of mean)

This may indicate GPU contention. Proceeding with caution...
```
# Phase 3: Optimization Iterations

Iteratively optimize forward and backward passes with profiler-guided changes.

---

<rule>
<goal>Optimizing for multi-GPU cluster training speed.</goal>

<goal>Maximize forward/backward pass speed.</goal>
<rule>Keep memory usage neutral (avoid increases).</rule>

<rule>
If an optimization increases memory:
→ Ask user before applying (they may accept the trade-off)
→ Avoid auto-rejecting and auto-applying; let user decide
</rule>
</rule>

<section>Optimization Priority (start with low-hanging fruit):</section>

| Priority | Optimization Type | Why | Effort |
|----------|------------------|-----|--------|
| 1 (FIRST) | **PyTorch glue code / CPU-bound fixes** | Low-hanging fruit: easy wins, low risk, memory-neutral | Low |
| 2 | **Forward pass speedup** | Directly reduces training iteration time | Medium |
| 3 | **Backward pass speedup** | On critical path, often 2x forward time (skip if module doesn't need backward) | Medium |
| 4 | Communication-computation overlap | Hides all-reduce latency in DDP | High |
| 5 (LAST) | Kernel micro-optimization | Diminishing returns, requires deep expertise | High |

<note>If backward was auto-detected as not needed (Step 0), skip Priority 3 entirely. Focus optimization effort on forward pass and glue code.</note>

<rule>All optimizations should be memory-neutral. If an optimization increases memory, ask user for guidance before applying.</rule>

---

<section>
## Low-Hanging Fruit - Do These First

<note>These optimizations are easy to find, low-risk, and often provide 2-10% speedup.</note>

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

<section>
## Common Mistakes - Avoid These

Before optimizing, review these anti-patterns:

| Anti-Pattern | Why It's Problematic | What To Do |
|--------------|---------------------|------------|
| **Use wrong buck mode (e.g., @//mode/dev-nosan instead of @mode/opt)** | Debug modes add overhead, making measurements invalid | Always use buck_run_command from state or read benchmark file header |
| **Guess benchmark CLI arguments** | Arguments vary by benchmark | Read benchmark file or run with --help first |
| **Disable gradient checkpointing without asking** | Trades memory for speed | Ask user first - they may accept or reject |
| **Add caching that increases memory without asking** | Uses more GPU memory | Ask user first - explain the trade-off |
| **Skip the memory impact check** | May cause OOM at scale | Always check memory impact |
| **Focus only on micro-optimizations** | Miss bigger wins in fwd/bwd | Prioritize glue code/CPU > fwd > bwd > overlap > kernel |
| **Ignore CPU-bounded code** | Python overhead can dominate | Check for `.item()`, loops, unnecessary sync |

<rule>If an optimization increases memory: Ask user with details before applying:
- How much memory increase?
- What speedup does it provide?
- Let user decide if trade-off is acceptable for their cluster setup.
</rule>

---

<section>
## Mandatory Pre-Flight Checklist

<rule>Complete all items in this checklist before any optimization work.</rule>

Document the following in your response before proceeding to optimization:

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

<rule>Ensure you can produce this output before proceeding to optimization.</rule>

---

## The Memory Impact Check

<rule>Check memory impact before applying any optimization.</rule>

For each proposed optimization, document:

```
=== OPTIMIZATION CHECK: {optimization_name} ===

1. MEMORY IMPACT: Does this increase peak GPU memory?
   Answer: {Yes|No|Uncertain}
   Details: {explanation}
   If YES → Ask user before applying (explain trade-off)
   If No/Uncertain → Proceed

2. EXPECTED SPEEDUP: What improvement do you expect?
   Forward: {X}% faster
   Backward: {X}% faster

3. RISK LEVEL: Could this break numerics or cause instability?
   Answer: {Low|Medium|High}
   Details: {explanation}

DECISION: {Proceed|Ask_User|Skip}
Reason: {explanation}

=== END OPTIMIZATION CHECK ===
```

<rule>Memory trade-off decisions are for the user to make, not auto-rejected.</rule>

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

<action>Running benchmarks: Use `buck_run_command` from state. If not available, read the benchmark file header for the correct command.</action>

## Shared Modules Used

- `shared/gpu_check.md`: GPU availability verification before each benchmark run
- `shared/numerical_check.md`: Numerical equivalence verification after each optimization

---

## Step 0: Backward Pass Necessity Check

<rule>Before optimizing backward pass, verify it is actually needed for this module.</rule>

The benchmark tool auto-detects whether backward pass is required. Check the benchmark output from Phase 2.

### Interpreting Benchmark Results

| Benchmark Output | Meaning | Action |
|-----------------|---------|--------|
| `Backward pass: SKIPPED (reason)` | Module doesn't need backward | Skip backward optimization entirely |
| `backward_ms: 0.0` with `backward_skipped: True` | Auto-detected as inference-only | Focus on forward optimization only |
| `backward_ms: X.XX` (non-zero) | Backward pass is active | Optimize both forward and backward |

### Patterns That Indicate No Backward Needed

| Pattern | Detection | Example |
|---------|-----------|---------|
| No trainable parameters | All `param.requires_grad == False` | Frozen pretrained model |
| No parameters at all | Module has no `nn.Parameter` | Pure functional module |
| Output is detached | `return output.detach()` in forward | Inference-only module |
| `torch.no_grad()` context | Forward wrapped in no_grad | Embedding lookup, preprocessing |
| `@torch.inference_mode()` | Decorator on forward method | Inference-optimized module |
| Inputs don't require grad | All inputs have `requires_grad=False` | Fixed input features |

### Document Backward Status

<output>
Before proceeding to optimization, document:

```
=== BACKWARD PASS CHECK ===
Benchmark reported: {backward_ms value or "SKIPPED"}
Backward required: {YES|NO}
Reason: {reason from auto-detection or analysis}
Action: {Optimize both forward+backward | Focus on forward only}
=== END BACKWARD CHECK ===
```
</output>

<decision-tree>
IF backward_skipped == True OR backward_ms == 0.0:
  → Update optimization priority to skip backward entirely
  → Remove backward from progress tracking table
  → Focus all effort on forward pass optimization
  → In commit messages, only report forward metrics

IF backward is needed:
  → Proceed with standard forward + backward optimization
  → Track both metrics in progress table
</decision-tree>

---

## Step 0.5: PyTorch Glue Code Review (Required First)

<rule>Before any trace analysis or kernel optimization, review the Python code.</rule>

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

<action>Check PyTorch Configuration:</action>

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

<action>Check for Distributed Training Overhead:</action>

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

<rule>Before writing any code, output a `<reasoning>` block (see `shared/reasoning_block.md`).</rule>

For each optimization attempt:

1. **Identify target**: Choose one operation to optimize
2. **4-Question Check**: Complete the required cluster impact check (see above)
   - If reject: skip this optimization, try another
   - If proceed: continue to step 3
3. **Plan change**: Document the optimization approach
4. **Output reasoning block**: Before implementing, output `<reasoning>` (see `shared/reasoning_block.md`)
5. **Implement**: Make the code change
6. **Measure**: Run benchmark
7. **Verify numerics**: Run quick numerical check (see `shared/numerical_check.md`)
   - If FAIL: rollback immediately, skip to step 10
   - If PASS: continue to step 8
8. **Check diminishing returns**: Calculate improvement percentage
   - If improvement < 2% for 3 consecutive attempts, consider stopping
9. **Commit**: Create a commit for this optimization
10. **Record**: Log results in optimization_history (including failures and rejections)

<section>Commit message format:</section>
```
[perf] {module}: {optimization_description}

- Before: {X} ms forward, {Y} ms backward
- After: {X'} ms forward, {Y'} ms backward
- Speedup: {Z}x
- Memory impact: [neutral/reduced/N/A]
```

---

## GPU Environment Consistency

<section>
Between all benchmark runs:
- Use the same GPU device (GPU index)
- Verify GPU is idle (< 5% utilization) before each run
- Check similar memory state before each run
- If conditions differ, results are invalid
</section>

---

## Track Progress

Maintain a progress table that includes MEMORY:

### If Backward Required (Standard Case)

```
| Iter | Change Description | Forward (ms) | Backward (ms) | Memory | Cluster OK? | Num Check |
|------|-------------------|--------------|---------------|--------|-------------|-----------|
| 0    | Baseline          | 100.0        | 150.0         | 512 MB | N/A         | N/A       |
| 1    | {description}     | 95.0         | 145.0         | 512 MB | YES         | PASS      |
| 2    | {description}     | 90.0         | 140.0         | 480 MB | YES (memory-)| PASS     |
```

### If Backward Skipped (Inference-Only Module)

```
| Iter | Change Description | Forward (ms) | Memory | Cluster OK? | Num Check |
|------|-------------------|--------------|--------|-------------|-----------|
| 0    | Baseline          | 100.0        | 512 MB | N/A         | N/A       |
| 1    | {description}     | 95.0         | 512 MB | YES         | PASS      |
| 2    | {description}     | 90.0         | 480 MB | YES (memory-)| PASS     |

Note: Backward optimization skipped - module does not require gradients.
Reason: {backward_skip_reason from Phase 2}
```

<rule>"Cluster OK?" column must show YES for every applied optimization.</rule>

---

## Rollback Strategy

If an optimization causes:
- Performance regression
- Numerical mismatch
- Instability

<action type="rollback">Immediately rollback:</action>
1. Revert the commit for this optimization (`hg revert` or `sl revert`)
2. Verify benchmark returns to previous metrics
3. Document the failed attempt
4. Try alternative approach

<rule>If optimization increases memory: Preserve the change and ask user if they want to keep it (trade-off decision).</rule>

---

## Diminishing Returns Threshold

<section>
Stop optimization when improvements become marginal:
- Forward optimization: Stop when improvement < 2% for 3 consecutive attempts
- Backward optimization: Stop when improvement < 2% for 3 consecutive attempts
- Overall: Stop when total improvement < 1% for 3 consecutive attempts
</section>

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

<output>
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
</output>

---

## Completion Checkpoint

```
Phase 3 Complete: Optimization Iterations

Pre-flight checklist: COMPLETED
Backward pass check: {REQUIRED|SKIPPED (reason)}
Cluster impact checks: {count} performed for each optimization

Optimization Summary:
- Total iterations: {count}
- Optimizations applied: {count}
- Optimizations rejected (cluster impact): {count}

Performance Improvement:
```

### If Backward Required
```
| Metric   | Baseline | Optimized | Speedup | Memory |
|----------|----------|-----------|---------|--------|
| Forward  | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
| Backward | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
| Total    | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
```

### If Backward Skipped
```
| Metric   | Baseline | Optimized | Speedup | Memory |
|----------|----------|-----------|---------|--------|
| Forward  | {X} ms   | {Y} ms    | {Z}x    | {M} MB |
| Backward | SKIPPED  | SKIPPED   | N/A     | N/A    |
| Total    | {X} ms   | {Y} ms    | {Z}x    | {M} MB |

Backward skipped because: {reason from auto-detection}
```

```
Proceeding to Phase 4: Validation...
```

<phase-transition>Proceed to Phase 4 automatically.</phase-transition>

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
<action type="auto-rollback">Automatically rollback the problematic optimization</action>
2. Log the failure and try a different approach
3. Continue with remaining optimizations

Only stop if ALL optimization approaches fail numerical equivalence after 10 total attempts.
# Phase 4: Validation

Verify numerical equivalence and validate performance gains in full MAST training.

## Prerequisites

From Phase 3:
- `optimization_history`: List of applied optimizations
- `current_best`: Best achieved metrics
- Working optimized code

## Shared Modules Used

- `shared/numerical_check.md`: Numerical equivalence verification functions and tolerance guidelines

## Step 4.1: Numerical Equivalence Verification

Ensure optimizations preserve numerical correctness.

### 4.1.1: Exact Match Check (When Possible)

For deterministic operations:
```python
assert torch.equal(baseline_output, optimized_output)
```

### 4.1.2: Approximate Match Check (Floating Point)

For floating-point operations:
```python
torch.testing.assert_close(
    baseline_output,
    optimized_output,
    rtol=1e-5,  # relative tolerance
    atol=1e-5,  # absolute tolerance
)
```

### 4.1.3: Gradient Equivalence (For Training)

Verify gradients match:
```python
torch.testing.assert_close(
    baseline_grad,
    optimized_grad,
    rtol=1e-4,
    atol=1e-4,
)
```

### 4.1.4: Statistical Equivalence (Stochastic Operations)

For operations with randomness:
- Run multiple iterations
- Compare distributions of outputs
- Verify loss curves converge similarly

### 4.1.5: Edge Case Testing

Test with:
- Various input shapes (small, large, edge cases)
- Edge values (zeros, very large/small values)
- Different dtypes (fp32, fp16, bf16)

### 4.1.6: Document Results

```
Numerical Equivalence Verification:

Forward Output:
- Exact match: {Yes|No}
- Max absolute diff: {value}
- Max relative diff: {value}
- Tolerance used: rtol={X}, atol={Y}
- Status: {PASS|FAIL}

Gradient:
- Max absolute diff: {value}
- Max relative diff: {value}
- Status: {PASS|FAIL}

Edge Cases Tested:
- {case_1}: {PASS|FAIL}
- {case_2}: {PASS|FAIL}

Overall: {PASS|FAIL}
```

---

## Step 4.2: Verify Training Performance Efficiency

Validate improvements in local training context before MAST.

### 4.2.1: Run Optimized Code in Training Mode

Test the optimized code path in a realistic training scenario:
- Use representative batch sizes
- Run multiple training steps
- Monitor throughput and memory

### 4.2.2: Compare Against Baseline

```
Local Training Efficiency Verification:

| Metric              | Baseline | Optimized | Improvement |
|---------------------|----------|-----------|-------------|
| Step time           | {X} ms   | {Y} ms    | {Z}%        |
| Throughput          | {X}/sec  | {Y}/sec   | {Z}%        |
| Memory usage        | {X} MB   | {Y} MB    | {Z}%        |
| Training stability  | OK       | OK        | -           |
```

---

## Step 4.3: MAST Job Validation

Validate the optimization in a full MAST training run.

### 4.3.1: Request MAST CLI from User

<critical-stop ref="STOP_MAST_CLI">
To validate the optimization in full MAST training, I need the MAST run CLI command.

Please provide:
1. The MAST run CLI command to trigger a training job
   Example: mast run --config /path/to/config.yaml --experiment my_experiment

2. The baseline MAST run metrics (if available):
   - Baseline QPS: {value}
   - Baseline step time: {value}
</critical-stop>

<action type="wait">Stop and wait for user to provide MAST CLI.</action>

### 4.3.2: Execute MAST Job

Once user provides CLI:

1. Run the MAST job with the user-provided command
2. Monitor the training run for:
   - QPS (queries per second) metrics
   - Step time
   - Memory usage
   - Training stability

### 4.3.3: Compare with Baseline

```
MAST Training Validation:

| Metric           | Baseline | Optimized | Improvement |
|------------------|----------|-----------|-------------|
| QPS              | {X}      | {Y}       | {Z}%        |
| Step time        | {X} ms   | {Y} ms    | {Z}%        |
| Memory per GPU   | {X} GB   | {Y} GB    | {Z}%        |
| Training stable  | Yes      | Yes       | -           |
```

### 4.3.4: Watch for Issues

Monitor for:
- Training instability (loss spikes, NaN)
- Convergence degradation
- Memory errors (OOM)
- Performance regression at scale

If issues detected:
```
WARNING: Issue detected in MAST training:
{description_of_issue}

This might be due to:
1. {hypothesis_1}
2. {hypothesis_2}

Recommended action:
{suggestion}
```

---

## Phase 4 Outputs

<output-format>
Update workflow state:

```yaml
numerical_equivalence: {true|false}
equivalence_details:
  forward_max_diff: {value}
  gradient_max_diff: {value}
  tolerance: "rtol=1e-5, atol=1e-5"

mast_validation:
  baseline_qps: {value}
  optimized_qps: {value}
  qps_improvement: "{X}%"
  training_stable: {true|false}
```
</output-format>

---

## Final Report

<output-format>
Generate comprehensive final report:

```
================================================================================
QPS OPTIMIZATION - FINAL REPORT
================================================================================

Target:
- Source file: {source_path}
- Entry point: {entry_point}

Optimizations Applied:
1. {optimization_1}: {description} - {improvement}%
2. {optimization_2}: {description} - {improvement}%
...

Performance Results:
| Metric          | Baseline | Optimized | Improvement |
|-----------------|----------|-----------|-------------|
| Forward pass    | {X} ms   | {Y} ms    | {Z}%        |
| Backward pass   | {X} ms   | {Y} ms    | {Z}%        |
| Total step      | {X} ms   | {Y} ms    | {Z}%        |
| MAST QPS        | {X}      | {Y}       | {Z}%        |

Numerical Equivalence: VERIFIED

Files Changed:
- {file_1}: {description_of_changes}
- {file_2}: {description_of_changes}

Benchmark Tool:
- Location: {benchmark_path}
- BUCK target: {buck_target}

================================================================================
```
</output-format>

---

## Completion Checkpoint

Log completion:

```
Phase 4 Complete: Validation

All validation steps completed:
1. Numerical equivalence verified
2. Local training efficiency confirmed
3. MAST job executed
4. QPS improvement validated

Final Results:
- QPS Improvement: {X}%
- Training Stability: Confirmed

The optimization is complete and ready for review and deployment.
```

<phase-complete>Workflow complete.</phase-complete>

---

## Error Handling

### Numerical Equivalence Failure

If numerical equivalence check fails:

1. Log the failure details:
```
Numerical equivalence check failed:

Forward output:
- Max diff: {value} (tolerance: {tolerance})
- Location of max diff: {tensor_index_or_location}

This could be due to:
1. Algorithm change with different numerical properties
2. Order of operations changed
3. Precision loss in optimization
```

<action type="auto-rollback">Automatically rollback the problematic optimization</action>
3. Re-run verification with remaining optimizations
4. Continue with workflow

Only stop if ALL optimizations fail numerical equivalence.

### MAST Training Issues

If MAST training shows issues:

1. Log the issue:
```
MAST training shows issues:

Issue: {description}
- Observed: {what_happened}
- Expected: {what_should_happen}
```

<action type="auto-rollback">Automatically rollback the most recent optimization</action>
3. Re-run MAST validation
4. Continue with remaining optimizations

### No Significant QPS Improvement

If QPS improvement is minimal, log and complete:

```
Note: MAST QPS improvement is minimal:
- Baseline QPS: {X}
- Optimized QPS: {Y}
- Improvement: {Z}%

Possible reasons:
1. Benchmark showed improvement but it's not the bottleneck in full training
2. Other components are now the bottleneck
3. Overhead from other training operations dominates

The benchmark-level optimizations have been applied. Further investigation
of the full training pipeline may be needed to identify other bottlenecks.
```

<phase-complete>Complete workflow with documented findings.</phase-complete>
# Bash Permissions Setup

Reusable module for granting Bash command permissions to enable autonomous skill execution.

## Why This Matters

Claude Code requires explicit permission to run Bash commands. Without pre-configured permissions, the skill will pause and prompt the user for each command type, breaking the autonomous workflow.

## Bootstrap Requirement

<critical-rules>
<requirement type="bootstrap">This skill requires a pre-existing permission file to allow Claude to create/modify settings.</requirement>

Before invoking this skill, the user must have one of these in place:
1. A `.claude/settings.local.json` file with `"Write(*)"` or broader permissions, OR
2. User manually creates the settings file before skill execution, OR
3. User runs Claude Code with `--dangerously-skip-permissions` flag

<warning>Without bootstrap permissions, Claude cannot self-configure.</warning>
</critical-rules>

### Pre-Requisite File (User Must Create)

Users should create this file in their project root before using this skill:

**File**: `<project_root>/.claude/settings.local.json`
```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Write(*)"
    ]
  }
}
```

Or create via terminal before starting Claude Code:
```bash
mkdir -p .claude && cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Write(*)"
    ]
  }
}
EOF
```

## Commands Used by This Skill

| Command | Purpose |
|---------|---------|
| `nvidia-smi` | Check GPU availability and select idle GPU |
| `buck run/build/test` | Run benchmark, build targets, and tests |
| `python3` | Run Python scripts for trace analysis and data processing |
| `hg/sl` commands | Version control operations |
| `mkdir`, `cat`, `ls`, `head`, `tail` | File and directory operations |
| `grep`, `awk`, `sed` | Text processing for parsing outputs |

These are typical commands, but the skill may need additional commands depending on your environment.

## Self-Unblocking Steps (Execute Before Phase 1)

**Step 1: Determine settings file path**
```
Settings file: <current_working_directory>/.claude/settings.local.json
```

**Step 2: Create the settings file using Bash**

Use Bash with mkdir and heredoc to create the file:

```bash
mkdir -p <current_working_directory>/.claude && cat > <current_working_directory>/.claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)"
    ]
  }
}
EOF
```

This grants permission for all Bash commands, enabling autonomous execution without interruptions.

If the file already exists, read it first and merge permissions into the existing `allow` array.

**Step 3: Confirm and proceed**

Log to user:
```
Permissions configured in .claude/settings.local.json
Proceeding to Phase 1...
```

## If Permission Prompts Still Appear

If the `Bash(*)` wildcard doesn't work for a specific command:

1. **Check if the settings file was created correctly**:
   ```bash
   cat <current_working_directory>/.claude/settings.local.json
   ```

2. **Add specific command patterns if needed**:
   You can use more explicit patterns like:
   ```json
   {
     "permissions": {
       "allow": [
         "Bash(nvidia-smi:*)",
         "Bash(buck:*)",
         "Bash(python3:*)",
         "Bash(hg:*)",
         "Bash(sl:*)",
         "Bash(mkdir:*)",
         "Bash(cat:*)"
       ]
     }
   }
   ```

3. **When prompted for a new command**: Select "Yes, and don't ask again for this session"

## Automatic Permission Self-Healing

<critical-rules>
<rule type="self-heal">When a Bash command is blocked by permissions, automatically add it to the settings file.</rule>

Self-heal automatically by following these steps:
</critical-rules>

**Step 1**: Read the current settings file:
```bash
cat <current_working_directory>/.claude/settings.local.json
```

**Step 2**: Parse the blocked command and determine the permission pattern needed:
- `python3 -c "..."` → `"Bash(python3:*)"`
- `nvidia-smi` → `"Bash(nvidia-smi:*)"`
- `buck run ...` → `"Bash(buck:*)"`

**Step 3**: Update the settings file to include the new permission pattern:
- Use `cat` with heredoc to rewrite the file with the new permission added to the `allow` array
- Preserve all existing permissions

**Step 4**: Retry the blocked command

**Example self-healing flow**:
```
Command blocked: python3 -c "print('hello')"

1. Read current settings.local.json
2. Add "Bash(python3:*)" to allow array
3. Write updated settings.local.json using:
   cat > .claude/settings.local.json << 'EOF'
   {
     "permissions": {
       "allow": [
         "Bash(*)",
         "Write(*)",
         "Bash(python3:*)"
       ]
     }
   }
   EOF
4. Retry: python3 -c "print('hello')"
```

## Manual Setup (Alternative)

If auto-setup fails:
1. Create/update `settings.local.json` in your working directory's `.claude` sub-directory with `"Bash(*)"` permission
2. Or when prompted during execution, select "Yes, and don't ask again for this session"
3. Or use `--dangerously-skip-permissions` flag when invoking Claude Code (use with caution)
# Benchmark Coding Template

Reusable template and guidelines for creating PyTorch benchmark scripts.

## Benchmark Script Template

Use this template as a starting point for any benchmark script:

```python
#!/usr/bin/env python3
# pyre-strict
"""
Benchmark script for <ModuleName> forward and backward passes.

Usage:
    # Standard run (profiler runs by default, generates trace.json):
    buck run @mode/opt //path/to/tests:benchmark_module

    # Skip profiler for quick debugging:
    buck run @mode/opt //path/to/tests:benchmark_module -- --skip-profile

    # Custom parameters:
    buck run @mode/opt //path/to/tests:benchmark_module -- \
        --batch-size 8 --max-seq-len 256 --num-iterations 50
"""

import argparse
import logging
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.profiler import profile, record_function

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def requires_backward_pass(
    model: nn.Module,
    inputs: Dict[str, Any],
    device: torch.device,
) -> Tuple[bool, str]:
    """
    Detect if a module requires backward pass benchmarking.

    Checks for common patterns where backward is unnecessary:
    - No trainable parameters (frozen model or no params)
    - No inputs require gradients
    - Output is detached from computation graph
    - Forward uses torch.no_grad() or inference_mode()

    Returns:
        (requires_backward, reason): Tuple with boolean and explanation
    """
    # Check 1: Any trainable parameters?
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    total_params = sum(1 for _ in model.parameters())

    if total_params == 0:
        return False, "Module has no parameters"

    if trainable_params == 0:
        return False, f"All {total_params} parameters are frozen (requires_grad=False)"

    # Check 2: Any input tensor requires gradient?
    grad_inputs = []
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            grad_inputs.append(k)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor) and sub_v.requires_grad:
                    grad_inputs.append(f"{k}.{sub_k}")

    if not grad_inputs:
        return False, "No input tensors require gradients"

    # Check 3: Try forward and check if output requires grad
    model.train()
    try:
        # Clone inputs with gradients enabled
        test_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                test_inputs[k] = v.detach().clone().requires_grad_(True)
            elif isinstance(v, dict):
                test_inputs[k] = {
                    sub_k: sub_v.detach().clone().requires_grad_(True)
                    if isinstance(sub_v, torch.Tensor) else sub_v
                    for sub_k, sub_v in v.items()
                }
            else:
                test_inputs[k] = v

        with torch.enable_grad():
            # Handle both positional and keyword argument styles
            if "input_tensor" in test_inputs:
                output = model(test_inputs["input_tensor"])
            else:
                output = model(**test_inputs)

            # Check if output requires grad
            if isinstance(output, torch.Tensor):
                if not output.requires_grad:
                    return False, "Output tensor does not require grad (likely detached or no_grad context)"
            elif isinstance(output, (tuple, list)):
                has_grad = any(
                    t.requires_grad for t in output
                    if isinstance(t, torch.Tensor)
                )
                if not has_grad:
                    return False, "No output tensors require grad (outputs are detached)"
            elif hasattr(output, "__dict__"):
                # Check dataclass or named tuple outputs
                tensor_outputs = [
                    v for v in output.__dict__.values()
                    if isinstance(v, torch.Tensor)
                ]
                if tensor_outputs and not any(t.requires_grad for t in tensor_outputs):
                    return False, "Output structure contains no tensors requiring grad"

    except Exception as e:
        # If forward fails during check, assume backward is needed
        logger.warning(f"Could not auto-detect backward requirement: {e}")
        return True, f"Auto-detection failed ({e}), assuming backward needed"

    return True, f"Module has {trainable_params} trainable params, inputs {grad_inputs} require grad, output requires grad"


def create_benchmark_inputs(
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Create inputs for benchmarking the target module.
    Returns a dict with all required input tensors.

    CUSTOMIZE THIS FUNCTION:
    - Match the exact input signature of the target module's forward()
    - Use correct tensor shapes from Phase 1 code analysis
    - Handle special formats (jagged tensors, offsets, etc.)
    """
    # Example: Create mock inputs matching the module's forward() signature
    x = torch.randn(
        (batch_size, max_seq_len, embedding_dim),
        device=device,
        dtype=dtype,
    ).requires_grad_(True)

    return {
        "input_tensor": x,
        # Add other required inputs based on code analysis
    }


def create_model(
    # Add configuration parameters as needed
) -> nn.Module:
    """
    Create and return the model/module to benchmark.

    CUSTOMIZE THIS FUNCTION:
    - Import the target module
    - Use correct constructor arguments
    - Match production configuration
    """
    # Example:
    # from path.to.module import TargetModule
    # return TargetModule(config=...)
    raise NotImplementedError("Customize this function for your target module")


def benchmark_forward_backward(
    model: nn.Module,
    inputs: Dict[str, Any],
    num_warmup: int = 5,
    num_iterations: int = 50,
    skip_backward: bool = False,
    auto_detect_backward: bool = True,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Benchmark forward and backward passes separately.
    Returns timing statistics for both.

    Args:
        model: The model to benchmark
        inputs: Dictionary of input tensors
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        skip_backward: If True, skip backward pass benchmarking entirely
        auto_detect_backward: If True, auto-detect if backward is needed
        device: Device for auto-detection (uses cuda if available)

    Returns:
        Dictionary with timing stats and backward_skipped indicator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backward_skip_reason = ""

    # Auto-detect if backward is needed
    if auto_detect_backward and not skip_backward:
        needs_backward, reason = requires_backward_pass(model, inputs, device)
        if not needs_backward:
            logger.info(f"Auto-detected: Skipping backward pass ({reason})")
            skip_backward = True
            backward_skip_reason = reason
    elif skip_backward:
        backward_skip_reason = "User specified --skip-backward"

    model.train()

    # Warmup - critical for accurate GPU timing
    logger.info(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        # Clone inputs to avoid in-place modification issues
        x = inputs["input_tensor"].detach().clone().requires_grad_(True)
        outputs = model(x)
        if not skip_backward:
            loss = outputs.sum()
            loss.backward()

    torch.cuda.synchronize()

    # Benchmark forward pass
    logger.info(f"Running {num_iterations} forward iterations...")
    forward_times = []
    for _ in range(num_iterations):
        x = inputs["input_tensor"].detach().clone().requires_grad_(True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with record_function("forward"):
            outputs = model(x)

        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

    # Compute forward statistics
    forward_times_t = torch.tensor(forward_times)

    result: Dict[str, Any] = {
        "forward_mean_ms": forward_times_t.mean().item() * 1000,
        "forward_std_ms": forward_times_t.std().item() * 1000,
        "forward_min_ms": forward_times_t.min().item() * 1000,
        "forward_max_ms": forward_times_t.max().item() * 1000,
        "backward_skipped": skip_backward,
        "backward_skip_reason": backward_skip_reason,
    }

    # Benchmark backward pass (if needed)
    if skip_backward:
        logger.info(f"Backward pass: SKIPPED ({backward_skip_reason})")
        result.update({
            "backward_mean_ms": 0.0,
            "backward_std_ms": 0.0,
            "backward_min_ms": 0.0,
            "backward_max_ms": 0.0,
        })
    else:
        logger.info(f"Running {num_iterations} backward iterations...")
        backward_times = []
        for _ in range(num_iterations):
            x = inputs["input_tensor"].detach().clone().requires_grad_(True)
            outputs = model(x)
            loss = outputs.sum()

            torch.cuda.synchronize()
            start = time.perf_counter()

            with record_function("backward"):
                loss.backward()

            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)

        backward_times_t = torch.tensor(backward_times)
        result.update({
            "backward_mean_ms": backward_times_t.mean().item() * 1000,
            "backward_std_ms": backward_times_t.std().item() * 1000,
            "backward_min_ms": backward_times_t.min().item() * 1000,
            "backward_max_ms": backward_times_t.max().item() * 1000,
        })

    return result


def run_profiler(
    model: nn.Module,
    inputs: Dict[str, Any],
    output_dir: str = "./profiler_output",
    skip_backward: bool = False,
) -> str:
    """
    Run torch.profiler to get detailed performance breakdown.
    Exports trace JSON and prints summary tables.

    MANDATORY: This function must always be called to generate trace.json
    for use in the optimization phase. The trace file enables:
    - Identifying top time-consuming operations
    - Mapping expensive ops to source code lines
    - Understanding the call hierarchy
    - Guiding optimization decisions

    Args:
        output_dir: Directory for trace.json. Defaults to "./profiler_output".
        skip_backward: If True, skip backward pass in profiling.

    Returns:
        Path to the exported trace JSON file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model.train()

    profile_type = "forward only" if skip_backward else "forward + backward"
    logger.info(f"Running profiler (3 iterations, {profile_type})...")

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(3):
            x = inputs["input_tensor"].detach().clone().requires_grad_(True)

            with record_function("forward_pass"):
                outputs = model(x)

            if not skip_backward:
                with record_function("backward_pass"):
                    loss = outputs.sum()
                    loss.backward()

            torch.cuda.synchronize()

    # Print summary tables
    print("\n" + "=" * 80)
    print("PROFILER SUMMARY - CUDA Time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("PROFILER SUMMARY - CPU Time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Export trace JSON for analysis - MANDATORY for optimization phase
    trace_path = f"{output_dir}/trace.json"
    prof.export_chrome_trace(trace_path)
    logger.info(f"Trace JSON exported to: {trace_path}")
    print(f"\n*** TRACE FILE: {trace_path} ***")
    print("Use this trace file in Phase 3 (optimization) for profiler-guided changes.")

    # Print memory summary
    print("\n" + "=" * 80)
    print("MEMORY SUMMARY")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark <ModuleName>")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profiler (NOT recommended - trace.json is needed for optimization)")
    parser.add_argument("--skip-backward", action="store_true", help="Skip backward pass benchmarking (use for inference-only modules)")
    parser.add_argument("--no-auto-detect", action="store_true", help="Disable auto-detection of backward requirement")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="./profiler_output", help="Profiler output directory")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print CUDA diagnostics
    if args.device == "cuda":
        logger.info("CUDA diagnostics:")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  Device name: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"  Compute capability: {props.major}.{props.minor}")

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Log configuration
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max sequence length: {args.max_seq_len}")

    # Create model
    logger.info("Creating model...")
    model = create_model()  # Customize create_model() for your target
    model = model.to(device=device, dtype=dtype)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create inputs
    logger.info("Creating inputs...")
    inputs = create_benchmark_inputs(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim,
        dtype=dtype,
        device=device,
    )

    # Run benchmark
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING BENCHMARK")
    logger.info("=" * 80)

    timing_stats = benchmark_forward_backward(
        model=model,
        inputs=inputs,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        skip_backward=args.skip_backward,
        auto_detect_backward=not args.no_auto_detect,
        device=device,
    )

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Forward pass:  {timing_stats['forward_mean_ms']:.3f} ± {timing_stats['forward_std_ms']:.3f} ms")
    print(f"  min: {timing_stats['forward_min_ms']:.3f} ms, max: {timing_stats['forward_max_ms']:.3f} ms")

    if timing_stats.get("backward_skipped", False):
        print(f"Backward pass: SKIPPED ({timing_stats.get('backward_skip_reason', 'Not required')})")
        print(f"Total:         {timing_stats['forward_mean_ms']:.3f} ms (forward only)")
    else:
        print(f"Backward pass: {timing_stats['backward_mean_ms']:.3f} ± {timing_stats['backward_std_ms']:.3f} ms")
        print(f"  min: {timing_stats['backward_min_ms']:.3f} ms, max: {timing_stats['backward_max_ms']:.3f} ms")
        print(f"Total:         {timing_stats['forward_mean_ms'] + timing_stats['backward_mean_ms']:.3f} ms")
    print("=" * 80)

    # Run profiler to generate trace.json (MANDATORY for optimization phase)
    # Use --skip-profile only for quick iteration during debugging
    if not args.skip_profile:
        skip_backward_for_profile = timing_stats.get("backward_skipped", args.skip_backward)
        trace_path = run_profiler(
            model=model,
            inputs=inputs,
            output_dir=args.output_dir,
            skip_backward=skip_backward_for_profile,
        )
        print(f"\n*** Trace file for optimization: {trace_path} ***")
    else:
        logger.warning("Profiler skipped. trace.json will NOT be generated.")
        logger.warning("Re-run without --skip-profile before starting optimization phase.")


if __name__ == "__main__":
    main()
```

---

## BUCK Configuration

### Standard Python Binary Target

```python
python_binary(
    name = "benchmark_module",
    srcs = ["benchmark_module.py"],
    keep_gpu_sections = True,  # REQUIRED for GPU benchmarks
    main_function = "path.to.tests.benchmark_module.main",
    deps = [
        "//caffe2:torch",
        # Add dependencies to the target module
        "//path/to:module",
    ],
)
```

### Important BUCK Notes

| Setting | Requirement | Why |
|---------|-------------|-----|
| `keep_gpu_sections = True` | **Mandatory** | Preserves CUDA kernels; without it GPU sections may be stripped |
| `@mode/opt` | **Required for benchmarking** | Debug mode adds overhead and disables optimizations |
| `gpu_python_binary` | Consider for complex GPU deps | Alternative to `python_binary` for heavy GPU usage |

---

## Key Best Practices

### 1. Input Size Limits for Fast Benchmarking

<rule>Use small default input sizes to enable fast benchmark iteration. Production-scale inputs make benchmarking slow and impractical.</rule>

| Parameter | Recommended Default | Maximum for Quick Benchmarks | Notes |
|-----------|-------------------|------------------------------|-------|
| `batch_size` | 4 | 8 | Larger batches increase memory and compute time proportionally |
| `max_seq_len` | 256 | 512 | Attention is O(n²); doubling length quadruples time |
| `num_targets` | 10 | 50 | Keep small relative to sequence length |
| `embedding_dim` | 256 | 256 | Match production value for benchmarks |
| `num_layers` | 4 | 6 | Each layer adds linear overhead |

<note>Why This Matters:</note>
- ✅ Good: `batch_size=4, max_seq_len=256, num_targets=10` → ~5-20ms per iteration
- ❌ Bad: `batch_size=32, max_seq_len=3072, num_targets=1024` → 700+ ms per iteration

<note>Scaling Guidelines:</note>
- For attention-based models, compute scales O(batch × seq_len²)
- Keep total benchmark iteration time under 50ms for rapid experimentation
- Use `--batch-size`, `--max-seq-len` CLI flags to test at production scale when needed

<example type="good">Example Default Args (Good):</example>
```python
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
parser.add_argument("--num-targets", type=int, default=10, help="Number of targets")
```

<example type="bad">Example Default Args (Bad - Avoid):</example>
```python
# Use small defaults for fast iteration
parser.add_argument("--batch-size", default=32, ...)  # Too large
parser.add_argument("--num-targets", default=1024, ...)  # Too large
parser.add_argument("--base-seq-len", default=2048, ...)  # Too large
```

### 2. Warmup Iterations
GPU kernels need warmup for accurate timing. Always run 5+ warmup iterations before measuring.

### 3. Clone Inputs Each Iteration
Use `detach().clone().requires_grad_(True)` to break computation graphs between iterations:
```python
x = inputs["input_tensor"].detach().clone().requires_grad_(True)
```

### 4. CUDA Synchronization
Required before/after timing for accurate GPU measurements:
```python
torch.cuda.synchronize()
start = time.perf_counter()
# ... operation ...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

### 5. Record Functions
Use `record_function()` to label operations for profiler visibility:
```python
with record_function("forward"):
    outputs = model(x)
```

### 6. Compute Statistics
Report mean, std, min, max for reliable comparisons - single measurements are not reliable.

### 7. Log CUDA Diagnostics
Device name and compute capability help with reproducibility.

### 8. Export Trace JSON
The trace file enables detailed analysis in chrome://tracing or other tools.

---

## Debugging Guidelines

<rule>
<rule>Never modify the source code being tested.</rule>
<rule>Only modify the benchmark tool code.</rule>
<rule>Focus on mocking input logic.</rule>
</rule>

### Common Issues and Fixes

| Issue | Fix Location | Solution |
|-------|--------------|----------|
| Shape mismatch | `create_benchmark_inputs()` | Adjust tensor dimensions |
| Wrong dtype | `create_benchmark_inputs()` | Change `dtype` parameter |
| Missing inputs | `create_benchmark_inputs()` | Add required fields to return dict |
| Device mismatch | `create_benchmark_inputs()` | Ensure all tensors on same device |
| Constructor error | `create_model()` | Fix constructor arguments |
| Import error | BUCK deps | Add missing dependencies |
| Jagged tensor | `create_benchmark_inputs()` | Create proper jagged/nested format |

### Debugging Workflow

```
Error occurs
    │
    ▼
Identify error type
    │
    ├─► Shape error ──► Fix create_benchmark_inputs()
    ├─► Type error ───► Fix dtype in create_benchmark_inputs()
    ├─► Import error ─► Fix BUCK dependencies
    ├─► Init error ───► Fix create_model()
    │
    ▼
Retry (max 2-3 attempts per error type)
    │
    ▼
If still failing ──► Ask user for help
```

### Max Attempts Before Asking User

| Error Type | Max Self-Fix Attempts |
|------------|----------------------|
| Import errors | 2 |
| Shape mismatches | 2 |
| BUCK build failures | 2 |
| CUDA errors | 1 |
| Model initialization | 1 |

---

## File Naming and Location

| Item | Convention | Example |
|------|------------|---------|
| Directory | `tests/` under source | `/path/to/module/tests/` |
| Filename | `benchmark_{module}.py` | `benchmark_attention.py` |
| BUCK target | Same as filename | `benchmark_attention` |

---

## Run Commands

```bash
# Basic run (always use @mode/opt) - includes profiler by default
# Generates trace.json for use in optimization phase
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module

# Skip profiler for quick debugging iterations (NOT recommended for final baseline)
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- --skip-profile

# Custom parameters
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- \
    --batch-size 8 --max-seq-len 256 --num-iterations 50

# Custom output directory for trace.json
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- \
    --output-dir /path/to/traces
```

<note>The profiler runs by default to generate `trace.json`. This trace file is required for the optimization phase (Phase 3). Only use `--skip-profile` during debugging when you need faster iteration.</note>

---

## Profiling Best Practices

### Profiler Iteration Limits

<warning>The PyTorch profiler can cause CPU max usage and process hang if misused. Follow these rules strictly.</warning>

| Rule | Requirement | Why |
|------|-------------|-----|
| Max profiled iterations | 2-3 iterations only | Profiler stores detailed timing, stack traces, and shape info for every operation. More iterations = exponential memory growth |
| Separate from benchmark | Run profiler in dedicated function | Never wrap full benchmark loop (warmup + benchmark iters) in profiler context |
| Clone inputs each iteration | `detach().clone().requires_grad_(True)` | Prevents computation graph accumulation across iterations |

### What Goes Wrong Without These Rules

<example type="bad">Bad Pattern (causes CPU hang):</example>
```python
# Avoid this pattern - wraps 30 iterations in profiler
with torch_profile(...) as prof:
    run_benchmark(warmup=5, benchmark=25)  # 30 iterations profiled!
```

<warning>Problems:</warning>
1. Memory explosion: Profiler data accumulates for 30 full forward+backward passes
2. CPU-intensive post-processing: `prof.key_averages()` must aggregate thousands of recorded operations
3. Graph accumulation: Without input cloning, computation graphs grow across iterations
4. Result: Process appears stuck at 100% CPU, may take 10+ minutes or OOM

### Correct Profiler Implementation

<example type="good">Good Pattern:</example>
```python
def run_profiler(model, inputs, num_iters=3):
    """Run profiler with only 2-3 iterations."""
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(num_iters):  # Only 2-3 iterations!
            # Clone inputs to break computation graph
            x = inputs["input"].detach().clone().requires_grad_(True)
            outputs = model(x)
            loss = outputs.sum()
            loss.backward()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    # Run benchmark first WITHOUT profiling
    timing_stats = benchmark_forward_backward(model, inputs, ...)

    # Print results
    print(f"Forward: {timing_stats['forward_mean_ms']:.3f} ms")

    # Run profiler by default to generate trace.json (skip with --skip-profile)
    if not args.skip_profile:
        run_profiler(model, inputs, num_iters=3)
```

### Input Cloning Helper

For complex input dictionaries, use a helper function:
```python
def clone_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Clone inputs with fresh tensors to break computation graph."""
    cloned = dict(inputs)

    # Clone main input tensor
    cloned["input_tensor"] = inputs["input_tensor"].detach().clone().requires_grad_(True)

    # Clone nested payloads if present
    if "payloads" in cloned:
        cloned["payloads"] = dict(inputs["payloads"])
        for key in ["embeddings", "nro_embeddings"]:
            if key in cloned["payloads"]:
                cloned["payloads"][key] = (
                    inputs["payloads"][key].detach().clone().requires_grad_(True)
                )

    return cloned
```

### Profiler Checklist

Before implementing profiler support in a benchmark:

- [ ] Profiler runs in separate function from benchmark loop
- [ ] Profiler limited to 2-3 iterations max
- [ ] Inputs are cloned each profiled iteration with `detach().clone()`
- [ ] `torch.cuda.synchronize()` called after each profiled iteration
- [ ] Benchmark runs first, profiler runs after (not instead of) benchmark
# GPU Availability Check

Reusable module for verifying GPU availability before benchmarking.

## Why This Matters

<note>If the GPU is busy or unavailable, benchmark results will have extremely high variance and be unreliable. This check must be performed before any benchmarking run.</note>

## Required Bash Permissions

The `nvidia-smi` command requires Bash permission. See `shared/bash_permissions.md` for permission setup instructions.

## Automated GPU Selection

<note>This process is fully automated. No user interaction required unless no idle GPU is available.</note>

### Step 1: Check GPU Status

Run this command to check GPU availability:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits
```

### Step 2: Automatically Select an Idle GPU

<selection-criteria>
Parse the output and select the first GPU with:
- GPU utilization < 5% - Indicates the GPU is idle
- Sufficient free memory - At least 10GB free
</selection-criteria>

Example output:
```
0, 0, 79500
1, 95, 10000
2, 2, 79000
```

<action>Automatically select GPU 0 (first idle GPU with < 5% utilization).</action>

Record the selected GPU:
```yaml
gpu_device: "0"
```

Log to user (informational only, no confirmation needed):
```
GPU Check: Selected GPU 0 (0% utilization, 79500 MiB free)
```

### Step 3: If No Idle GPU - Critical Stop

<condition>Only stop if ALL GPUs have > 5% utilization.</condition>

<critical-stop ref="STOP_NO_GPU">
No idle GPU available for benchmarking

Current GPU status:
[nvidia-smi output]

All GPUs are currently in use (> 5% utilization). Cannot proceed with reliable benchmarks.

Please either:
1. Wait for other jobs to complete
2. Free up GPU resources on one of the devices

Reply when ready to continue.
</critical-stop>

<note>This is one of the 3 critical stop conditions for this skill.</note>

## Consistency Across Runs

<requirement type="consistency">When comparing benchmark results (e.g., baseline vs optimized), the GPU environment should be identical:</requirement>

| Requirement | Why |
|-------------|-----|
| **Same GPU device** | Different GPUs may have different performance characteristics |
| **Same utilization state** | All runs should start with idle GPU (< 5%) |
| **Same memory state** | Similar free memory available before each run |

### Before Each Benchmark Run

Automatically verify GPU is still idle:
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits -i [GPU_INDEX]
```

If GPU utilization has increased > 5% since selection, wait briefly or select a different idle GPU automatically.

## Quick Reference

```bash
# GPU metrics for automatic selection
nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits

# Check specific GPU (e.g., GPU 2)
nvidia-smi -i 2

# List processes using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
# Numerical Equivalence Check

Reusable module for verifying numerical equivalence between baseline and optimized outputs.

## Why This Matters

Optimizations must preserve numerical correctness. Even small differences can compound during training and cause divergence. <usage-timing>
This check must be performed:
1. After each optimization iteration (Phase 3) - early detection of issues
2. During final validation (Phase 4) - comprehensive verification
</usage-timing>

## Quick Check (Use During Optimization Iterations)

<note>This is a lightweight check for rapid iteration. Run after each optimization change.</note>

### Implementation

Add this to your benchmark script or run as a separate verification:

```python
import torch
from typing import Dict, Any, Tuple

def quick_numerical_check(
    model_baseline: torch.nn.Module,
    model_optimized: torch.nn.Module,
    inputs: Dict[str, Any],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick numerical equivalence check for optimization iterations.

    Returns:
        Tuple of (passed: bool, details: dict)
    """
    # Clone inputs for both runs
    x_baseline = inputs["input_tensor"].detach().clone().requires_grad_(True)
    x_optimized = inputs["input_tensor"].detach().clone().requires_grad_(True)

    # Forward pass
    with torch.no_grad():
        out_baseline = model_baseline(x_baseline)
        out_optimized = model_optimized(x_optimized)

    # Check forward equivalence
    try:
        torch.testing.assert_close(
            out_optimized, out_baseline,
            rtol=rtol, atol=atol,
            msg="Forward output mismatch"
        )
        forward_passed = True
        forward_max_diff = (out_optimized - out_baseline).abs().max().item()
    except AssertionError as e:
        forward_passed = False
        forward_max_diff = (out_optimized - out_baseline).abs().max().item()

    # Backward pass (with gradients enabled)
    x_baseline = inputs["input_tensor"].detach().clone().requires_grad_(True)
    x_optimized = inputs["input_tensor"].detach().clone().requires_grad_(True)

    out_baseline = model_baseline(x_baseline)
    out_optimized = model_optimized(x_optimized)

    out_baseline.sum().backward()
    out_optimized.sum().backward()

    # Check gradient equivalence
    try:
        torch.testing.assert_close(
            x_optimized.grad, x_baseline.grad,
            rtol=rtol, atol=atol,
            msg="Gradient mismatch"
        )
        backward_passed = True
        grad_max_diff = (x_optimized.grad - x_baseline.grad).abs().max().item()
    except AssertionError as e:
        backward_passed = False
        grad_max_diff = (x_optimized.grad - x_baseline.grad).abs().max().item()

    passed = forward_passed and backward_passed

    details = {
        "forward_passed": forward_passed,
        "forward_max_diff": forward_max_diff,
        "backward_passed": backward_passed,
        "grad_max_diff": grad_max_diff,
        "rtol": rtol,
        "atol": atol,
    }

    return passed, details
```

### Usage in Optimization Loop

```python
# After applying an optimization
passed, details = quick_numerical_check(
    model_baseline=baseline_model,
    model_optimized=optimized_model,
    inputs=benchmark_inputs,
)

if not passed:
    print(f"NUMERICAL MISMATCH DETECTED:")
    print(f"  Forward: {'PASS' if details['forward_passed'] else 'FAIL'} (max diff: {details['forward_max_diff']:.2e})")
    print(f"  Backward: {'PASS' if details['backward_passed'] else 'FAIL'} (max diff: {details['grad_max_diff']:.2e})")
    # Rollback this optimization
else:
    print(f"Numerical equivalence: PASS (forward diff: {details['forward_max_diff']:.2e}, grad diff: {details['grad_max_diff']:.2e})")
```

## Tolerance Guidelines

| Scenario | rtol | atol | Notes |
|----------|------|------|-------|
| **Quick check (Phase 3)** | 1e-4 | 1e-4 | Balance between catching issues and allowing minor variations |
| **Deterministic ops** | 0 | 0 | Use `torch.equal()` instead |
| **Mixed precision (fp16/bf16)** | 1e-2 | 1e-2 | Lower precision requires looser tolerance |
| **Final validation (Phase 4)** | 1e-5 | 1e-5 | Stricter for final verification |
| **Stochastic ops** | N/A | N/A | Use statistical comparison instead |

## When Same Model Has Been Modified

If you're modifying the source code directly (not comparing two separate model instances), use this pattern:

```python
def check_before_after_optimization(
    create_model_fn,
    inputs: Dict[str, Any],
    apply_optimization_fn,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check numerical equivalence before and after applying an optimization.

    Args:
        create_model_fn: Function that creates a fresh model instance
        inputs: Benchmark inputs
        apply_optimization_fn: Function that modifies the model in-place
    """
    # Get baseline output before optimization
    model = create_model_fn()
    x = inputs["input_tensor"].detach().clone().requires_grad_(True)
    baseline_out = model(x)
    baseline_out.sum().backward()
    baseline_grad = x.grad.clone()
    baseline_out = baseline_out.detach().clone()

    # Apply optimization and get new output
    apply_optimization_fn()  # This modifies source code

    model_opt = create_model_fn()  # Reload with optimized code
    x_opt = inputs["input_tensor"].detach().clone().requires_grad_(True)
    opt_out = model_opt(x_opt)
    opt_out.sum().backward()

    # Compare
    try:
        torch.testing.assert_close(opt_out, baseline_out, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_opt.grad, baseline_grad, rtol=rtol, atol=atol)
        return True, {"forward_max_diff": (opt_out - baseline_out).abs().max().item(),
                      "grad_max_diff": (x_opt.grad - baseline_grad).abs().max().item()}
    except AssertionError:
        return False, {"forward_max_diff": (opt_out - baseline_out).abs().max().item(),
                       "grad_max_diff": (x_opt.grad - baseline_grad).abs().max().item()}
```

## Reporting Format

Use this format when reporting numerical check results:

```
Numerical Check: [PASS/FAIL]
  Forward output:  max_diff = [X.XXe-XX] (tolerance: rtol=[X], atol=[X])
  Gradient:        max_diff = [X.XXe-XX] (tolerance: rtol=[X], atol=[X])
  Status: [PASS/FAIL - reason if failed]
```

## Edge Cases to Test (Phase 4 Only)

For comprehensive validation in Phase 4, test with:

| Edge Case | How to Test |
|-----------|-------------|
| Zero inputs | `torch.zeros(...)` |
| Very small values | `torch.randn(...) * 1e-6` |
| Very large values | `torch.randn(...) * 1e6` |
| Different batch sizes | Test with batch=1, 4, 16 |
| Different sequence lengths | Test with seq_len=1, 128, 512 |
| Different dtypes | Test fp32, fp16, bf16 |

## Integration with Phase 3

In Phase 3 optimization iterations, use the quick check after each optimization:

```
For each optimization iteration:
1. Apply optimization
2. Run benchmark (get timing)
3. Run quick_numerical_check()
4. If FAIL → rollback optimization, log failure, try next approach
5. If PASS → commit optimization, record improvement
```

This catches numerical issues immediately, preventing wasted time on optimizations that break correctness.
# Pre-Code Reasoning Block

Reusable module for structured reasoning before writing any code.

## Why This Matters

Outputting a reasoning block before code changes ensures:
- Constraints are explicitly acknowledged and followed
- The approach is documented before implementation
- File changes are planned and traceable
- Mistakes are caught before code is written

## When to Use

Output a `<reasoning>` block before:
- Writing new code files (benchmarks, tests, utilities)
- Modifying existing source code (optimizations, fixes)
- Any file change that affects behavior

## Reasoning Block Format

```
<reasoning>
Constraints:
- [Constraint 1]: [How this change respects it]
- [Constraint 2]: [How this change respects it]

Adherence:
- [Explain how the code follows each constraint]
- [Note any trade-offs and why they're acceptable]

Files to modify:
- [file_path_1] (lines X-Y: description of change)
- [file_path_2] (new file: description of purpose)
</reasoning>
```

## Examples

### Example 1: Optimization Change

```
<reasoning>
Constraints:
- Memory-neutral: this optimization uses tensor views instead of copies, so memory stays constant
- Priority: this is a PyTorch glue code fix (priority 1), addressing redundant .contiguous() calls

Adherence:
- Using in-place slicing instead of index_select avoids new allocations
- Numerical equivalence preserved since slicing produces identical values

Files to modify:
- fbcode/path/to/module.py (lines 45-52: replace index_select with slice notation)
</reasoning>
```

### Example 2: Benchmark Creation

```
<reasoning>
Constraints:
- Standalone code: benchmark imports target module but does not modify it
- Template adherence: using benchmark_template.md structure with customized inputs

Adherence:
- create_benchmark_inputs() matches target module's forward() signature
- create_model() correctly instantiates target with production config
- Profiler generates trace.json for Phase 3 optimization

Files to modify:
- fbcode/path/to/tests/benchmark_module.py (new file: benchmark script)
- fbcode/path/to/tests/TARGETS (add python_binary target)
</reasoning>
```

### Example 3: Multi-File Optimization

```
<reasoning>
Constraints:
- Memory-neutral: replacing CPU-bound loop with vectorized operation
- Numerical equivalence: output values identical within rtol=1e-5

Adherence:
- torch.where() replaces Python loop, no new allocations
- Verified with quick_numerical_check() before committing

Files to modify:
- fbcode/path/to/attention.py (lines 120-135: vectorize masking loop)
- fbcode/path/to/utils.py (lines 45-50: add helper function)
</reasoning>
```

## Integration with Phases

### Phase 2: Benchmark Creation
Before writing benchmark files, output reasoning with:
1. Constraints: benchmark is standalone, never modify source being tested
2. Adherence: how benchmark follows template and matches target interface
3. Files: benchmark file path and BUCK file path

### Phase 3: Optimization
Before each code change, output reasoning with:
1. Constraints: memory-neutral, optimization priority level
2. Adherence: how change follows constraints, numerical equivalence plan
3. Files: specific files and line ranges to modify

## Reference in Code

When a phase file says `<requirement type="reasoning">`, refer to this document for the full format and examples.
# QPS Optimization Skill Set

A modular, interactive workflow for optimizing MAST training performance bottlenecks while ensuring numerical equivalence.

---

<critical-rules id="MEMORY_NEUTRAL">
## Core Constraints

<goal>Maximize forward/backward pass speed for multi-GPU cluster training.</goal>

<constraint type="memory">All optimizations should be memory-neutral. If an optimization increases memory usage, ask user before applying (they decide the trade-off).</constraint>

<priority-order>Optimization Priority (follow this order):</priority-order>
1. PyTorch glue code / CPU-bound fixes (low-hanging fruit)
2. Forward pass speedup
3. Backward pass speedup
4. Communication-computation overlap
5. Kernel micro-optimization (last resort)

Reference this constraint as `MEMORY_NEUTRAL` throughout phases.
</critical-rules>

<critical-rules id="PHASE_READS">
## Phase File Reading

<action>Before starting any phase, use the Read tool to load the phase instructions:</action>
<phase-files>
- Phase 1: `Read("phases/01_setup.md")`
- Phase 2: `Read("phases/02_benchmark.md")`
- Phase 3: `Read("phases/03_optimize.md")` - complete pre-flight checklist before optimization
- Phase 4: `Read("phases/04_validate.md")`
</phase-files>
</critical-rules>

<critical-rules id="REASONING_BEFORE_CODE">
## Pre-Code Reasoning

Before writing any code, output a `<reasoning>` block. See `shared/reasoning_block.md` for full format and examples.

The reasoning block must include:
1. Constraints being followed (memory-neutral, optimization priority, etc.)
2. How your code adheres to them
3. Files you intend to modify
</critical-rules>

---

## Overview

This skill set is organized into **4 phases** with clear handoffs and user checkpoints. Each phase has dedicated instructions and can be resumed independently.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PHASE 1        │     │  PHASE 2        │     │  PHASE 3        │     │  PHASE 4        │
│  Setup          │────▶│  Benchmark      │────▶│  Optimize       │────▶│  Validate       │
│                 │     │                 │     │                 │     │                 │
│  01_setup.md    │     │  02_benchmark.md│     │  03_optimize.md │     │  04_validate.md │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
   USER INPUT              USER INPUT              ITERATIVE              USER INPUT
   (target info)           (if blocked)            (checkpoint           (MAST CLI)
                                                    each iter)
```

## When to Use

Use this skill set when:
- You have identified a module/operation to optimize from MAST training
- You need to optimize training efficiency for a specific code path
- You want to profile and benchmark specific code paths in training

## Required Bash Permissions

<critical-rules id="BOOTSTRAP_PERMISSIONS">
<requirement type="bootstrap">This skill requires pre-existing permissions to run autonomously.</requirement>

Before invoking this skill, ensure one of these is in place:
1. A `.claude/settings.local.json` file with `"Bash(*)"` and `"Write(*)"` permissions exists, OR
2. User runs Claude Code with `--dangerously-skip-permissions` flag

<action>Create the permission file manually before starting (run in terminal):</action>
```bash
mkdir -p .claude && cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Write(*)"
    ]
  }
}
EOF
```

See `shared/bash_permissions.md` for details.
</critical-rules>

## Phase Summary

| Phase | File | Purpose | User Interaction |
|-------|------|---------|------------------|
| 1. Setup | `phases/01_setup.md` | Gather target info, understand code | Required at start |
| 2. Benchmark | `phases/02_benchmark.md` | Create benchmark tool, establish baseline | Only if critical error |
| 3. Optimize | `phases/03_optimize.md` | PyTorch glue code review + iterative forward/backward optimization (targeting multi-GPU cluster training) | Only if critical error |
| 4. Validate | `phases/04_validate.md` | Numerical equivalence + MAST validation | MAST CLI required |

## Shared Components

| Component | File | Purpose |
|-----------|------|---------|
| Bash Permissions | `shared/bash_permissions.md` | Permission setup for autonomous Bash execution (used before Phase 1) |
| GPU Check | `shared/gpu_check.md` | GPU availability verification (used by Phase 2, 3) |
| Benchmark Template | `shared/benchmark_template.md` | Reusable benchmark script template, BUCK config, best practices (used by Phase 2) |
| Numerical Check | `shared/numerical_check.md` | Numerical equivalence verification (used by Phase 3, 4) |
| Reasoning Block | `shared/reasoning_block.md` | Pre-code reasoning format and examples (used by Phase 2, 3) |

## State Persistence

<requirement type="persistence">State must be persisted to enable session resumption and track progress across interruptions.</requirement>

### State File Location

Create a state file in the **same directory as the source file being optimized**:

```
<source_directory>/.qps_optimization_state.yaml
```

Example: If optimizing `/path/to/module.py`, create `/path/to/.qps_optimization_state.yaml`

### When to Update State

- **After Phase 1**: Write initial state with source_path, entry_point, code_understanding
- **After Phase 2**: Add benchmark_path, buck_target, baseline_metrics
- **After each optimization iteration** (Phase 3): Update optimization_history, current_best
- **After Phase 4**: Add final validation results

### State File Format

Maintain this state throughout the optimization workflow:

```yaml
# Metadata
state_version: "1.0"
created_at: ""              # ISO timestamp when optimization started
updated_at: ""              # ISO timestamp of last update
current_phase: ""           # "setup", "benchmark", "optimize", "validate", "complete"

# Phase 1 outputs
source_path: ""           # Path to source file being optimized
entry_point: ""           # Function/class entry point
code_understanding: ""    # Summary of code structure

# Phase 2 outputs
benchmark_path: ""        # Path to created benchmark script
buck_target: ""           # BUCK target for benchmark (e.g., //path/to/tests:benchmark_module)
buck_mode: ""             # Buck mode for benchmarking (e.g., @mode/opt) - use same mode in Phase 3
buck_run_command: ""      # Full buck run command (e.g., buck run @mode/opt //path/to/tests:benchmark_module)
benchmark_commit: ""      # Commit hash for benchmark
trace_json_path: ""       # Path to trace.json generated by profiler
baseline_metrics:
  forward_ms: 0.0
  forward_std_ms: 0.0
  backward_ms: 0.0
  backward_std_ms: 0.0
  memory_mb: 0.0
gpu_device: ""            # GPU index used for benchmarking
reproducibility:
  forward_cv_percent: 0.0
  backward_cv_percent: 0.0
  status: ""              # "PASS", "WARNING", or "FAIL"

# Phase 3 outputs
optimization_history: []  # List of {iteration, optimization, status, before_ms, after_ms, speedup, equivalent}
current_best:
  forward_ms: 0.0
  backward_ms: 0.0
  total_ms: 0.0
  memory_mb: 0.0
  speedup_vs_baseline: "1.0x"
consecutive_low_improvement:  # For diminishing returns tracking
  forward: 0
  backward: 0
  total: 0

# Phase 4 outputs
numerical_equivalence: false
equivalence_details:
  forward_max_diff: 0.0
  gradient_max_diff: 0.0
  tolerance: ""           # e.g., "rtol=1e-5, atol=1e-5"
mast_validation:
  baseline_qps: 0.0
  optimized_qps: 0.0
  qps_improvement: ""
  training_stable: false
```

### Reading State on Resume

When user says "continue optimization" or "resume":

1. Search for `.qps_optimization_state.yaml` in current directory and subdirectories
2. If found, read the state file
3. Determine current phase from `current_phase` field
4. Resume from that phase with preserved context

```
Found existing optimization state:
- Source: [source_path]
- Current phase: [current_phase]
- Progress: [summary based on phase]

Resuming from Phase [N]...
```

### Writing State

Use YAML format for human readability. Update the `updated_at` timestamp on every write.

## How to Use

### Starting Fresh
```
User: "Optimize the attention module for MAST training"
→ Begin with Phase 1 (01_setup.md)
```

### Resuming After Interruption
```
User: "Continue optimization"
→ Check current state, resume from last phase

User: "Skip to Phase 3"
→ Verify Phase 1-2 outputs exist, jump to Phase 3
```

### Checking Status
```
User: "Show optimization status"
→ Display current state and progress
```

## Phase Execution Rules

### Before Each Phase
1. Read the phase-specific instructions from `phases/0N_*.md`
2. Verify prerequisites from previous phases are met
3. Inform user which phase is starting

### After Each Phase
1. Update state with phase outputs
2. Summarize results to user
<action>Proceed to next phase automatically (no user confirmation needed)</action>

### Phase Transitions
<phase-requirements>
- Phase 1 → 2: Requires source_path and entry_point
- Phase 2 → 3: Requires working benchmark with baseline metrics
- Phase 3 → 4: Requires optimization with improvement
- Phase 4 → Done: Requires MAST validation complete
</phase-requirements>

## Error Handling

<critical-stops>
### Critical Stops (Reference by ID in phase files)

Only stop and ask for user input in these critical situations:

| Stop ID | Phase | Condition | Action |
|---------|-------|-----------|--------|
| `STOP_NO_SOURCE` | 1 | Cannot access source file (path does not exist) | Ask user for correct path |
| `STOP_NO_GPU` | 2 | No idle GPU available (all GPUs > 5% utilization) | Ask user to free GPU or wait |
| `STOP_BENCHMARK_FAILED` | 2 | Benchmark failed after 10 total attempts | Ask user for debugging help |
| `STOP_MAST_CLI` | 4 | MAST CLI command required for validation | Ask user to provide command |

<fallback-behavior>All other issues: Handle automatically (rollback, try alternatives, log and proceed).</fallback-behavior>
</critical-stops>

## Progress Tracking

Track completion across all phases:

### Phase 1: Setup
1. User provided source file path
2. User provided entry point
3. Source code read and understood
4. Dependencies documented

### Phase 2: Benchmark
1. GPU availability verified
2. Benchmark script created
3. BUCK target added
4. Benchmark runs successfully
5. Baseline metrics captured

### Phase 3: Optimize
1. PyTorch glue code reviewed for low-hanging fruit
2. Multi-GPU cluster impact considered (per `MEMORY_NEUTRAL`)
3. Trace JSON generated and analyzed
4. Forward path optimization complete
5. Backward path optimization complete
6. Performance improvement confirmed

### Phase 4: Validate
1. Numerical equivalence confirmed
2. User provided MAST CLI
3. MAST job executed
4. QPS improvement validated
