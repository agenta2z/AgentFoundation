# Phase 2: Create Benchmark Tool

Create a standalone benchmark script with profiling capabilities and establish baseline metrics.

## Prerequisites

From Phase 1:
- `source_path`: Path to source file
- `entry_point`: Function/class to benchmark
- `code_understanding`: Summary of code structure

<critical-rules>
**Rule 1**: Create benchmark as fresh, standalone code. Never modify the source being tested.
**Rule 2**: When debugging, ONLY modify benchmark tool code, never the source module.
</critical-rules>

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

**File Location**:
- Create in `tests/` subdirectory under source file's directory
- Naming: `benchmark_{module_name}.py`
- Example: If source is `/path/to/module.py`, create `/path/to/tests/benchmark_module.py`

**Use the benchmark template from `shared/benchmark_template.md`**:
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

**Use the BUCK configuration from `shared/benchmark_template.md`**:

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

**Run command**:
```bash
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target}
```

**For debugging guidelines, refer to `shared/benchmark_template.md`**:
- Focus on mocking input logic in `create_benchmark_inputs()`
- Fix model initialization in `create_model()`
- Update BUCK dependencies as needed

**Max debugging attempts**: 10 total across all error types. After 10 failures → trigger `STOP_BENCHMARK_FAILED`.

### Step 2.4.1: Benchmark Timeout Handling

<critical-rules>
Benchmarks can hang due to profiler issues, CUDA errors, or infinite loops.
</critical-rules>

**Timeout Policy:**
- If benchmark produces no output within **120 seconds**, consider it hung
- Kill the process and count as 1 failed attempt toward the 10-attempt limit

**Common Hang Causes and Fixes:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Hangs during profiler | Too many profiler iterations | Reduce to 2-3 iterations |
| Hangs at CUDA init | GPU memory exhausted | Reduce batch size |
| Hangs at import | Circular import or slow module load | Check BUCK deps |
| 100% CPU after benchmark | Profiler post-processing | Reduce profiler iterations |

**When timeout occurs:**
```
Benchmark timed out after 120 seconds.
Attempting fix: {attempted_fix_description}
Attempts remaining: {N}/10
```

**If profiler hang suspected**: Add `--skip-profile` flag temporarily to verify benchmark runs without profiler.

### Step 2.5: Capture Baseline Metrics and Trace

<action>
Run benchmark multiple times and record metrics.
</action>

Once benchmark runs successfully:

1. Run benchmark multiple times to ensure consistency
2. Verify GPU environment consistency (see `shared/gpu_check.md`)
3. **The benchmark automatically generates `trace.json`** (profiler runs by default)
4. Record baseline metrics and trace location:

```yaml
baseline_metrics:
  forward_ms: {mean_value}
  forward_std_ms: {std_value}
  backward_ms: {mean_value}
  backward_std_ms: {std_value}
  total_ms: {forward_plus_backward}
  memory_mb: {value}
trace_json_path: "./profiler_output/trace.json"
```

**Important**: The trace.json file is **mandatory** for Phase 3 optimization.

### Step 2.5.1: Reproducibility Verification

<critical-rules>
Before proceeding, verify that benchmark results are reproducible and have low variance.
</critical-rules>

**Run 3 Independent Benchmark Executions:**

```bash
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run1.log
sleep 5
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run2.log
sleep 5
CUDA_VISIBLE_DEVICES={gpu_device} buck run @mode/opt {buck_target} 2>&1 | tee run3.log
```

**Analyze Variance Across Runs:**

```
Run 1: forward={X1} ms, backward={Y1} ms
Run 2: forward={X2} ms, backward={Y2} ms
Run 3: forward={X3} ms, backward={Y3} ms

Mean forward:  {mean} ms
Std forward:   {std} ms
CV forward:    {cv}%

Mean backward: {mean} ms
Std backward:  {std} ms
CV backward:   {cv}%
```

**Reproducibility Criteria:**

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

**Commit the benchmark tool as a separate commit** before starting optimizations.

```bash
hg add {benchmark_path} {buck_file}
hg commit -m "[benchmark] Add benchmark for {module_name}

- Target: {entry_point}
- Baseline forward: {forward_ms} ms
- Baseline backward: {backward_ms} ms"
```

---

## Output Format

<output-format>
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
- Backward: {backward_ms} ± {backward_std_ms} ms
- Total:    {total_ms} ms

Reproducibility Check:
- Forward CV:  {forward_cv}% - {PASS|WARNING|FAIL}
- Backward CV: {backward_cv}% - {PASS|WARNING|FAIL}
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
  backward_ms: {value}
  backward_std_ms: {value}
  memory_mb: {value}
reproducibility:
  forward_cv_percent: {value}
  backward_cv_percent: {value}
  status: "{PASS|WARNING|FAIL}"
```
</output-format>

**Proceed to Phase 3 automatically.**

---

## Error Handling

### No Idle GPU → `STOP_NO_GPU`

<critical-stop ref="STOP_NO_GPU">
No idle GPU available. Current GPU status:
{nvidia_smi_output}

All GPUs are currently in use. Benchmark results would have high variance.

Please either:
1. Wait for other jobs to complete
2. Free up GPU resources on one of the devices
3. Specify which GPU to wait for

Reply when ready to continue.
</critical-stop>

### Benchmark Failed After 10 Attempts → `STOP_BENCHMARK_FAILED`

<critical-stop ref="STOP_BENCHMARK_FAILED">
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
</critical-stop>

### High Variance Results (Log warning, proceed)

```
WARNING: High variance in benchmark results:
- Forward: {mean} ± {std} ms (std = {percent}% of mean)

This may indicate GPU contention. Proceeding with caution...
```
