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

**STOP and wait for user to provide MAST CLI.**

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

**Workflow complete.**

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

2. **Automatically rollback** the problematic optimization
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

2. **Automatically rollback** the most recent optimization
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

**Complete workflow with documented findings.**
