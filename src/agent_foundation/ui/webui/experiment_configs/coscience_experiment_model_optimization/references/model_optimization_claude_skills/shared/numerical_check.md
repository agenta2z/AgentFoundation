# Numerical Equivalence Check

Reusable module for verifying numerical equivalence between baseline and optimized outputs.

## Why This Matters

Optimizations must preserve numerical correctness. Even small differences can compound during training and cause divergence. This check must be performed:
1. **After each optimization iteration** (Phase 3) - early detection of issues
2. **During final validation** (Phase 4) - comprehensive verification

## Quick Check (Use During Optimization Iterations)

**This is a lightweight check for rapid iteration. Run after each optimization change.**

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
