# GPU Availability Check

Reusable module for verifying GPU availability before benchmarking.

## Why This Matters

If the GPU is busy or unavailable, benchmark results will have **extremely high variance** and be unreliable. This check must be performed before any benchmarking run.

## Required Bash Permissions

The `nvidia-smi` command requires Bash permission. See `shared/bash_permissions.md` for permission setup instructions.

## Automated GPU Selection

**This process is fully automated. No user interaction required unless no idle GPU is available.**

### Step 1: Check GPU Status

Run this command to check GPU availability:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits
```

### Step 2: Automatically Select an Idle GPU

Parse the output and select the first GPU with:
- **GPU utilization < 5%** - Indicates the GPU is idle
- **Sufficient free memory** - At least 10GB free

Example output:
```
0, 0, 79500
1, 95, 10000
2, 2, 79000
```

**Automatically select GPU 0** (first idle GPU with < 5% utilization).

Record the selected GPU:
```yaml
gpu_device: "0"
```

Log to user (informational only, no confirmation needed):
```
GPU Check: Selected GPU 0 (0% utilization, 79500 MiB free)
```

### Step 3: If No Idle GPU - CRITICAL STOP

**Only stop if ALL GPUs have > 5% utilization.**

```
CRITICAL ERROR: No idle GPU available for benchmarking

Current GPU status:
[nvidia-smi output]

All GPUs are currently in use (> 5% utilization). Cannot proceed with reliable benchmarks.

Please either:
1. Wait for other jobs to complete
2. Free up GPU resources on one of the devices

Reply when ready to continue.
```

**This is one of the 3 critical stop conditions for this skill.**

## Consistency Across Runs

When comparing benchmark results (e.g., baseline vs optimized), the GPU environment **MUST** be identical:

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
