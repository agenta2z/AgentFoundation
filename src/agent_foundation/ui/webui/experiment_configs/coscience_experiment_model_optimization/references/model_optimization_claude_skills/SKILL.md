# QPS Optimization Skill Set

A modular, interactive workflow for optimizing MAST training performance bottlenecks while ensuring numerical equivalence.

---

<critical-rules id="MEMORY_NEUTRAL">
## Core Constraints

**Goal**: Maximize forward/backward pass speed for multi-GPU cluster training.

**Memory Constraint**: All optimizations MUST be memory-neutral. If an optimization increases memory usage, ASK USER before applying (they decide the trade-off).

**Optimization Priority** (follow this order):
1. PyTorch glue code / CPU-bound fixes (low-hanging fruit)
2. Forward pass speedup
3. Backward pass speedup
4. Communication-computation overlap
5. Kernel micro-optimization (last resort)

Reference this constraint as `MEMORY_NEUTRAL` throughout phases.
</critical-rules>

<critical-rules id="PHASE_READS">
## Phase File Reading

**Before starting any phase**, use the Read tool to load the phase instructions:
- Phase 1: `Read("phases/01_setup.md")`
- Phase 2: `Read("phases/02_benchmark.md")`
- Phase 3: `Read("phases/03_optimize.md")` - MUST complete pre-flight checklist
- Phase 4: `Read("phases/04_validate.md")`
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
**BOOTSTRAP REQUIREMENT**: This skill requires pre-existing permissions to run autonomously.

Before invoking this skill, ensure one of these is in place:
1. A `.claude/settings.local.json` file with `"Bash(*)"` and `"Write(*)"` permissions exists, OR
2. User runs Claude Code with `--dangerously-skip-permissions` flag

**Create the permission file manually before starting** (run in terminal):
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

## State Persistence

**CRITICAL**: State must be persisted to enable session resumption and track progress across interruptions.

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
3. **Proceed to next phase automatically** (no user confirmation needed)

### Phase Transitions
- **Phase 1 → 2**: Must have source_path and entry_point
- **Phase 2 → 3**: Must have working benchmark with baseline metrics
- **Phase 3 → 4**: Must have optimization with improvement
- **Phase 4 → Done**: Must have MAST validation complete

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

**All other issues**: Handle automatically (rollback, try alternatives, log and proceed).
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
