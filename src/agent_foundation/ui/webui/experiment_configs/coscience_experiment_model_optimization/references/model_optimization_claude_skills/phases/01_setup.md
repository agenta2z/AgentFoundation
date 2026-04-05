# Phase 1: Setup

Gather optimization target information and understand the code structure.

## Prerequisites

- User has identified a module/operation to optimize from MAST training
- User has access to the MAST run trace or knows the bottleneck location

## Steps

### Step 1.0: Proactive Code Exploration (Before Prompting User)

<critical-rules>
**DO NOT ask user any questions during this step.** Gather all information silently first.
</critical-rules>

**1.0.1: Check for Existing State**

Search for existing optimization state files:
```bash
find . -name ".qps_optimization_state.yaml" -type f 2>/dev/null | head -5
```

Store findings for Step 1.1 (do not prompt user yet).

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

**Proceed to Phase 2 automatically.**

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
