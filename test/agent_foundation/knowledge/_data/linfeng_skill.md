
# Phase 1: Problem Analysis
​​​
## Steps
​​​
1. **Parse**: Current limitation, desired improvement, context
2. **Categorize**: Memory | Compute | Expressiveness | Scalability
3. **Quantify**: Current vs target (memory, FLOPs, QPS)
4. **Define metrics**: Primary, secondary, constraints
​​​
## Example: Scaling MLP Width
​​​
```yaml
problem:
  statement: "Double MLP hidden dim (4x→8x) without 2× FLOPs"
  bottleneck_type: "compute"
  current_metrics:
    hidden_mult: 4
    flops_per_layer: "8B"     # 2 × n × d × 4d
    memory_per_layer: "16MB"  # 4d × d × 2 bytes
    qps: 1000
  target_metrics:
    hidden_mult: 8
    flops_budget: "<1.5x current"
    memory_budget: "32MB"
    qps_target: ">800"
  context: "MLP is 60% of total model FLOPs"
```
​​​
### Key Formulas
​​​
| Metric | Formula | Example (n=1k, d=256, mult=4) |
|--------|---------|-------------------------------|
| MLP FLOPs | 2 × n × d × (mult×d) | 2 × 1k × 256 × 1024 = 524M |
| MLP Memory | mult × d² × 2B | 4 × 256² × 2 = 512KB |
| QPS impact | ~1/FLOPs | Inversely proportional |
​​​
→ Proceed to Phase 2
fbcode/cmsl/rank_evolve/model/skills/model_innovation/phases/02_solution_exploration.md
A
+26
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/phases‎/‎
02_solution_exploration
.md

Viewed
This file was added.
# Phase 2: Solution Exploration
​​​
## Steps
​​​
1. **Generate proposals** from multiple categories
2. **Analyze**: memory, FLOPs, QPS impact, feasibility
3. **Ensure diversity**: algorithmic, architectural, systems
​​​
## Example: Scaling MLP (4x→8x hidden)
​​​
| ID | Proposal | FLOPs | Memory | QPS | Feasibility |
|----|----------|-------|--------|-----|-------------|
| P1 | MoE (8 experts, top-2) | 2×base | 8×base | ~900 | MEDIUM |
| P2 | Low-Rank (r=64) | 0.5×base | 0.5×base | ~1100 | HIGH |
| P3 | Sparse MLP (50%) | 0.5×base | 1×base | ~950 | MEDIUM |
| P4 | GLU Variants | 1.5×base | 1.5×base | ~850 | HIGH |
| P5 | Factorized (d→k→8d) | 0.3×base | 0.6×base | ~1000 | HIGH |
| P6 | Quantized (INT8) | 1×base | 0.25×base | ~1200 | HIGH |
​​​
### Categories
​​​
- **Algorithmic**: P2, P3, P5 (change computation)
- **Architectural**: P1, P4 (change structure)
- **Systems**: P6 (optimize precision)
​​​
→ Proceed to Phase 3
fbcode/cmsl/rank_evolve/model/skills/model_innovation/phases/03_constraint_discovery.md
A
+41
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/phases‎/‎
03_constraint_discovery
.md

Viewed
This file was added.
# Phase 3: Constraint Discovery
​​​
## Steps
​​​
1. **Ask user** for constraints
2. **Register** with ID, type, impact
3. **Evaluate** proposals against constraints
4. **Eliminate** non-viable (cite constraint ID)
​​​
## Example: Scaling MLP
​​​
**User constraint**: "Need deterministic training, no dynamic routing. Also must maintain dense gradient flow."
​​​
```yaml
constraints:
  - id: "C_001"
    type: "infrastructure"
    statement: "No dynamic/conditional computation"
    impact: "MoE routing is non-deterministic"
    eliminates: [P1, P3]
  - id: "C_002"
    type: "quality"
    statement: "Dense gradient flow required"
    impact: "Sparse activations break gradient"
    eliminates: [P3]
```
​​​
### Evaluation
​​​
| Proposal | Deterministic | Dense Grad | Status |
|----------|---------------|------------|--------|
| P1 MoE | ❌ routing | ✅ | ❌ C_001 |
| P2 Low-Rank | ✅ | ✅ | ✅ |
| P3 Sparse | ❌ dynamic | ❌ | ❌ C_001, C_002 |
| P4 GLU | ✅ | ✅ | ✅ |
| P5 Factorized | ✅ | ✅ | ✅ |
| P6 Quantized | ✅ | ✅ | ✅ |
​​​
**Insight**: Constraint pushes toward static factorization over dynamic sparsity.
​​​
→ Proceed to Phase 4
fbcode/cmsl/rank_evolve/model/skills/model_innovation/phases/04_design_refinement.md
A
+39
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/phases‎/‎
04_design_refinement
.md

Viewed
This file was added.
# Phase 4: Design Refinement
​​​
## Steps
​​​
1. **Compare** surviving proposals
2. **Make decisions**: question, options, rationale, constraints
3. **Create diagram**
​​​
## Example: MLP Scaling Refinement
​​​
### Decision 1: Expansion Method
​​​
| Option | FLOPs | Quality | Complexity |
|--------|-------|---------|------------|
| P2 Low-Rank | 0.5× | Medium | Low |
| P4 GLU | 1.5× | High | Low |
| P5 Factorized | 0.3× | Medium | Medium |
​​​
```yaml
decision:
  id: "DD1"
  question: "MLP expansion method?"
  selected: "P5 Factorized + P4 GLU hybrid"
  rationale: "Best FLOPs/quality tradeoff"
  constraints_applied: ["C_001", "C_002"]
  ablation_needed: "ABL_001"
```
​​​
### Decision 2: Bottleneck Dim
​​​
| k | FLOPs Savings | Quality Risk |
|---|---------------|--------------|
| 32 | 75% | High |
| 64 | 50% | Medium |
| 128 | 25% | Low |
​​​
**Selected**: k=64 (balance efficiency/quality)
​​​
→ Proceed to Phase 5
fbcode/cmsl/rank_evolve/model/skills/model_innovation/phases/05_ablation_planning.md
A
+38
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/phases‎/‎
05_ablation_planning
.md

Viewed
This file was added.
# Phase 5: Ablation Planning
​​​
## Steps
​​​
1. **Identify ablations** from design decisions
2. **Define**: hypothesis, baseline, variants, metrics
3. **Create matrix** with resource estimates
​​​
## Example: Factorized GLU MLP
​​​
| ID | Component | Baseline | Variants | Metric |
|----|-----------|----------|----------|--------|
| ABL_001 | MLP type | Dense 4x | Factorized, GLU, Hybrid | NE, QPS |
| ABL_002 | Bottleneck k | k=64 | 32, 128 | NE, FLOPs |
| ABL_003 | GLU variant | SiLU | GELU, Swish | NE |
​​​
### ABL_001 Detail
​​​
```yaml
ablation:
  id: "ABL_001"
  hypothesis: "Factorized GLU matches dense quality at 50% FLOPs"
  baseline: {mlp_type: "dense", hidden_mult: 4}
  variants:
    - {mlp_type: "factorized", bottleneck: 64}
    - {mlp_type: "glu", hidden_mult: 4}
    - {mlp_type: "factorized_glu", bottleneck: 64}
  metrics: ["NE", "QPS", "FLOPs"]
  resources: "4 runs × 8 GPU-hours = 32 GPU-hours"
```
​​​
### Success Criteria
​​​
- NE regression ≤ 0.05%
- FLOPs reduction ≥ 40%
- QPS ≥ 900
​​​
→ Proceed to Phase 6
fbcode/cmsl/rank_evolve/model/skills/model_innovation/phases/06_implementation_planning.md
A
+44
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/phases‎/‎
06_implementation_planning
.md

Viewed
This file was added.
# Phase 6: Implementation Planning
​​​
## Steps
​​​
1. **Code pointers**: file, lines, change
2. **Phases**: group logically, estimate effort
3. **Config flags** for ablations
​​​
## Example: Factorized GLU MLP
​​​
### Code Pointers
​​​
```yaml
pointers:
  - file: "mlp.py:L50-100"
    change: "Add FactorizedMLP, GLUMLP modules"
  - file: "config.py:L40"
    change: "Add mlp_type, bottleneck_dim params"
  - file: "transformer.py:L120"
    change: "Replace MLP with configurable variant"
```
​​​
### Implementation Phases
​​​
| Phase | Work | Effort | Risk |
|-------|------|--------|------|
| 1 | Add config flags | 1d | Low |
| 2 | Implement FactorizedMLP | 2d | Low |
| 3 | Implement GLUMLP | 1d | Low |
| 4 | Hybrid + tests | 2d | Med |
| 5 | Ablation runs | 3d | Low |
​​​
### Config for Ablations
​​​
```python
@dataclass
class MLPConfig:
    mlp_type: str = "dense"      # "dense" | "factorized" | "glu" | "factorized_glu"
    hidden_mult: int = 4         # Base: 4, Target: 8
    bottleneck_dim: int = 64     # ABL_002: try 32, 128
    activation: str = "silu"     # ABL_003: try "gelu", "swish"
```
​​​
→ Done
fbcode/cmsl/rank_evolve/model/skills/model_innovation/shared/templates.md
A
+75
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/shared‎/‎
templates
.md

Viewed
This file was added.
# Templates
​​​
## Problem
​​​
```yaml
problem:
  statement: ""
  bottleneck_type: "memory|compute|expressiveness|scalability"
  current_metrics: {}
  target_metrics: {}
  context: ""
```
​​​
## Proposal
​​​
```yaml
proposal:
  id: "P1"
  name: ""
  category: "algorithmic|architectural|systems|hybrid"
  description: ""
  resource_impact: ""
  feasibility: "HIGH|MEDIUM|LOW"
  trade_offs: []
  effort: ""
```
​​​
## Constraint
​​​
```yaml
constraint:
  id: "C_001"
  type: "resource|architecture|infrastructure|quality|timeline"
  statement: ""
  impact: ""
  implication: ""
  eliminates: []
```
​​​
## Design Decision
​​​
```yaml
decision:
  id: "DD1"
  question: ""
  options: []
  selected: ""
  rationale: ""
  constraints_applied: []
  ablation_needed: ""
```
​​​
## Ablation
​​​
```yaml
ablation:
  id: "ABL_001"
  component: ""
  category: "component|hyperparameter|architecture|scale"
  hypothesis: ""
  baseline: ""
  variants: []
  metrics: []
  expected_outcome: ""
```
​​​
## Code Pointer
​​​
```yaml
code_pointer:
  file: ""
  lines: ""
  purpose: ""
  change: ""
```
fbcode/cmsl/rank_evolve/model/skills/model_innovation/SKILL.md
A
+53
-0
fbcode/
‎cmsl‎/rank_evolve‎/model‎/skills‎/model_innovation‎/‎
SKILL
.md

Viewed
This file was added.
# Model Innovation Skill
​​​
Problem-driven workflow for ML architecture innovation.
​​​
## Phases
​​​
```
Problem Analysis → Solution Exploration → Constraint Discovery → Design Refinement → Ablation Planning → Implementation
```
​​​
## Rules
​​​
1. **Problem First**: Quantify bottleneck before proposing solutions
2. **Constraint Traceability**: Every design decision traces to a constraint
3. **Ablation Ready**: Every component must be independently testable
​​​
## Phase Files
​​​
| Phase | File | Purpose |
|-------|------|---------|
| 1 | `phases/01_problem_analysis.md` | Quantify bottleneck, define metrics |
| 2 | `phases/02_solution_exploration.md` | Generate diverse proposals |
| 3 | `phases/03_constraint_discovery.md` | Apply constraints, narrow options |
| 4 | `phases/04_design_refinement.md` | Compare, decide, document |
| 5 | `phases/05_ablation_planning.md` | Plan validation experiments |
| 6 | `phases/06_implementation_planning.md` | Code pointers, .plan.md |
​​​
## Shared Templates
​​​
| File | Purpose |
|------|---------|
| `shared/templates.md` | All templates (problem, proposal, constraint, decision, ablation) |
​​​
## State File
​​​
```yaml
# .model_innovation_state.yaml
current_phase: ""
problem: {}
proposals: []
constraints: []
design_decisions: []
ablation_experiments: []
```
​​​
## Critical Stops
​​​
Only pause for user input when:
- Cannot identify bottleneck
- Cannot generate proposals
- No constraints provided
- Tie between proposals
- Cannot locate code
