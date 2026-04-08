 Design Discussion: Unified Workspace & Output Path Contract for Inferencers

 Context

 The three composable flow inferencers (DualInferencer, PTI, BTA) each handle workspace directories and output paths differently. This creates inconsistency and makes composition fragile. The user proposes a unified design. Below is my analysis of each design point.

 ---
 Point-by-Point Analysis

 (1) BTA should NOT own per-worker output paths — only the workspace root

 I agree. This is already the correct separation of concerns in the current code.

 Current state:
 - BTA has checkpoint_dir (line 119 of breakdown_then_aggregate_inferencer.py) which serves as its workspace root
 - BTA passes workers to worker_factory(sub_query, index) — the factory decides internal layout
 - In executor.py:463, the application layer creates checkpoint_dir/bta/worker_{index}/outputs/facet_{index}.md
 - BTA itself never references these paths

 Why this is right: A worker could be a PTI, a DualInferencer, a simple API inferencer, or anything else. Each has its own workspace conventions. BTA should only say "here's your workspace root" and let the worker organize beneath it. The worker's internal directory layout is its
  own business.

 What needs to change: BTA should formalize passing a workspace_root (not just relying on the application layer to construct paths from checkpoint_dir). The worker_factory signature could receive the workspace root for that worker, e.g. worker_factory(sub_query, index,
 workspace_root=<bta_workspace>/workers/worker_{index}).

 ---
 (2) Align PTI workspace structure/naming with BTA

 Current PTI workspace:
 workspace_path/
 ├── outputs/          # plan + implementation files
 ├── analysis/
 ├── results/
 ├── checkpoints/pti/  # Workflow checkpoints + child DualInferencer checkpoints
 ├── logs/
 ├── _runtime/
 └── followup_iterations/iteration_N/  # same structure recursively

 Current BTA workspace (as constructed by executor.py):
 checkpoint_dir/
 ├── breakdown_result.json        # BTA's own checkpoint
 ├── aggregator_result/           # aggregator checkpoint
 └── bta/worker_N/outputs/        # worker outputs (app-layer convention)

 Proposed aligned structure:

 The key insight: both PTI and BTA are "parent inferencers" that manage child inferencers. They should follow the same workspace convention. The workspace separates three concerns:

 - outputs/ — final output only. The "answer." One file (or a small set). This is what the parent reads.
 - artifacts/ — intermediate round-by-round files, audit trail, raw outputs. Useful for debugging and transparency, but not the deliverable.
 - checkpoints/ — resume/checkpoint data for crash recovery.

 workspace_root/                    # the inferencer's workspace root
 ├── outputs/                       # FINAL output only (the deliverable)
 ├── artifacts/                     # intermediate outputs, audit trail (old "outputs/" behavior)
 ├── checkpoints/                   # resume/checkpoint data
 │   └── ...                        # Workflow/WorkGraph checkpoint files
 └── children/                      # child inferencer workspaces
     ├── <child_name>/              # each child gets its own workspace_root
     │   ├── outputs/               # child's final output
     │   ├── artifacts/             # child's intermediate outputs
     │   ├── checkpoints/
     │   └── children/              # recursive if the child is also a composing inferencer
     └── ...

 The class-level output_path, when relative, resolves relative to outputs/ (not workspace_root directly). This means output_path="plan.md" → workspace_root/outputs/plan.md.

 For DualInferencer specifically (with output_path="plan.md"):
 workspace_root/
 ├── outputs/
 │   └── plan.md                    # FINAL consensus result (output_path = "plan.md")
 ├── artifacts/
 │   ├── round01_plan.md            # round 1: initial proposal (base_inferencer)
 │   ├── round02_plan.md            # round 2: after first fix cycle (fixer_inferencer)
 │   └── round03_plan.md            # round 3: after second fix cycle → consensus reached
 ├── logs/
 │   ├── Round01/                   # per-round prompts/responses (from Debuggable logging)
 │   │   ├── InitialPrompt.md
 │   │   └── InitialResponse.md
 │   ├── Round02/
 │   │   ├── ReviewPrompt.md
 │   │   ├── ReviewResponse.md
 │   │   ├── FollowupPrompt.md
 │   │   └── FollowupResponse.md
 │   └── Round03/
 │       └── ...
 └── checkpoints/
     └── attempt_01/step_*.json     # Workflow step checkpoints

 DualInferencer Artifact Naming Convention

 Artifact filenames are derived from output_path with a round index prefix:

 round{NN}_{output_path_basename}

 - NN = zero-padded round number (01, 02, 03, ...), using total_iterations counter
 - output_path_basename = the basename of the class-level output_path (e.g., plan.md)

 How rounds map to the consensus loop:

 ┌───────┬───────────────────────────────────────────────┬───────────────────────────┐
 │ Round │                 What happens                  │     Artifact written      │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 01    │ base_inferencer proposes                      │ artifacts/round01_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 02    │ fixer_inferencer addresses review issues      │ artifacts/round02_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 03    │ fixer_inferencer addresses second review      │ artifacts/round03_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ ...   │ (continues until consensus or max_iterations) │ ...                       │
 └───────┴───────────────────────────────────────────────┴───────────────────────────┘

 After consensus or loop exhaustion, _finalize_response() copies the last round's artifact to outputs/{output_path}:
 def _finalize_response(self):
     # Last round artifact → final output
     last_round = self._state["total_iterations"]
     basename = os.path.basename(self.output_path)  # e.g., "plan.md"
     src = self.workspace.artifact_path(f"round{last_round:02d}_{basename}")
     dst = self.workspace.output_path(basename)
     if os.path.isfile(src):
         shutil.copy2(src, dst)

 Where _maybe_replace_with_file_reference writes (before → after):
 # Before: writes to inference_config["output_path"] with {{ round_index }}
 resolved_path = output_path_template.replace("{{ round_index }}", str(round_index))

 # After: writes to workspace artifacts with round-indexed basename
 basename = os.path.basename(self.output_path)  # "plan.md"
 resolved_path = self.workspace.artifact_path(f"round{total_iterations:02d}_{basename}")

 For PTI specifically — single iteration (output_mode=IMPLEMENTATION):
 workspace_root/
 ├── outputs/
 │   └── implementation.md          # ← copied from children/executor/outputs/ by _finalize_outputs()
 ├── artifacts/
 │   ├── .plan_completed            # phase completion markers
 │   ├── .impl_completed
 │   ├── request.txt                # original request text
 │   └── analysis_summary.json      # iteration control flow decision
 ├── checkpoints/                   # __wf_checkpoint__.json, step_*.json
 └── children/
     ├── planner/
     │   ├── outputs/
     │   │   └── plan.md
     │   ├── artifacts/
     │   │   ├── round1_plan.md
     │   │   └── round2_plan.md
     │   └── checkpoints/
     ├── executor/
     │   ├── outputs/
     │   │   └── implementation.md
     │   ├── artifacts/
     │   └── checkpoints/
     └── analyzer/
         ├── outputs/
         │   └── analysis.md
         ├── artifacts/
         └── checkpoints/

 For PTI — multi-iteration (3 iterations, output_mode=IMPLEMENTATION):
 workspace_root/
 ├── outputs/
 │   └── implementation.md          # ← copied from LAST iteration (iteration_3) by _finalize_outputs()
 ├── artifacts/
 │   ├── request.txt
 │   ├── .plan_completed
 │   ├── .impl_completed
 │   └── analysis_summary.json
 ├── checkpoints/                   # PTI's own workflow checkpoints (stable across iterations)
 ├── children/                      # iteration 1's children
 │   ├── planner/...
 │   ├── executor/...
 │   └── analyzer/...
 └── followup_iterations/
     ├── iteration_2/               # full workspace for iteration 2
     │   ├── artifacts/
     │   │   ├── request.txt        # iteration handoff text
     │   │   └── analysis_summary.json
     │   ├── checkpoints/
     │   └── children/
     │       ├── planner/...
     │       ├── executor/...
     │       └── analyzer/...
     └── iteration_3/               # last iteration — source of final output
         ├── artifacts/
         │   └── ...
         ├── checkpoints/
         └── children/
             ├── planner/outputs/plan.md
             ├── executor/outputs/implementation.md   # ← _finalize_outputs() copies FROM here
             └── analyzer/outputs/analysis.md

 For PTI with multiple output mode (output_mode=PLAN | IMPLEMENTATION):
 workspace_root/
 ├── outputs/                       # multiple deliverables, all from last iteration
 │   ├── plan.md                    # ← from last iteration's planner
 │   └── implementation.md          # ← from last iteration's executor
 ├── ...

 Key insight: Regardless of how many iterations ran, workspace_root/outputs/ always contains the final deliverables. _finalize_outputs() handles the copy from the last iteration's children. The parent (or user) only needs to look at workspace_root/outputs/ — never needs to
 traverse followup_iterations/.

 PTI Output Mode

 PTI's final output is ambiguous — it could be the plan, implementation, analysis, or a combination. A class-level output_mode enum determines which child outputs PTI surfaces as its own final output.

 from enum import Flag, auto

 class PTIOutputMode(Flag):
     """Which child outputs PTI surfaces as its own final output."""
     PLAN = auto()
     IMPLEMENTATION = auto()
     ANALYSIS = auto()

     # Common combinations
     PLAN_AND_IMPLEMENTATION = PLAN | IMPLEMENTATION
     ALL = PLAN | IMPLEMENTATION | ANALYSIS

 On PlanThenImplementInferencer:
 output_mode: PTIOutputMode = attrib(default=PTIOutputMode.IMPLEMENTATION)

 Behavior by cardinality:

 ┌───────────────────────────────┬───────────────────────────────────────────────────────────────────┬──────────────────────────────────────────┐
 │          output_mode          │                         outputs/ content                          │      resolve_output_path() returns       │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ PLAN (single)                 │ outputs/plan.md                                                   │ workspace_root/outputs/plan.md           │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ IMPLEMENTATION (single)       │ outputs/implementation.md                                         │ workspace_root/outputs/implementation.md │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ ANALYSIS (single)             │ outputs/analysis.md                                               │ workspace_root/outputs/analysis.md       │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ PLAN | IMPLEMENTATION (multi) │ outputs/plan.md + outputs/implementation.md                       │ workspace_root/outputs/ (the folder)     │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ ALL (multi)                   │ outputs/plan.md + outputs/implementation.md + outputs/analysis.md │ workspace_root/outputs/ (the folder)     │
 └───────────────────────────────┴───────────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘

 Flag-to-child mapping (single source of truth):
 _OUTPUT_MODE_MAP = {
     PTIOutputMode.PLAN: ("planner", "plan.md"),
     PTIOutputMode.IMPLEMENTATION: ("executor", "implementation.md"),
     PTIOutputMode.ANALYSIS: ("analyzer", "analysis.md"),
 }

 Finalization — multi-iteration aware, driven by the map:

 _finalize_outputs() always copies from the last iteration's children to the workspace root's outputs/. This is called at the end of _ainfer(), after the workflow completes.

 def _finalize_outputs(self):
     """Copy selected child outputs from the LAST iteration to workspace root outputs/.
  
     Regardless of how many iterations ran, the workspace root's outputs/
     always contains the final deliverables. The parent or user only needs
     to look at workspace_root/outputs/.
     """
     last_iter = self._state.get("iteration", 1)

     # Get the workspace for the last iteration
     iter_ws_path = self._get_iteration_workspace(self.workspace.root, last_iter)
     iter_workspace = InferencerWorkspace(root=iter_ws_path)

     os.makedirs(self.workspace.outputs_dir, exist_ok=True)

     for flag, (child_name, filename) in _OUTPUT_MODE_MAP.items():
         if flag in self.output_mode:
             src = iter_workspace.child_output(child_name, filename)
             dst = self.workspace.output_path(filename)
             if os.path.isfile(src):
                 shutil.copy2(src, dst)

 def resolve_output_path(self, runtime_override=None):
     if runtime_override:
         return super().resolve_output_path(runtime_override)
     # Count active flags (exclude compound flags like ALL)
     active = [f for f in _OUTPUT_MODE_MAP if f in self.output_mode]
     if len(active) == 1:
         _, filename = _OUTPUT_MODE_MAP[active[0]]
         return self.workspace.output_path(filename)
     else:
         # Multiple outputs → return the outputs/ folder
         return self.workspace.outputs_dir

 Where _finalize_outputs() is called:
 async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
     # ... setup, resume detection ...
     await Workflow._arun(self, inference_input, **_inference_args)

     # Copy final outputs from last iteration to workspace root
     if self.workspace:
         self._finalize_outputs()

     return self._build_response_from_state(self._state)

 Resume safety: _finalize_outputs() is idempotent — it just copies files via shutil.copy2. If PTI crashes after the last iteration completes but before _finalize_outputs() runs, resume will detect all steps completed (via checkpoints), skip re-execution, and _finalize_outputs()
 runs again safely producing the same result.

 Multi-iteration workspace resolution reuses the existing _get_iteration_workspace:
 - Iteration 1 → workspace_root/ (children are at workspace_root/children/)
 - Iteration N>1 → workspace_root/followup_iterations/iteration_N/ (children are at that path's children/)
 - _finalize_outputs() resolves the last iteration's workspace, finds the children there, copies to workspace_root/outputs/

 What changed vs. old PTI layout:
 - results/ eliminated → analysis_summary.json moves to artifacts/ (it's PTI's own control-flow artifact, not a deliverable)
 - analysis/ eliminated → analyzer gets its own workspace under children/analyzer/ (consistent with planner/executor treatment)
 - Every child inferencer now lives under children/ — no special-case directories
 - output_mode enum declaratively controls which child outputs become PTI's deliverables

 Comparison — how each inferencer determines its final output:

 ┌────────────────┬─────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
 │   Inferencer   │                 Final output is...                  │              Determined by...               │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ DualInferencer │ Last consensus round result                         │ Always — inherent to consensus loop         │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ BTA            │ Aggregator output (or last worker if no aggregator) │ Always — inherent to diamond graph          │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ PTI            │ Plan, implementation, analysis, or combination      │ output_mode: PTIOutputMode class-level enum │
 └────────────────┴─────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

 For BTA specifically:
 workspace_root/
 ├── outputs/
 │   └── result.md                  # aggregated final output (output_path = "result.md")
 ├── artifacts/
 │   └── aggregator_raw_output.md   # raw aggregator response (audit trail)
 ├── checkpoints/                   # breakdown_result.json, aggregator_result/
 └── children/
     ├── worker_0/                  # each worker's workspace_root
     │   ├── outputs/               # worker's final output
     │   ├── artifacts/             # worker's intermediate outputs
     │   └── ...                    # could be PTI, DualInferencer, etc.
     ├── worker_1/
     └── ...

 Key naming alignment:
 - workspace_root (not workspace_path or checkpoint_dir) as the universal parameter name
 - outputs/ — final deliverable output only; class-level output_path resolves relative to this
 - artifacts/ — intermediate/round-by-round files (audit trail, debugging)
 - checkpoints/ — resume/checkpoint data
 - children/ — child inferencer workspace roots

 ---
 (3) DualInferencer should have its own workspace root

 I agree. Currently DualInferencer has:
 - checkpoint_dir — used for Workflow step checkpoints (attempt_NN/step_*.json)
 - output_path — passed purely via inference_config["output_path"] at inference time, not a class-level attribute

 DualInferencer has no class-level output path and no workspace root. Its output path is always injected at inference time through inference_config["output_path"] (see _maybe_replace_with_file_reference at line 1027). When PTI composes DualInferencer, PTI rewrites this
 inference_config output_path in _build_iteration_config (line 582-584).

 With the new design: DualInferencer gets workspace, and its layout becomes:
 workspace_root/
 ├── outputs/                       # FINAL consensus output only
 │   └── output.md                  # (output_path = "output.md")
 ├── artifacts/                     # round-indexed intermediate files
 │   ├── round01_output.md          # initial proposal
 │   ├── round02_output.md          # after fix cycle 1
 │   └── round03_output.md          # after fix cycle 2 (consensus)
 ├── logs/                          # per-round prompts/responses
 │   ├── Round01/...
 │   └── Round02/...
 └── checkpoints/                   # attempt_NN/step_*.json

 DualInferencer's _maybe_replace_with_file_reference currently writes each round to inference_config["output_path"] with {{ round_index }} substitution. Under the new design, those per-round files go to artifacts/round{NN}_{output_path}, and _finalize_response copies the final
 consensus result to outputs/{output_path}. See "DualInferencer Artifact Naming Convention" above for details.

 ---
 (4) Unified workspace + class-level output_path + inference-time override

 I agree with this layered approach. Formalized via InferencerWorkspace:

 On InferencerBase:
 workspace: Optional[InferencerWorkspace] = attrib(default=None)
 output_path: Optional[str] = attrib(default=None)  # relative to workspace.outputs_dir

 Resolution rules:
 1. If output_path is relative and workspace is set, resolve to workspace.outputs_dir / output_path
 2. If output_path is absolute, use as-is
 3. Inference-time output_path (via infer(..., output_path=...)) overrides class-level
 4. Same resolution rules apply to inference-time override

 On InferencerBase:
 def resolve_output_path(self, runtime_override: Optional[str] = None) -> Optional[str]:
     path = runtime_override or self.output_path
     if path is None:
         return None
     if self.workspace and not os.path.isabs(path):
         return self.workspace.output_path(path)
     return path

 Simple API inferencers (ClaudeApiInferencer, etc.) leave workspace=None — no overhead.

 ---
 (5) Composing inferencers use class-level relative output_path only — no inference-time override for children

 I agree — this is the critical design rule for predictable composition.

 The rule: When a parent (BTA/PTI) composes child inferencers, it:
 1. Sets child.workspace = parent.workspace.child("<child_name>")
 2. Relies on the child's class-level output_path (relative to child's workspace.outputs_dir)
 3. Never passes inference-time output_path override to the child

 This means:
 - The parent can predetermine where the child's output will be via child.resolve_output_path()
 - Or equivalently via parent.workspace.child_output("child_name", "output.md")
 - No duplicate path construction in two places (the executor.py problem)

 Example — PTI composing DualInferencer:
 # PTI sets up planner — one line
 self.planner_inferencer.workspace = self.workspace.child("planner")
 self.planner_inferencer.workspace.ensure_dirs()
 # planner.output_path = "plan.md" (set at construction, class-level)

 # PTI reads the plan after inference:
 plan_path = self.planner_inferencer.resolve_output_path()
 # → workspace_root/children/planner/outputs/plan.md

 # Or without touching the child object:
 plan_path = self.workspace.child_output("planner", "plan.md")

 # Intermediate rounds are in the child's artifacts:
 # workspace_root/children/planner/artifacts/round*_plan.md

 Example — BTA composing workers:
 def _build_diamond_graph(self, sub_queries, ...):
     for i, sq in enumerate(sub_queries):
         worker = self.worker_factory(sub_query=sq, index=i)
         worker.workspace = self.workspace.child(f"worker_{i}")
         worker.workspace.ensure_dirs()
         # worker.output_path is whatever the worker class defines
         # BTA reads from: worker.resolve_output_path() after inference

 Aggregator prompt builder — single source of truth via closure-captured paths:

 Note: WorkGraph passes *worker_results (varargs) to the aggregation function — NOT worker objects. So worker output paths must be captured at graph construction time via closure.

 def _build_diamond_graph(self, sub_queries, ...):
     worker_output_paths = []
     for i, sq in enumerate(sub_queries):
         worker = self.worker_factory(sub_query=sq, index=i)
         worker.workspace = self.workspace.child(f"worker_{i}")
         worker.workspace.ensure_dirs()
         # Capture resolved path ONCE at construction time
         worker_output_paths.append(worker.resolve_output_path())
         # ... create WorkGraphNode ...

     # Aggregator closure captures worker_output_paths — same paths workers write to
     def _agg_with_paths(worker_results, original_query):
         parts = []
         for idx, path in enumerate(worker_output_paths):
             parts.append(f"### Facet {idx+1}\nRead from: `{path}`")
         return "\n\n".join(parts)

 Current gap this fixes: In executor.py, the same path checkpoint_dir/bta/worker_{idx}/outputs/facet_{idx}.md is constructed independently in worker_factory (line 463) and agg_prompt_builder (line 504). With InferencerWorkspace, the path is resolved once at graph construction
 time and shared via closure — single source of truth.

 ---
 InferencerWorkspace Class

 Motivation

 All three inferencers duplicate the same workspace operations: os.path.join for path construction, os.makedirs for directory creation, glob.glob for artifact scanning, and marker file management. An InferencerWorkspace class formalizes the directory layout, centralizes these
 operations, and makes composition clean.

 Design Principles

 - Path management + directory layout, not file I/O. Inferencers read/write their own files. The workspace resolves paths and manages directories.
 - Marker files are the exception — markers (.plan_completed, etc.) are a workspace concern (state tracking), not an inferencer output concern, so the workspace owns them.
 - Recursive composition via child() — workspace.child("planner") returns a new InferencerWorkspace rooted at children/planner/. This is how parent inferencers set up child workspaces.
 - subdir() for inferencer-specific directories — PTI can create analysis/, results/ without polluting the base class.

 Class Definition

 @attrs
 class InferencerWorkspace:
     """Standard directory layout manager for inferencer workspaces.
  
     Provides path resolution, directory creation, child workspace management,
     and artifact scanning. Does NOT own file I/O for outputs/artifacts —
     inferencers handle that themselves.
  
     Standard layout:
         root/
         ├── outputs/       # final deliverable output
         ├── artifacts/     # intermediate round-by-round files, audit trail
         ├── checkpoints/   # resume/checkpoint data
         ├── logs/          # prompt/response logs
         ├── children/      # child inferencer workspace roots
         └── (custom)/      # inferencer-specific dirs via subdir()
     """
     root: str = attrib()  # absolute path to workspace root

     # ── Standard directory properties ──

     @property
     def outputs_dir(self) -> str:
         """Final output directory. output_path resolves relative to this."""
         return os.path.join(self.root, "outputs")

     @property
     def artifacts_dir(self) -> str:
         """Intermediate round-by-round files, audit trail."""
         return os.path.join(self.root, "artifacts")

     @property
     def checkpoints_dir(self) -> str:
         """Resume/checkpoint data."""
         return os.path.join(self.root, "checkpoints")

     @property
     def logs_dir(self) -> str:
         """Logging output (prompts, responses per round)."""
         return os.path.join(self.root, "logs")

     @property
     def children_dir(self) -> str:
         """Child inferencer workspace roots."""
         return os.path.join(self.root, "children")

     # ── Directory initialization ──

     def ensure_dirs(self, *extra_subdirs: str) -> None:
         """Create standard directories + optional extras.
  
         Example:
             workspace.ensure_dirs()  # standard dirs only; no PTI-specific extras needed
         """
         for d in (self.outputs_dir, self.artifacts_dir,
                   self.checkpoints_dir, self.logs_dir):
             os.makedirs(d, exist_ok=True)
         for sub in extra_subdirs:
             os.makedirs(os.path.join(self.root, sub), exist_ok=True)

     # ── Path resolution ──

     def output_path(self, relative: str) -> str:
         """Resolve a path relative to outputs/."""
         return os.path.join(self.outputs_dir, relative)

     def artifact_path(self, relative: str) -> str:
         """Resolve a path relative to artifacts/."""
         return os.path.join(self.artifacts_dir, relative)

     def checkpoint_path(self, relative: str) -> str:
         """Resolve a path relative to checkpoints/."""
         return os.path.join(self.checkpoints_dir, relative)

     def log_path(self, relative: str) -> str:
         """Resolve a path relative to logs/."""
         return os.path.join(self.logs_dir, relative)

     def subdir(self, name: str) -> str:
         """Access a custom subdirectory (e.g., 'analysis', 'results').
         Does NOT create it — call ensure_dirs() or os.makedirs() separately.
         """
         return os.path.join(self.root, name)

     # ── Resolve output_path (relative or absolute) ──

     def resolve(self, path: str) -> str:
         """If path is relative, resolve to outputs/. If absolute, return as-is."""
         if os.path.isabs(path):
             return path
         return self.output_path(path)

     # ── Child workspace management ──

     def child(self, name: str) -> 'InferencerWorkspace':
         """Create a child workspace under children/<name>/.
  
         Returns a new InferencerWorkspace instance. Does not create
         directories on disk — call child.ensure_dirs() when ready.
         """
         return InferencerWorkspace(root=os.path.join(self.children_dir, name))

     def child_output(self, child_name: str, output_path: str) -> str:
         """Resolve a child's output path without creating the child workspace object.
  
         Equivalent to: self.child(child_name).output_path(output_path)
         Useful when the parent just needs to read a child's known output.
         """
         return os.path.join(self.children_dir, child_name, "outputs", output_path)

     # ── Artifact scanning (replaces scattered glob patterns) ──

     def glob_artifacts(self, pattern: str) -> List[str]:
         """Glob within artifacts/ dir. Returns sorted list.
  
         Example:
             workspace.glob_artifacts("round*_plan.md")
         """
         return sorted(glob.glob(os.path.join(self.artifacts_dir, pattern)))

     def glob_outputs(self, pattern: str) -> List[str]:
         """Glob within outputs/ dir. Returns sorted list."""
         return sorted(glob.glob(os.path.join(self.outputs_dir, pattern)))

     # ── Marker files (replaces PTI's _write_step_completion_marker) ──

     def write_marker(self, name: str, metadata: dict = None) -> None:
         """Write a completion marker to artifacts/.<name>_completed.
  
         Example:
             workspace.write_marker("plan")  # writes artifacts/.plan_completed
         """
         marker_path = self.artifact_path(f".{name}_completed")
         os.makedirs(os.path.dirname(marker_path), exist_ok=True)
         with open(marker_path, "w") as f:
             json.dump(metadata or {
                 "completed_at": datetime.now(timezone.utc).isoformat(),
                 "step": name,
             }, f)

     def has_marker(self, name: str) -> bool:
         """Check if a completion marker exists."""
         return os.path.isfile(self.artifact_path(f".{name}_completed"))

     def clear_marker(self, name: str) -> None:
         """Remove a completion marker (e.g., on re-attempt)."""
         marker_path = self.artifact_path(f".{name}_completed")
         if os.path.isfile(marker_path):
             os.remove(marker_path)

 Integration with InferencerBase

 @attrs
 class InferencerBase(Debuggable, ABC):
     # ... existing attrs ...

     # Workspace support — optional, only used by flow inferencers
     workspace: Optional[InferencerWorkspace] = attrib(default=None)
     output_path: Optional[str] = attrib(default=None)  # relative to workspace.outputs_dir

     def resolve_output_path(self, runtime_override: Optional[str] = None) -> Optional[str]:
         """Resolve the effective output path.
  
         Priority: runtime_override > self.output_path
         Resolution: relative paths resolve to workspace.outputs_dir
         """
         path = runtime_override or self.output_path
         if path is None:
             return None
         if self.workspace and not os.path.isabs(path):
             return self.workspace.output_path(path)
         return path

 What This Centralizes

 ┌─────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
 │                         Current duplication                         │               InferencerWorkspace replacement               │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _setup_iteration_workspace (5x os.makedirs)                     │ workspace.ensure_dirs("analysis", "results")                │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI os.path.join(ws, "outputs", "round*_plan.md") (12+ occurrences) │ workspace.glob_artifacts("round*_plan.md")                  │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _write_step_completion_marker (write + mkdir)                   │ workspace.write_marker("plan")                              │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI os.path.join(ws, "outputs", ".plan_completed") checks           │ workspace.has_marker("plan")                                │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ BTA os.path.join(self.checkpoint_dir, "breakdown_result.json")      │ workspace.checkpoint_path("breakdown_result.json")          │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ BTA os.path.join(self.checkpoint_dir, "aggregator_result")          │ workspace.checkpoint_path("aggregator_result")              │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ DualInferencer os.makedirs(os.path.dirname(resolved_path))          │ workspace.ensure_dirs() (once at start)                     │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _setup_child_workflows child path construction                  │ workspace.child("planner")                                  │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ executor.py os.path.join(checkpoint_dir, "bta", ...)                │ workspace.child(f"worker_{index}")                          │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _detect_workspace_state path construction                       │ workspace.has_marker("plan"), workspace.glob_artifacts(...) │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _save_analysis_summary to results/                              │ workspace.artifact_path("analysis_summary.json")            │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI analysis/ dir + manual output_path rewriting                    │ workspace.child("analyzer") — analyzer owns its workspace   │
 └─────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘

 How Composition Works with InferencerWorkspace

 PTI composing children (planner, executor, analyzer):
 class PlanThenImplementInferencer(InferencerBase, Workflow):
     def _setup_children(self):
         # Parent creates child workspaces from its own workspace
         for name, inferencer in [
             ("planner", self.planner_inferencer),
             ("executor", self.executor_inferencer),
             ("analyzer", self.analyzer_inferencer),
         ]:
             if inferencer is not None:
                 child_ws = self.workspace.child(name)
                 child_ws.ensure_dirs()
                 inferencer.workspace = child_ws

         # PTI can predetermine where outputs will be:
         plan_path = self.workspace.child_output("planner", "plan.md")
         # → workspace_root/children/planner/outputs/plan.md

         analysis_path = self.workspace.child_output("analyzer", "analysis.md")
         # → workspace_root/children/analyzer/outputs/analysis.md

         # analysis_summary.json is PTI's own artifact (not analyzer's output):
         summary_path = self.workspace.artifact_path("analysis_summary.json")
         # → workspace_root/artifacts/analysis_summary.json

 BTA composing workers:
 class BreakdownThenAggregateInferencer(InferencerBase, WorkGraph):
     def _build_diamond_graph(self, sub_queries, ...):
         for i, sq in enumerate(sub_queries):
             worker_ws = self.workspace.child(f"worker_{i}")
             worker_ws.ensure_dirs()
             worker = self.worker_factory(sub_query=sq, index=i)
             worker.workspace = worker_ws
             # BTA reads worker output via:
             # worker.resolve_output_path()

 Aggregator prompt builder — paths captured at construction time:

 WorkGraph passes *worker_results to the agg function, not worker objects. Paths are captured via closure at graph construction time (see BTA composition example in section 5 above for full implementation).

 ---
 Summary of the Unified Contract

 ┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
 │               Concept                │                            Where                            │                                     Description                                     │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ InferencerWorkspace                  │ Standalone class                                            │ Directory layout manager + path resolver + child workspace factory                  │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace                            │ InferencerBase attr                                         │ Optional InferencerWorkspace instance                                               │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ output_path                          │ InferencerBase attr                                         │ Default output path; if relative, resolves to workspace.outputs_dir/<output_path>   │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ resolve_output_path()                │ InferencerBase method                                       │ Resolves effective output path (relative → workspace.outputs_dir, absolute → as-is) │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.outputs_dir                │ Convention                                                  │ Final deliverable output only — what the parent reads                               │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.artifacts_dir              │ Convention                                                  │ Intermediate round-by-round files, audit trail, raw outputs                         │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.checkpoints_dir            │ Convention                                                  │ Resume/checkpoint data                                                              │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.logs_dir                   │ Convention                                                  │ Prompt/response logs per round                                                      │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.child(name)                │ Method                                                      │ Creates child InferencerWorkspace under children/<name>/                            │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.child_output(name, path)   │ Method                                                      │ Resolves a child's output path (parent reads child output)                          │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Inference-time override              │ infer(..., output_path=...)                                 │ Overrides class-level output_path; only used by top-level callers                   │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Children get workspace               │ Parent sets it                                              │ child.workspace = parent.workspace.child(name)                                      │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Children use class-level output_path │ Design rule                                                 │ Parent never passes inference-time output_path override                             │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Parent reads child output            │ Via child.resolve_output_path() or workspace.child_output() │ Deterministic — same source of truth                                                │
 └──────────────────────────────────────┴─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

 ---
 Resumability Analysis

 Existing Resume Test Coverage

 The codebase has comprehensive resume tests across all three inferencers:

 DualInferencer (test_dual_inferencer_resume.py — 19 tests):
 - Crash-at-review, crash-at-fix, crash-at-second-review resume
 - Proposal text preserved across resume
 - Iteration counter correctness after resume
 - Multi-attempt with checkpoint
 - Corrupted checkpoint fallback
 - Checkpoint file validity (JSON roundtrip)

 PTI (test_pti_resume.py + test_resume_detection.py — 40+ tests):
 - Workspace state detection (plan done/impl pending, complete, followup iteration)
 - Checkpoint synthesis from workspace markers
 - Step-in-progress markers (written before, cleared after, persists on failure)
 - Resume context injection into executor prompt
 - Backward-compat with markers
 - _get_result_path stability across iteration changes

 PTI recursive resume (test_recursive_resume.py — 12 tests):
 - _setup_child_workflows propagates _result_root_override
 - Checkpoint settings propagation to children
 - Child paths isolated per iteration
 - DualInferencer child mode (_get_result_path with/without checkpoint_dir)

 BTA (test_breakdown_then_aggregate.py — TestResumability):
 - Resume after partial workers (breakdown checkpoint reload)

 Why This Design Preserves Resumability

 The resume mechanism is unchanged — Workflow's _try_load_checkpoint, _save_result, _load_result, _get_result_path all work the same way. What changes is where those paths point:

 ┌─────────────────────────────┬─────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │          Component          │                    Current path                     │                                                              New path                                                              │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer round files  │ inference_config["output_path"] (round*)            │ workspace_root/artifacts/round*_output.md                                                                                          │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer final output │ (same as last round file)                           │ workspace_root/outputs/<output_path>                                                                                               │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer checkpoints  │ checkpoint_dir/attempt_NN/step_*.json               │ workspace_root/checkpoints/attempt_NN/step_*.json                                                                                  │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI own checkpoints         │ _current_base_workspace/checkpoints/pti/step_*.json │ workspace_root/checkpoints/step_*.json                                                                                             │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI child checkpoints       │ base/checkpoints/pti/iter_N/<attr_name>/            │ workspace_root/children/<attr_name>/checkpoints/ (per-iteration via _setup_child_workflows)                                        │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI plan/impl files         │ workspace_path/outputs/round*_plan.md               │ workspace_root/children/planner/artifacts/round*_plan.md (intermediates) + workspace_root/children/planner/outputs/plan.md (final) │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI analysis output         │ workspace_path/analysis/iteration_N_analysis.md     │ workspace_root/children/analyzer/outputs/analysis.md                                                                               │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI analysis summary        │ workspace_path/results/analysis_summary.json        │ workspace_root/artifacts/analysis_summary.json                                                                                     │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA breakdown checkpoint    │ checkpoint_dir/breakdown_result.json                │ workspace_root/checkpoints/breakdown_result.json                                                                                   │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA aggregator checkpoint   │ checkpoint_dir/aggregator_result/                   │ workspace_root/checkpoints/aggregator_result/                                                                                      │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA worker outputs          │ checkpoint_dir/bta/worker_N/outputs/ (app-layer)    │ workspace_root/children/worker_N/outputs/ (worker-owned)                                                                           │
 └─────────────────────────────┴─────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 checkpoint_dir Migration — How workspace.checkpoints_dir Replaces checkpoint_dir

 Key finding: checkpoint_dir is NOT a Workflow base class attribute. The Workflow base class (in RichPythonUtils/common_objects/workflow/) uses:
 - _get_result_path(result_id) — abstract, each subclass implements
 - _resolve_result_path(result_id) — wraps _get_result_path + applies _result_root_override
 - _result_root_override — set by parent's _setup_child_workflows() for child isolation

 checkpoint_dir is defined independently on each inferencer:
 - DualInferencer: checkpoint_dir: Optional[str] = attrib(default=None, kw_only=True) (line 172)
 - BTA: checkpoint_dir: Optional[str] = attrib(default=None) (line 119)
 - PTI: uses workspace_path / _current_base_workspace (no checkpoint_dir attr at all)

 No property bridge is needed. The migration is a clean replacement:

 ┌────────────────┬────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
 │   Inferencer   │                   Current _get_result_path uses                    │                               New implementation                               │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer │ os.path.join(self.checkpoint_dir, f"attempt_{attempt:02d}", ...)   │ self.workspace.checkpoint_path(f"attempt_{attempt:02d}/step_{result_id}.json") │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ BTA            │ os.path.join(self.checkpoint_dir, f"{result_id}_result{ext}")      │ self.workspace.checkpoint_path(f"{result_id}_result{ext}")                     │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ PTI            │ os.path.join(base, "checkpoints", "pti", f"step_{result_id}.json") │ self.workspace.checkpoint_path(f"step_{result_id}.json")                       │
 └────────────────┴────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

 Other checkpoint_dir usages that change:

 # BTA breakdown checkpoint — before:
 ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
 # After:
 ckpt = self.workspace.checkpoint_path("breakdown_result.json")

 # BTA aggregator result path — before:
 _agg_ckpt = os.path.join(self.checkpoint_dir, "aggregator_result")
 # After:
 _agg_ckpt = self.workspace.checkpoint_path("aggregator_result")

 # DualInferencer checkpoint enable check — before:
 elif self.enable_checkpoint and self.checkpoint_dir:
 # After:
 elif self.enable_checkpoint and self.workspace:

 Child workflow setup — per-iteration isolation via iteration workspace:

 The Workflow base's _setup_child_workflows sets child._result_root_override to redirect where child step results are saved. PTI overrides this to isolate children per-iteration.

 Current approach manually constructs iter_N/ inside checkpoints using the stable base workspace:
 # Current — uses base workspace + manual iter_N nesting:
 child_dir = os.path.join(base, "checkpoints", "pti", f"iter_{iteration}", attr_name)
 child._result_root_override = child_dir

 New approach uses the iteration workspace instead of the base workspace. Since each iteration already has its own full workspace tree via _get_iteration_workspace(), child isolation is natural — no manual iter_N/ construction needed:

 # New — uses iteration workspace, isolation is structural:
 def _setup_child_workflows(self, state, *args, **kwargs):
     if state is None:
         return
     iteration = state.get("iteration", 1) if isinstance(state, dict) else 1

     # Key: use ITERATION workspace, not base workspace
     iter_ws_path = self._get_iteration_workspace(self.workspace.root, iteration)
     iter_workspace = InferencerWorkspace(root=iter_ws_path)

     all_children = {}
     all_children.update(self._find_child_workflows_in(self))
     all_children.update(self._find_child_workflows_in(state))

     for attr_name, (child, entry) in all_children.items():
         child.workspace = iter_workspace.child(attr_name)
         child.workspace.ensure_dirs()
         child._result_root_override = child.workspace.checkpoints_dir
         child.enable_result_save = self.enable_result_save
         child.resume_with_saved_results = self.resume_with_saved_results
         child.checkpoint_mode = self.checkpoint_mode

 Why this works — isolation is structural, not manual:

 Each iteration gets its own workspace via _get_iteration_workspace():
 - Iteration 1 → workspace_root/
 - Iteration 2 → workspace_root/followup_iterations/iteration_2/
 - Iteration 3 → workspace_root/followup_iterations/iteration_3/

 Each iteration workspace has its own children/ tree, so child checkpoints are naturally isolated:

 workspace_root/
 ├── children/                                    # iteration 1's children
 │   └── planner/checkpoints/step_*.json          # planner checkpoints for iter 1
 └── followup_iterations/
     ├── iteration_2/
     │   └── children/                            # iteration 2's children — ISOLATED
     │       └── planner/checkpoints/step_*.json  # planner checkpoints for iter 2
     └── iteration_3/
         └── children/                            # iteration 3's children — ISOLATED
             └── planner/checkpoints/step_*.json  # planner checkpoints for iter 3

 No overwriting, no iter_N/ nesting inside checkpoints. Cleaner than the current approach.

 Constructor backward compat (optional, low priority):

 If callers still pass checkpoint_dir= to DualInferencer or BTA, __attrs_post_init__ can auto-create a workspace:
 def __attrs_post_init__(self):
     if self.checkpoint_dir and self.workspace is None:
         self.workspace = InferencerWorkspace(root=self.checkpoint_dir)
     # ... rest of init ...
 This is a convenience for migration — not a permanent feature.

 Migration Strategy for Existing Checkpoints

 Approach: clean break.

 Existing on-disk checkpoints from prior runs use old paths. Old checkpoints won't resume under the new layout. This is acceptable because:
 - These are development/research workloads, not production databases
 - Checkpoint resume is a convenience feature, not a durability guarantee
 - Users can re-run from scratch with minimal cost

 Test Update Plan

 All path assertions in the following test files need updating:

 - test_dual_inferencer_resume.py: checkpoint_dir → workspace, checkpoint paths under workspace.checkpoints_dir
 - test_pti_resume.py: _get_result_path assertions, workspace state detection paths
 - test_recursive_resume.py: _result_root_override paths, child isolation paths now use child.workspace.checkpoints_dir
 - test_breakdown_then_aggregate.py: breakdown checkpoint paths under workspace.checkpoints_dir
 - test_resume_detection.py: marker detection now uses workspace.has_marker(), artifact paths change

 The test structure stays the same — only path string literals and constructor args change. The resume mechanism itself is not modified.

 ---
 Resolved Questions

 1. Where to add workspace/output_path? → InferencerWorkspace is a standalone class. workspace (optional attr) and output_path live on InferencerBase. Simple API inferencers just leave workspace=None. No mixin needed.
 2. Naming: → workspace (the InferencerWorkspace object) replaces both workspace_path (PTI) and checkpoint_dir (BTA/DualInferencer). The root path is workspace.root. PTI's workspace_path attr becomes workspace.
 3. children/ vs current naming: → Clean break. PTI's checkpoints/pti/iter_N/<attr_name>/ → children/<attr_name>/checkpoints/. Old checkpoints won't resume. (See Migration Strategy above.)
 4. DualInferencer's round_index template: → Resolved by outputs/ vs artifacts/ split. {{ round_index }} applies to artifacts/ filenames. Class-level output_path is template-free, points to final output in outputs/.

 Remaining Open Questions

 1. DualInferencer children/: DualInferencer has base_inferencer, review_inferencer, fixer_inferencer. These are typically stateless API callers that don't need workspaces. However, they could be flow inferencers. Recommendation: DualInferencer sets child.workspace =
 self.workspace.child(name) only if the child already has a non-None workspace or output_path, or if it's an instance of a flow inferencer type. Simple API inferencers are left alone.

 ~~2. **PTI iteration isolation:**~~ → Resolved. Each iteration already gets its own full workspace via _get_iteration_workspace() (iteration 1 → workspace_root/, iteration N → followup_iterations/iteration_N/). Each iteration workspace has its own children/ tree, so child
 checkpoints are naturally isolated without manual iter_N/ nesting. PTI's _setup_child_workflows now uses the iteration workspace to set child workspaces. See "Child workflow setup" in the checkpoint_dir migration section for the full implementation.
 Design Discussion: Unified Workspace & Output Path Contract for Inferencers

 Context

 The three composable flow inferencers (DualInferencer, PTI, BTA) each handle workspace directories and output paths differently. This creates inconsistency and makes composition fragile. The user proposes a unified design. Below is my analysis of each design point.

 ---
 Point-by-Point Analysis

 (1) BTA should NOT own per-worker output paths — only the workspace root

 I agree. This is already the correct separation of concerns in the current code.

 Current state:
 - BTA has checkpoint_dir (line 119 of breakdown_then_aggregate_inferencer.py) which serves as its workspace root
 - BTA passes workers to worker_factory(sub_query, index) — the factory decides internal layout
 - In executor.py:463, the application layer creates checkpoint_dir/bta/worker_{index}/outputs/facet_{index}.md
 - BTA itself never references these paths

 Why this is right: A worker could be a PTI, a DualInferencer, a simple API inferencer, or anything else. Each has its own workspace conventions. BTA should only say "here's your workspace root" and let the worker organize beneath it. The worker's internal directory layout is its
  own business.

 What needs to change: BTA should formalize passing a workspace_root (not just relying on the application layer to construct paths from checkpoint_dir). The worker_factory signature could receive the workspace root for that worker, e.g. worker_factory(sub_query, index,
 workspace_root=<bta_workspace>/workers/worker_{index}).

 ---
 (2) Align PTI workspace structure/naming with BTA

 Current PTI workspace:
 workspace_path/
 ├── outputs/          # plan + implementation files
 ├── analysis/
 ├── results/
 ├── checkpoints/pti/  # Workflow checkpoints + child DualInferencer checkpoints
 ├── logs/
 ├── _runtime/
 └── followup_iterations/iteration_N/  # same structure recursively

 Current BTA workspace (as constructed by executor.py):
 checkpoint_dir/
 ├── breakdown_result.json        # BTA's own checkpoint
 ├── aggregator_result/           # aggregator checkpoint
 └── bta/worker_N/outputs/        # worker outputs (app-layer convention)

 Proposed aligned structure:

 The key insight: both PTI and BTA are "parent inferencers" that manage child inferencers. They should follow the same workspace convention. The workspace separates three concerns:

 - outputs/ — final output only. The "answer." One file (or a small set). This is what the parent reads.
 - artifacts/ — intermediate round-by-round files, audit trail, raw outputs. Useful for debugging and transparency, but not the deliverable.
 - checkpoints/ — resume/checkpoint data for crash recovery.

 workspace_root/                    # the inferencer's workspace root
 ├── outputs/                       # FINAL output only (the deliverable)
 ├── artifacts/                     # intermediate outputs, audit trail (old "outputs/" behavior)
 ├── checkpoints/                   # resume/checkpoint data
 │   └── ...                        # Workflow/WorkGraph checkpoint files
 └── children/                      # child inferencer workspaces
     ├── <child_name>/              # each child gets its own workspace_root
     │   ├── outputs/               # child's final output
     │   ├── artifacts/             # child's intermediate outputs
     │   ├── checkpoints/
     │   └── children/              # recursive if the child is also a composing inferencer
     └── ...

 The class-level output_path, when relative, resolves relative to outputs/ (not workspace_root directly). This means output_path="plan.md" → workspace_root/outputs/plan.md.

 For DualInferencer specifically (with output_path="plan.md"):
 workspace_root/
 ├── outputs/
 │   └── plan.md                    # FINAL consensus result (output_path = "plan.md")
 ├── artifacts/
 │   ├── round01_plan.md            # round 1: initial proposal (base_inferencer)
 │   ├── round02_plan.md            # round 2: after first fix cycle (fixer_inferencer)
 │   └── round03_plan.md            # round 3: after second fix cycle → consensus reached
 ├── logs/
 │   ├── Round01/                   # per-round prompts/responses (from Debuggable logging)
 │   │   ├── InitialPrompt.md
 │   │   └── InitialResponse.md
 │   ├── Round02/
 │   │   ├── ReviewPrompt.md
 │   │   ├── ReviewResponse.md
 │   │   ├── FollowupPrompt.md
 │   │   └── FollowupResponse.md
 │   └── Round03/
 │       └── ...
 └── checkpoints/
     └── attempt_01/step_*.json     # Workflow step checkpoints

 DualInferencer Artifact Naming Convention

 Artifact filenames are derived from output_path with a round index prefix:

 round{NN}_{output_path_basename}

 - NN = zero-padded round number (01, 02, 03, ...), using total_iterations counter
 - output_path_basename = the basename of the class-level output_path (e.g., plan.md)

 How rounds map to the consensus loop:

 ┌───────┬───────────────────────────────────────────────┬───────────────────────────┐
 │ Round │                 What happens                  │     Artifact written      │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 01    │ base_inferencer proposes                      │ artifacts/round01_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 02    │ fixer_inferencer addresses review issues      │ artifacts/round02_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ 03    │ fixer_inferencer addresses second review      │ artifacts/round03_plan.md │
 ├───────┼───────────────────────────────────────────────┼───────────────────────────┤
 │ ...   │ (continues until consensus or max_iterations) │ ...                       │
 └───────┴───────────────────────────────────────────────┴───────────────────────────┘

 After consensus or loop exhaustion, _finalize_response() copies the last round's artifact to outputs/{output_path}:
 def _finalize_response(self):
     # Last round artifact → final output
     last_round = self._state["total_iterations"]
     basename = os.path.basename(self.output_path)  # e.g., "plan.md"
     src = self.workspace.artifact_path(f"round{last_round:02d}_{basename}")
     dst = self.workspace.output_path(basename)
     if os.path.isfile(src):
         shutil.copy2(src, dst)

 Where _maybe_replace_with_file_reference writes (before → after):
 # Before: writes to inference_config["output_path"] with {{ round_index }}
 resolved_path = output_path_template.replace("{{ round_index }}", str(round_index))

 # After: writes to workspace artifacts with round-indexed basename
 basename = os.path.basename(self.output_path)  # "plan.md"
 resolved_path = self.workspace.artifact_path(f"round{total_iterations:02d}_{basename}")

 For PTI specifically — single iteration (output_mode=IMPLEMENTATION):
 workspace_root/
 ├── outputs/
 │   └── implementation.md          # ← copied from children/executor/outputs/ by _finalize_outputs()
 ├── artifacts/
 │   ├── .plan_completed            # phase completion markers
 │   ├── .impl_completed
 │   ├── request.txt                # original request text
 │   └── analysis_summary.json      # iteration control flow decision
 ├── checkpoints/                   # __wf_checkpoint__.json, step_*.json
 └── children/
     ├── planner/
     │   ├── outputs/
     │   │   └── plan.md
     │   ├── artifacts/
     │   │   ├── round1_plan.md
     │   │   └── round2_plan.md
     │   └── checkpoints/
     ├── executor/
     │   ├── outputs/
     │   │   └── implementation.md
     │   ├── artifacts/
     │   └── checkpoints/
     └── analyzer/
         ├── outputs/
         │   └── analysis.md
         ├── artifacts/
         └── checkpoints/

 For PTI — multi-iteration (3 iterations, output_mode=IMPLEMENTATION):
 workspace_root/
 ├── outputs/
 │   └── implementation.md          # ← copied from LAST iteration (iteration_3) by _finalize_outputs()
 ├── artifacts/
 │   ├── request.txt
 │   ├── .plan_completed
 │   ├── .impl_completed
 │   └── analysis_summary.json
 ├── checkpoints/                   # PTI's own workflow checkpoints (stable across iterations)
 ├── children/                      # iteration 1's children
 │   ├── planner/...
 │   ├── executor/...
 │   └── analyzer/...
 └── followup_iterations/
     ├── iteration_2/               # full workspace for iteration 2
     │   ├── artifacts/
     │   │   ├── request.txt        # iteration handoff text
     │   │   └── analysis_summary.json
     │   ├── checkpoints/
     │   └── children/
     │       ├── planner/...
     │       ├── executor/...
     │       └── analyzer/...
     └── iteration_3/               # last iteration — source of final output
         ├── artifacts/
         │   └── ...
         ├── checkpoints/
         └── children/
             ├── planner/outputs/plan.md
             ├── executor/outputs/implementation.md   # ← _finalize_outputs() copies FROM here
             └── analyzer/outputs/analysis.md

 For PTI with multiple output mode (output_mode=PLAN | IMPLEMENTATION):
 workspace_root/
 ├── outputs/                       # multiple deliverables, all from last iteration
 │   ├── plan.md                    # ← from last iteration's planner
 │   └── implementation.md          # ← from last iteration's executor
 ├── ...

 Key insight: Regardless of how many iterations ran, workspace_root/outputs/ always contains the final deliverables. _finalize_outputs() handles the copy from the last iteration's children. The parent (or user) only needs to look at workspace_root/outputs/ — never needs to
 traverse followup_iterations/.

 PTI Output Mode

 PTI's final output is ambiguous — it could be the plan, implementation, analysis, or a combination. A class-level output_mode enum determines which child outputs PTI surfaces as its own final output.

 from enum import Flag, auto

 class PTIOutputMode(Flag):
     """Which child outputs PTI surfaces as its own final output."""
     PLAN = auto()
     IMPLEMENTATION = auto()
     ANALYSIS = auto()

     # Common combinations
     PLAN_AND_IMPLEMENTATION = PLAN | IMPLEMENTATION
     ALL = PLAN | IMPLEMENTATION | ANALYSIS

 On PlanThenImplementInferencer:
 output_mode: PTIOutputMode = attrib(default=PTIOutputMode.IMPLEMENTATION)

 Behavior by cardinality:

 ┌───────────────────────────────┬───────────────────────────────────────────────────────────────────┬──────────────────────────────────────────┐
 │          output_mode          │                         outputs/ content                          │      resolve_output_path() returns       │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ PLAN (single)                 │ outputs/plan.md                                                   │ workspace_root/outputs/plan.md           │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ IMPLEMENTATION (single)       │ outputs/implementation.md                                         │ workspace_root/outputs/implementation.md │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ ANALYSIS (single)             │ outputs/analysis.md                                               │ workspace_root/outputs/analysis.md       │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ PLAN | IMPLEMENTATION (multi) │ outputs/plan.md + outputs/implementation.md                       │ workspace_root/outputs/ (the folder)     │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
 │ ALL (multi)                   │ outputs/plan.md + outputs/implementation.md + outputs/analysis.md │ workspace_root/outputs/ (the folder)     │
 └───────────────────────────────┴───────────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘

 Flag-to-child mapping (single source of truth):
 _OUTPUT_MODE_MAP = {
     PTIOutputMode.PLAN: ("planner", "plan.md"),
     PTIOutputMode.IMPLEMENTATION: ("executor", "implementation.md"),
     PTIOutputMode.ANALYSIS: ("analyzer", "analysis.md"),
 }

 Finalization — multi-iteration aware, driven by the map:

 _finalize_outputs() always copies from the last iteration's children to the workspace root's outputs/. This is called at the end of _ainfer(), after the workflow completes.

 def _finalize_outputs(self):
     """Copy selected child outputs from the LAST iteration to workspace root outputs/.
  
     Regardless of how many iterations ran, the workspace root's outputs/
     always contains the final deliverables. The parent or user only needs
     to look at workspace_root/outputs/.
     """
     last_iter = self._state.get("iteration", 1)

     # Get the workspace for the last iteration
     iter_ws_path = self._get_iteration_workspace(self.workspace.root, last_iter)
     iter_workspace = InferencerWorkspace(root=iter_ws_path)

     os.makedirs(self.workspace.outputs_dir, exist_ok=True)

     for flag, (child_name, filename) in _OUTPUT_MODE_MAP.items():
         if flag in self.output_mode:
             src = iter_workspace.child_output(child_name, filename)
             dst = self.workspace.output_path(filename)
             if os.path.isfile(src):
                 shutil.copy2(src, dst)

 def resolve_output_path(self, runtime_override=None):
     if runtime_override:
         return super().resolve_output_path(runtime_override)
     # Count active flags (exclude compound flags like ALL)
     active = [f for f in _OUTPUT_MODE_MAP if f in self.output_mode]
     if len(active) == 1:
         _, filename = _OUTPUT_MODE_MAP[active[0]]
         return self.workspace.output_path(filename)
     else:
         # Multiple outputs → return the outputs/ folder
         return self.workspace.outputs_dir

 Where _finalize_outputs() is called:
 async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
     # ... setup, resume detection ...
     await Workflow._arun(self, inference_input, **_inference_args)

     # Copy final outputs from last iteration to workspace root
     if self.workspace:
         self._finalize_outputs()

     return self._build_response_from_state(self._state)

 Resume safety: _finalize_outputs() is idempotent — it just copies files via shutil.copy2. If PTI crashes after the last iteration completes but before _finalize_outputs() runs, resume will detect all steps completed (via checkpoints), skip re-execution, and _finalize_outputs()
 runs again safely producing the same result.

 Multi-iteration workspace resolution reuses the existing _get_iteration_workspace:
 - Iteration 1 → workspace_root/ (children are at workspace_root/children/)
 - Iteration N>1 → workspace_root/followup_iterations/iteration_N/ (children are at that path's children/)
 - _finalize_outputs() resolves the last iteration's workspace, finds the children there, copies to workspace_root/outputs/

 What changed vs. old PTI layout:
 - results/ eliminated → analysis_summary.json moves to artifacts/ (it's PTI's own control-flow artifact, not a deliverable)
 - analysis/ eliminated → analyzer gets its own workspace under children/analyzer/ (consistent with planner/executor treatment)
 - Every child inferencer now lives under children/ — no special-case directories
 - output_mode enum declaratively controls which child outputs become PTI's deliverables

 Comparison — how each inferencer determines its final output:

 ┌────────────────┬─────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
 │   Inferencer   │                 Final output is...                  │              Determined by...               │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ DualInferencer │ Last consensus round result                         │ Always — inherent to consensus loop         │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ BTA            │ Aggregator output (or last worker if no aggregator) │ Always — inherent to diamond graph          │
 ├────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ PTI            │ Plan, implementation, analysis, or combination      │ output_mode: PTIOutputMode class-level enum │
 └────────────────┴─────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

 For BTA specifically:
 workspace_root/
 ├── outputs/
 │   └── result.md                  # aggregated final output (output_path = "result.md")
 ├── artifacts/
 │   └── aggregator_raw_output.md   # raw aggregator response (audit trail)
 ├── checkpoints/                   # breakdown_result.json, aggregator_result/
 └── children/
     ├── worker_0/                  # each worker's workspace_root
     │   ├── outputs/               # worker's final output
     │   ├── artifacts/             # worker's intermediate outputs
     │   └── ...                    # could be PTI, DualInferencer, etc.
     ├── worker_1/
     └── ...

 Key naming alignment:
 - workspace_root (not workspace_path or checkpoint_dir) as the universal parameter name
 - outputs/ — final deliverable output only; class-level output_path resolves relative to this
 - artifacts/ — intermediate/round-by-round files (audit trail, debugging)
 - checkpoints/ — resume/checkpoint data
 - children/ — child inferencer workspace roots

 ---
 (3) DualInferencer should have its own workspace root

 I agree. Currently DualInferencer has:
 - checkpoint_dir — used for Workflow step checkpoints (attempt_NN/step_*.json)
 - output_path — passed purely via inference_config["output_path"] at inference time, not a class-level attribute

 DualInferencer has no class-level output path and no workspace root. Its output path is always injected at inference time through inference_config["output_path"] (see _maybe_replace_with_file_reference at line 1027). When PTI composes DualInferencer, PTI rewrites this
 inference_config output_path in _build_iteration_config (line 582-584).

 With the new design: DualInferencer gets workspace, and its layout becomes:
 workspace_root/
 ├── outputs/                       # FINAL consensus output only
 │   └── output.md                  # (output_path = "output.md")
 ├── artifacts/                     # round-indexed intermediate files
 │   ├── round01_output.md          # initial proposal
 │   ├── round02_output.md          # after fix cycle 1
 │   └── round03_output.md          # after fix cycle 2 (consensus)
 ├── logs/                          # per-round prompts/responses
 │   ├── Round01/...
 │   └── Round02/...
 └── checkpoints/                   # attempt_NN/step_*.json

 DualInferencer's _maybe_replace_with_file_reference currently writes each round to inference_config["output_path"] with {{ round_index }} substitution. Under the new design, those per-round files go to artifacts/round{NN}_{output_path}, and _finalize_response copies the final
 consensus result to outputs/{output_path}. See "DualInferencer Artifact Naming Convention" above for details.

 ---
 (4) Unified workspace + class-level output_path + inference-time override

 I agree with this layered approach. Formalized via InferencerWorkspace:

 On InferencerBase:
 workspace: Optional[InferencerWorkspace] = attrib(default=None)
 output_path: Optional[str] = attrib(default=None)  # relative to workspace.outputs_dir

 Resolution rules:
 1. If output_path is relative and workspace is set, resolve to workspace.outputs_dir / output_path
 2. If output_path is absolute, use as-is
 3. Inference-time output_path (via infer(..., output_path=...)) overrides class-level
 4. Same resolution rules apply to inference-time override

 On InferencerBase:
 def resolve_output_path(self, runtime_override: Optional[str] = None) -> Optional[str]:
     path = runtime_override or self.output_path
     if path is None:
         return None
     if self.workspace and not os.path.isabs(path):
         return self.workspace.output_path(path)
     return path

 Simple API inferencers (ClaudeApiInferencer, etc.) leave workspace=None — no overhead.

 ---
 (5) Composing inferencers use class-level relative output_path only — no inference-time override for children

 I agree — this is the critical design rule for predictable composition.

 The rule: When a parent (BTA/PTI) composes child inferencers, it:
 1. Sets child.workspace = parent.workspace.child("<child_name>")
 2. Relies on the child's class-level output_path (relative to child's workspace.outputs_dir)
 3. Never passes inference-time output_path override to the child

 This means:
 - The parent can predetermine where the child's output will be via child.resolve_output_path()
 - Or equivalently via parent.workspace.child_output("child_name", "output.md")
 - No duplicate path construction in two places (the executor.py problem)

 Example — PTI composing DualInferencer:
 # PTI sets up planner — one line
 self.planner_inferencer.workspace = self.workspace.child("planner")
 self.planner_inferencer.workspace.ensure_dirs()
 # planner.output_path = "plan.md" (set at construction, class-level)

 # PTI reads the plan after inference:
 plan_path = self.planner_inferencer.resolve_output_path()
 # → workspace_root/children/planner/outputs/plan.md

 # Or without touching the child object:
 plan_path = self.workspace.child_output("planner", "plan.md")

 # Intermediate rounds are in the child's artifacts:
 # workspace_root/children/planner/artifacts/round*_plan.md

 Example — BTA composing workers:
 def _build_diamond_graph(self, sub_queries, ...):
     for i, sq in enumerate(sub_queries):
         worker = self.worker_factory(sub_query=sq, index=i)
         worker.workspace = self.workspace.child(f"worker_{i}")
         worker.workspace.ensure_dirs()
         # worker.output_path is whatever the worker class defines
         # BTA reads from: worker.resolve_output_path() after inference

 Aggregator prompt builder — single source of truth via closure-captured paths:

 Note: WorkGraph passes *worker_results (varargs) to the aggregation function — NOT worker objects. So worker output paths must be captured at graph construction time via closure.

 def _build_diamond_graph(self, sub_queries, ...):
     worker_output_paths = []
     for i, sq in enumerate(sub_queries):
         worker = self.worker_factory(sub_query=sq, index=i)
         worker.workspace = self.workspace.child(f"worker_{i}")
         worker.workspace.ensure_dirs()
         # Capture resolved path ONCE at construction time
         worker_output_paths.append(worker.resolve_output_path())
         # ... create WorkGraphNode ...

     # Aggregator closure captures worker_output_paths — same paths workers write to
     def _agg_with_paths(worker_results, original_query):
         parts = []
         for idx, path in enumerate(worker_output_paths):
             parts.append(f"### Facet {idx+1}\nRead from: `{path}`")
         return "\n\n".join(parts)

 Current gap this fixes: In executor.py, the same path checkpoint_dir/bta/worker_{idx}/outputs/facet_{idx}.md is constructed independently in worker_factory (line 463) and agg_prompt_builder (line 504). With InferencerWorkspace, the path is resolved once at graph construction
 time and shared via closure — single source of truth.

 ---
 InferencerWorkspace Class

 Motivation

 All three inferencers duplicate the same workspace operations: os.path.join for path construction, os.makedirs for directory creation, glob.glob for artifact scanning, and marker file management. An InferencerWorkspace class formalizes the directory layout, centralizes these
 operations, and makes composition clean.

 Design Principles

 - Path management + directory layout, not file I/O. Inferencers read/write their own files. The workspace resolves paths and manages directories.
 - Marker files are the exception — markers (.plan_completed, etc.) are a workspace concern (state tracking), not an inferencer output concern, so the workspace owns them.
 - Recursive composition via child() — workspace.child("planner") returns a new InferencerWorkspace rooted at children/planner/. This is how parent inferencers set up child workspaces.
 - subdir() for inferencer-specific directories — PTI can create analysis/, results/ without polluting the base class.

 Class Definition

 @attrs
 class InferencerWorkspace:
     """Standard directory layout manager for inferencer workspaces.
  
     Provides path resolution, directory creation, child workspace management,
     and artifact scanning. Does NOT own file I/O for outputs/artifacts —
     inferencers handle that themselves.
  
     Standard layout:
         root/
         ├── outputs/       # final deliverable output
         ├── artifacts/     # intermediate round-by-round files, audit trail
         ├── checkpoints/   # resume/checkpoint data
         ├── logs/          # prompt/response logs
         ├── children/      # child inferencer workspace roots
         └── (custom)/      # inferencer-specific dirs via subdir()
     """
     root: str = attrib()  # absolute path to workspace root

     # ── Standard directory properties ──

     @property
     def outputs_dir(self) -> str:
         """Final output directory. output_path resolves relative to this."""
         return os.path.join(self.root, "outputs")

     @property
     def artifacts_dir(self) -> str:
         """Intermediate round-by-round files, audit trail."""
         return os.path.join(self.root, "artifacts")

     @property
     def checkpoints_dir(self) -> str:
         """Resume/checkpoint data."""
         return os.path.join(self.root, "checkpoints")

     @property
     def logs_dir(self) -> str:
         """Logging output (prompts, responses per round)."""
         return os.path.join(self.root, "logs")

     @property
     def children_dir(self) -> str:
         """Child inferencer workspace roots."""
         return os.path.join(self.root, "children")

     # ── Directory initialization ──

     def ensure_dirs(self, *extra_subdirs: str) -> None:
         """Create standard directories + optional extras.
  
         Example:
             workspace.ensure_dirs()  # standard dirs only; no PTI-specific extras needed
         """
         for d in (self.outputs_dir, self.artifacts_dir,
                   self.checkpoints_dir, self.logs_dir):
             os.makedirs(d, exist_ok=True)
         for sub in extra_subdirs:
             os.makedirs(os.path.join(self.root, sub), exist_ok=True)

     # ── Path resolution ──

     def output_path(self, relative: str) -> str:
         """Resolve a path relative to outputs/."""
         return os.path.join(self.outputs_dir, relative)

     def artifact_path(self, relative: str) -> str:
         """Resolve a path relative to artifacts/."""
         return os.path.join(self.artifacts_dir, relative)

     def checkpoint_path(self, relative: str) -> str:
         """Resolve a path relative to checkpoints/."""
         return os.path.join(self.checkpoints_dir, relative)

     def log_path(self, relative: str) -> str:
         """Resolve a path relative to logs/."""
         return os.path.join(self.logs_dir, relative)

     def subdir(self, name: str) -> str:
         """Access a custom subdirectory (e.g., 'analysis', 'results').
         Does NOT create it — call ensure_dirs() or os.makedirs() separately.
         """
         return os.path.join(self.root, name)

     # ── Resolve output_path (relative or absolute) ──

     def resolve(self, path: str) -> str:
         """If path is relative, resolve to outputs/. If absolute, return as-is."""
         if os.path.isabs(path):
             return path
         return self.output_path(path)

     # ── Child workspace management ──

     def child(self, name: str) -> 'InferencerWorkspace':
         """Create a child workspace under children/<name>/.
  
         Returns a new InferencerWorkspace instance. Does not create
         directories on disk — call child.ensure_dirs() when ready.
         """
         return InferencerWorkspace(root=os.path.join(self.children_dir, name))

     def child_output(self, child_name: str, output_path: str) -> str:
         """Resolve a child's output path without creating the child workspace object.
  
         Equivalent to: self.child(child_name).output_path(output_path)
         Useful when the parent just needs to read a child's known output.
         """
         return os.path.join(self.children_dir, child_name, "outputs", output_path)

     # ── Artifact scanning (replaces scattered glob patterns) ──

     def glob_artifacts(self, pattern: str) -> List[str]:
         """Glob within artifacts/ dir. Returns sorted list.
  
         Example:
             workspace.glob_artifacts("round*_plan.md")
         """
         return sorted(glob.glob(os.path.join(self.artifacts_dir, pattern)))

     def glob_outputs(self, pattern: str) -> List[str]:
         """Glob within outputs/ dir. Returns sorted list."""
         return sorted(glob.glob(os.path.join(self.outputs_dir, pattern)))

     # ── Marker files (replaces PTI's _write_step_completion_marker) ──

     def write_marker(self, name: str, metadata: dict = None) -> None:
         """Write a completion marker to artifacts/.<name>_completed.
  
         Example:
             workspace.write_marker("plan")  # writes artifacts/.plan_completed
         """
         marker_path = self.artifact_path(f".{name}_completed")
         os.makedirs(os.path.dirname(marker_path), exist_ok=True)
         with open(marker_path, "w") as f:
             json.dump(metadata or {
                 "completed_at": datetime.now(timezone.utc).isoformat(),
                 "step": name,
             }, f)

     def has_marker(self, name: str) -> bool:
         """Check if a completion marker exists."""
         return os.path.isfile(self.artifact_path(f".{name}_completed"))

     def clear_marker(self, name: str) -> None:
         """Remove a completion marker (e.g., on re-attempt)."""
         marker_path = self.artifact_path(f".{name}_completed")
         if os.path.isfile(marker_path):
             os.remove(marker_path)

 Integration with InferencerBase

 @attrs
 class InferencerBase(Debuggable, ABC):
     # ... existing attrs ...

     # Workspace support — optional, only used by flow inferencers
     workspace: Optional[InferencerWorkspace] = attrib(default=None)
     output_path: Optional[str] = attrib(default=None)  # relative to workspace.outputs_dir

     def resolve_output_path(self, runtime_override: Optional[str] = None) -> Optional[str]:
         """Resolve the effective output path.
  
         Priority: runtime_override > self.output_path
         Resolution: relative paths resolve to workspace.outputs_dir
         """
         path = runtime_override or self.output_path
         if path is None:
             return None
         if self.workspace and not os.path.isabs(path):
             return self.workspace.output_path(path)
         return path

 What This Centralizes

 ┌─────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
 │                         Current duplication                         │               InferencerWorkspace replacement               │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _setup_iteration_workspace (5x os.makedirs)                     │ workspace.ensure_dirs("analysis", "results")                │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI os.path.join(ws, "outputs", "round*_plan.md") (12+ occurrences) │ workspace.glob_artifacts("round*_plan.md")                  │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _write_step_completion_marker (write + mkdir)                   │ workspace.write_marker("plan")                              │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI os.path.join(ws, "outputs", ".plan_completed") checks           │ workspace.has_marker("plan")                                │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ BTA os.path.join(self.checkpoint_dir, "breakdown_result.json")      │ workspace.checkpoint_path("breakdown_result.json")          │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ BTA os.path.join(self.checkpoint_dir, "aggregator_result")          │ workspace.checkpoint_path("aggregator_result")              │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ DualInferencer os.makedirs(os.path.dirname(resolved_path))          │ workspace.ensure_dirs() (once at start)                     │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _setup_child_workflows child path construction                  │ workspace.child("planner")                                  │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ executor.py os.path.join(checkpoint_dir, "bta", ...)                │ workspace.child(f"worker_{index}")                          │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _detect_workspace_state path construction                       │ workspace.has_marker("plan"), workspace.glob_artifacts(...) │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI _save_analysis_summary to results/                              │ workspace.artifact_path("analysis_summary.json")            │
 ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ PTI analysis/ dir + manual output_path rewriting                    │ workspace.child("analyzer") — analyzer owns its workspace   │
 └─────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘

 How Composition Works with InferencerWorkspace

 PTI composing children (planner, executor, analyzer):
 class PlanThenImplementInferencer(InferencerBase, Workflow):
     def _setup_children(self):
         # Parent creates child workspaces from its own workspace
         for name, inferencer in [
             ("planner", self.planner_inferencer),
             ("executor", self.executor_inferencer),
             ("analyzer", self.analyzer_inferencer),
         ]:
             if inferencer is not None:
                 child_ws = self.workspace.child(name)
                 child_ws.ensure_dirs()
                 inferencer.workspace = child_ws

         # PTI can predetermine where outputs will be:
         plan_path = self.workspace.child_output("planner", "plan.md")
         # → workspace_root/children/planner/outputs/plan.md

         analysis_path = self.workspace.child_output("analyzer", "analysis.md")
         # → workspace_root/children/analyzer/outputs/analysis.md

         # analysis_summary.json is PTI's own artifact (not analyzer's output):
         summary_path = self.workspace.artifact_path("analysis_summary.json")
         # → workspace_root/artifacts/analysis_summary.json

 BTA composing workers:
 class BreakdownThenAggregateInferencer(InferencerBase, WorkGraph):
     def _build_diamond_graph(self, sub_queries, ...):
         for i, sq in enumerate(sub_queries):
             worker_ws = self.workspace.child(f"worker_{i}")
             worker_ws.ensure_dirs()
             worker = self.worker_factory(sub_query=sq, index=i)
             worker.workspace = worker_ws
             # BTA reads worker output via:
             # worker.resolve_output_path()

 Aggregator prompt builder — paths captured at construction time:

 WorkGraph passes *worker_results to the agg function, not worker objects. Paths are captured via closure at graph construction time (see BTA composition example in section 5 above for full implementation).

 ---
 Summary of the Unified Contract

 ┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
 │               Concept                │                            Where                            │                                     Description                                     │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ InferencerWorkspace                  │ Standalone class                                            │ Directory layout manager + path resolver + child workspace factory                  │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace                            │ InferencerBase attr                                         │ Optional InferencerWorkspace instance                                               │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ output_path                          │ InferencerBase attr                                         │ Default output path; if relative, resolves to workspace.outputs_dir/<output_path>   │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ resolve_output_path()                │ InferencerBase method                                       │ Resolves effective output path (relative → workspace.outputs_dir, absolute → as-is) │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.outputs_dir                │ Convention                                                  │ Final deliverable output only — what the parent reads                               │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.artifacts_dir              │ Convention                                                  │ Intermediate round-by-round files, audit trail, raw outputs                         │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.checkpoints_dir            │ Convention                                                  │ Resume/checkpoint data                                                              │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.logs_dir                   │ Convention                                                  │ Prompt/response logs per round                                                      │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.child(name)                │ Method                                                      │ Creates child InferencerWorkspace under children/<name>/                            │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ workspace.child_output(name, path)   │ Method                                                      │ Resolves a child's output path (parent reads child output)                          │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Inference-time override              │ infer(..., output_path=...)                                 │ Overrides class-level output_path; only used by top-level callers                   │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Children get workspace               │ Parent sets it                                              │ child.workspace = parent.workspace.child(name)                                      │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Children use class-level output_path │ Design rule                                                 │ Parent never passes inference-time output_path override                             │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ Parent reads child output            │ Via child.resolve_output_path() or workspace.child_output() │ Deterministic — same source of truth                                                │
 └──────────────────────────────────────┴─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

 ---
 Resumability Analysis

 Existing Resume Test Coverage

 The codebase has comprehensive resume tests across all three inferencers:

 DualInferencer (test_dual_inferencer_resume.py — 19 tests):
 - Crash-at-review, crash-at-fix, crash-at-second-review resume
 - Proposal text preserved across resume
 - Iteration counter correctness after resume
 - Multi-attempt with checkpoint
 - Corrupted checkpoint fallback
 - Checkpoint file validity (JSON roundtrip)

 PTI (test_pti_resume.py + test_resume_detection.py — 40+ tests):
 - Workspace state detection (plan done/impl pending, complete, followup iteration)
 - Checkpoint synthesis from workspace markers
 - Step-in-progress markers (written before, cleared after, persists on failure)
 - Resume context injection into executor prompt
 - Backward-compat with markers
 - _get_result_path stability across iteration changes

 PTI recursive resume (test_recursive_resume.py — 12 tests):
 - _setup_child_workflows propagates _result_root_override
 - Checkpoint settings propagation to children
 - Child paths isolated per iteration
 - DualInferencer child mode (_get_result_path with/without checkpoint_dir)

 BTA (test_breakdown_then_aggregate.py — TestResumability):
 - Resume after partial workers (breakdown checkpoint reload)

 Why This Design Preserves Resumability

 The resume mechanism is unchanged — Workflow's _try_load_checkpoint, _save_result, _load_result, _get_result_path all work the same way. What changes is where those paths point:

 ┌─────────────────────────────┬─────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │          Component          │                    Current path                     │                                                              New path                                                              │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer round files  │ inference_config["output_path"] (round*)            │ workspace_root/artifacts/round*_output.md                                                                                          │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer final output │ (same as last round file)                           │ workspace_root/outputs/<output_path>                                                                                               │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer checkpoints  │ checkpoint_dir/attempt_NN/step_*.json               │ workspace_root/checkpoints/attempt_NN/step_*.json                                                                                  │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI own checkpoints         │ _current_base_workspace/checkpoints/pti/step_*.json │ workspace_root/checkpoints/step_*.json                                                                                             │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI child checkpoints       │ base/checkpoints/pti/iter_N/<attr_name>/            │ workspace_root/children/<attr_name>/checkpoints/ (per-iteration via _setup_child_workflows)                                        │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI plan/impl files         │ workspace_path/outputs/round*_plan.md               │ workspace_root/children/planner/artifacts/round*_plan.md (intermediates) + workspace_root/children/planner/outputs/plan.md (final) │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI analysis output         │ workspace_path/analysis/iteration_N_analysis.md     │ workspace_root/children/analyzer/outputs/analysis.md                                                                               │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PTI analysis summary        │ workspace_path/results/analysis_summary.json        │ workspace_root/artifacts/analysis_summary.json                                                                                     │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA breakdown checkpoint    │ checkpoint_dir/breakdown_result.json                │ workspace_root/checkpoints/breakdown_result.json                                                                                   │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA aggregator checkpoint   │ checkpoint_dir/aggregator_result/                   │ workspace_root/checkpoints/aggregator_result/                                                                                      │
 ├─────────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ BTA worker outputs          │ checkpoint_dir/bta/worker_N/outputs/ (app-layer)    │ workspace_root/children/worker_N/outputs/ (worker-owned)                                                                           │
 └─────────────────────────────┴─────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 checkpoint_dir Migration — How workspace.checkpoints_dir Replaces checkpoint_dir

 Key finding: checkpoint_dir is NOT a Workflow base class attribute. The Workflow base class (in RichPythonUtils/common_objects/workflow/) uses:
 - _get_result_path(result_id) — abstract, each subclass implements
 - _resolve_result_path(result_id) — wraps _get_result_path + applies _result_root_override
 - _result_root_override — set by parent's _setup_child_workflows() for child isolation

 checkpoint_dir is defined independently on each inferencer:
 - DualInferencer: checkpoint_dir: Optional[str] = attrib(default=None, kw_only=True) (line 172)
 - BTA: checkpoint_dir: Optional[str] = attrib(default=None) (line 119)
 - PTI: uses workspace_path / _current_base_workspace (no checkpoint_dir attr at all)

 No property bridge is needed. The migration is a clean replacement:

 ┌────────────────┬────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
 │   Inferencer   │                   Current _get_result_path uses                    │                               New implementation                               │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ DualInferencer │ os.path.join(self.checkpoint_dir, f"attempt_{attempt:02d}", ...)   │ self.workspace.checkpoint_path(f"attempt_{attempt:02d}/step_{result_id}.json") │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ BTA            │ os.path.join(self.checkpoint_dir, f"{result_id}_result{ext}")      │ self.workspace.checkpoint_path(f"{result_id}_result{ext}")                     │
 ├────────────────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ PTI            │ os.path.join(base, "checkpoints", "pti", f"step_{result_id}.json") │ self.workspace.checkpoint_path(f"step_{result_id}.json")                       │
 └────────────────┴────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

 Other checkpoint_dir usages that change:

 # BTA breakdown checkpoint — before:
 ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
 # After:
 ckpt = self.workspace.checkpoint_path("breakdown_result.json")

 # BTA aggregator result path — before:
 _agg_ckpt = os.path.join(self.checkpoint_dir, "aggregator_result")
 # After:
 _agg_ckpt = self.workspace.checkpoint_path("aggregator_result")

 # DualInferencer checkpoint enable check — before:
 elif self.enable_checkpoint and self.checkpoint_dir:
 # After:
 elif self.enable_checkpoint and self.workspace:

 Child workflow setup — per-iteration isolation via iteration workspace:

 The Workflow base's _setup_child_workflows sets child._result_root_override to redirect where child step results are saved. PTI overrides this to isolate children per-iteration.

 Current approach manually constructs iter_N/ inside checkpoints using the stable base workspace:
 # Current — uses base workspace + manual iter_N nesting:
 child_dir = os.path.join(base, "checkpoints", "pti", f"iter_{iteration}", attr_name)
 child._result_root_override = child_dir

 New approach uses the iteration workspace instead of the base workspace. Since each iteration already has its own full workspace tree via _get_iteration_workspace(), child isolation is natural — no manual iter_N/ construction needed:

 # New — uses iteration workspace, isolation is structural:
 def _setup_child_workflows(self, state, *args, **kwargs):
     if state is None:
         return
     iteration = state.get("iteration", 1) if isinstance(state, dict) else 1

     # Key: use ITERATION workspace, not base workspace
     iter_ws_path = self._get_iteration_workspace(self.workspace.root, iteration)
     iter_workspace = InferencerWorkspace(root=iter_ws_path)

     all_children = {}
     all_children.update(self._find_child_workflows_in(self))
     all_children.update(self._find_child_workflows_in(state))

     for attr_name, (child, entry) in all_children.items():
         child.workspace = iter_workspace.child(attr_name)
         child.workspace.ensure_dirs()
         child._result_root_override = child.workspace.checkpoints_dir
         child.enable_result_save = self.enable_result_save
         child.resume_with_saved_results = self.resume_with_saved_results
         child.checkpoint_mode = self.checkpoint_mode

 Why this works — isolation is structural, not manual:

 Each iteration gets its own workspace via _get_iteration_workspace():
 - Iteration 1 → workspace_root/
 - Iteration 2 → workspace_root/followup_iterations/iteration_2/
 - Iteration 3 → workspace_root/followup_iterations/iteration_3/

 Each iteration workspace has its own children/ tree, so child checkpoints are naturally isolated:

 workspace_root/
 ├── children/                                    # iteration 1's children
 │   └── planner/checkpoints/step_*.json          # planner checkpoints for iter 1
 └── followup_iterations/
     ├── iteration_2/
     │   └── children/                            # iteration 2's children — ISOLATED
     │       └── planner/checkpoints/step_*.json  # planner checkpoints for iter 2
     └── iteration_3/
         └── children/                            # iteration 3's children — ISOLATED
             └── planner/checkpoints/step_*.json  # planner checkpoints for iter 3

 No overwriting, no iter_N/ nesting inside checkpoints. Cleaner than the current approach.

 Constructor backward compat (optional, low priority):

 If callers still pass checkpoint_dir= to DualInferencer or BTA, __attrs_post_init__ can auto-create a workspace:
 def __attrs_post_init__(self):
     if self.checkpoint_dir and self.workspace is None:
         self.workspace = InferencerWorkspace(root=self.checkpoint_dir)
     # ... rest of init ...
 This is a convenience for migration — not a permanent feature.

 Migration Strategy for Existing Checkpoints

 Approach: clean break.

 Existing on-disk checkpoints from prior runs use old paths. Old checkpoints won't resume under the new layout. This is acceptable because:
 - These are development/research workloads, not production databases
 - Checkpoint resume is a convenience feature, not a durability guarantee
 - Users can re-run from scratch with minimal cost

 Test Update Plan

 All path assertions in the following test files need updating:

 - test_dual_inferencer_resume.py: checkpoint_dir → workspace, checkpoint paths under workspace.checkpoints_dir
 - test_pti_resume.py: _get_result_path assertions, workspace state detection paths
 - test_recursive_resume.py: _result_root_override paths, child isolation paths now use child.workspace.checkpoints_dir
 - test_breakdown_then_aggregate.py: breakdown checkpoint paths under workspace.checkpoints_dir
 - test_resume_detection.py: marker detection now uses workspace.has_marker(), artifact paths change

 The test structure stays the same — only path string literals and constructor args change. The resume mechanism itself is not modified.

 ---
 Resolved Questions

 1. Where to add workspace/output_path? → InferencerWorkspace is a standalone class. workspace (optional attr) and output_path live on InferencerBase. Simple API inferencers just leave workspace=None. No mixin needed.
 2. Naming: → workspace (the InferencerWorkspace object) replaces both workspace_path (PTI) and checkpoint_dir (BTA/DualInferencer). The root path is workspace.root. PTI's workspace_path attr becomes workspace.
 3. children/ vs current naming: → Clean break. PTI's checkpoints/pti/iter_N/<attr_name>/ → children/<attr_name>/checkpoints/. Old checkpoints won't resume. (See Migration Strategy above.)
 4. DualInferencer's round_index template: → Resolved by outputs/ vs artifacts/ split. {{ round_index }} applies to artifacts/ filenames. Class-level output_path is template-free, points to final output in outputs/.

 Remaining Open Questions

 1. DualInferencer children/: DualInferencer has base_inferencer, review_inferencer, fixer_inferencer. These are typically stateless API callers that don't need workspaces. However, they could be flow inferencers. Recommendation: DualInferencer sets child.workspace =
 self.workspace.child(name) only if the child already has a non-None workspace or output_path, or if it's an instance of a flow inferencer type. Simple API inferencers are left alone.

 ~~2. **PTI iteration isolation:**~~ → Resolved. Each iteration already gets its own full workspace via _get_iteration_workspace() (iteration 1 → workspace_root/, iteration N → followup_iterations/iteration_N/). Each iteration workspace has its own children/ tree, so child
 checkpoints are naturally isolated without manual iter_N/ nesting. PTI's _setup_child_workflows now uses the iteration workspace to set child workspaces. See "Child workflow setup" in the checkpoint_dir migration section for the full implementation.
