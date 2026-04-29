# Workspace Regularization Plan

## 1. Background & Motivation

### Current State (Problems)

The `InferencerWorkspace` class exists and is well-designed, but its usage is ad-hoc and inconsistent across the inferencer hierarchy:

| Class | How workspace is used |
|---|---|
| `InferencerBase` | `_workspace` is NOT a declared field — only used via `getattr(self, '_workspace', None)` in `resolve_output_path()`. Duck-typed, fragile. |
| `BreakdownThenAggregateInferencer` | Has `workspace_root: str` attrib — creates `InferencerWorkspace` in `__attrs_post_init__`. Also has `use_final_deliverables_folder: Union[bool, str]` as a passthrough to workspace. |
| `DualInferencer` | Has `workspace_root: str` attrib — creates `InferencerWorkspace` in `__attrs_post_init__`. |
| `PlanThenImplementInferencer` | Uses `workspace_path`, `resume_workspace`, `_current_base_workspace`, `_current_iteration_workspace` — more complex multi-workspace pattern. |
| `TerminalInferencerBase` | Has `working_dir: str` attrib — a workspace concern handled separately. |
| All inferencers | `cache_folder`, `output_path`, `checkpoint_dir` are declared independently — all workspace concerns, all duplicated. |

### The Core Problems

1. **`_workspace` is not a declared field** — set dynamically via duck typing, invisible to type checkers and IDE tooling
2. **Workspace config scattered** — `use_final_deliverables_folder` had to be added to BTA (not InferencerWorkspace) just to pass it through
3. **`workspace_root: str` is a poor API** — forces callers to pass a string; they can't pass a pre-configured workspace with custom settings
4. **No uniformity** — flow inferencers each invent their own workspace pattern

### Goal

Regularize workspace usage across the inferencer hierarchy:
- Add `workspace: Optional[InferencerWorkspace]` as a **proper declared field** on `InferencerBase`
- BTA and DualInferencer accept `workspace` directly (not just `workspace_root`)
- `use_final_deliverables_folder` and similar workspace config live ONLY on `InferencerWorkspace`
- `workspace_root: str` remains as a convenience shorthand (backward compat)
- All other inferencers can OPT IN to workspace when useful — no forced migration

---

## 2. Design

### 2.1 `InferencerBase` — Add `workspace` field

```python
# In InferencerBase:
workspace: Optional["InferencerWorkspace"] = attrib(default=None)

def __attrs_post_init__(self):
    # ... existing code ...
    # Sync: if workspace provided, set _workspace for backward compat
    if self.workspace is not None:
        self._workspace = self.workspace
```

**Key points:**
- `workspace` is the new public field — declared, typed, YAML-instantiatable
- `_workspace` continues to work as an alias (backward compat for code that sets it directly)
- `resolve_output_path()` already uses `getattr(self, '_workspace', None)` — works with both
- Inferencers that don't need workspace simply leave `workspace=None` (default)
- Registered as `InferencerWorkspace` in YAML for config-driven instantiation

### 2.2 `BreakdownThenAggregateInferencer` — Accept `workspace` directly

```python
# Before:
workspace_root: Optional[str] = attrib(default=None)
use_final_deliverables_folder: Union[bool, str] = attrib(default=False)  # ← REMOVE

# After:
workspace_root: Optional[str] = attrib(default=None)  # kept for backward compat
# workspace: inherited from InferencerBase ← workspace takes precedence

def __attrs_post_init__(self):
    super().__attrs_post_init__()
    if self.workspace is not None:
        # Pre-configured workspace provided directly — use as-is
        self._workspace = self.workspace
        self._workspace.ensure_dirs()
    elif self.workspace_root is not None:
        # Convenience: create workspace from root path
        from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
        self._workspace = InferencerWorkspace(root=self.workspace_root)
        self._workspace.ensure_dirs()
    else:
        self._workspace = None
    # Auto-default output_path
    if not self.output_path:
        self.output_path = "aggregation_report.md"
```

**Key points:**
- `use_final_deliverables_folder` is REMOVED from BTA — it's an `InferencerWorkspace` attribute
- `workspace` takes precedence over `workspace_root`
- `workspace_root` still works for backward compat (creates a plain `InferencerWorkspace`)
- YAML can now specify full workspace config: `workspace: {_target_: InferencerWorkspace, root: ..., use_final_deliverables_folder: true}`

### 2.3 `DualInferencer` — Same pattern as BTA

```python
# Before:
workspace_root: Optional[str] = attrib(default=None, kw_only=True)

# After: workspace_root kept for compat, workspace from InferencerBase
def __attrs_post_init__(self):
    super().__attrs_post_init__()
    if self.workspace is not None:
        self._workspace = self.workspace
        self._workspace.ensure_dirs()
    elif self.workspace_root is not None:
        self._workspace = InferencerWorkspace(root=self.workspace_root)
        self._workspace.ensure_dirs()
    else:
        self._workspace = None
```

### 2.4 YAML config (inner_bta_skill_tool_creation.yaml)

```yaml
# Before:
_target_: BTA
use_final_deliverables_folder: true
output_path: "aggregation_report.md"  # auto-defaulted

# After:
_target_: BTA
workspace:
  _target_: InferencerWorkspace
  root: ""  # set at runtime via override
  use_final_deliverables_folder: true
```

Or with runtime override:
```python
overrides = {
    "workspace.root": str(workspace_root),
    # use_final_deliverables_folder: already in YAML
}
```

### 2.5 Registration

`InferencerWorkspace` must be registered for YAML instantiation:
```python
# In registered_targets.py:
register_alias(
    "InferencerWorkspace",
    "agent_foundation.common.inferencers.inferencer_workspace.InferencerWorkspace",
    "config",
)
```

---

## 3. Scope of Changes

### Phase 1 (NOW — this task)

| File | Change |
|---|---|
| `inferencer_base.py` | Add `workspace: Optional[InferencerWorkspace] = attrib(default=None)`, sync to `_workspace` in `__attrs_post_init__` |
| `breakdown_then_aggregate_inferencer.py` | Accept `workspace` directly; remove `use_final_deliverables_folder`; keep `workspace_root` for compat |
| `dual_inferencer.py` | Accept `workspace` directly; keep `workspace_root` for compat |
| `registered_targets.py` | Add `InferencerWorkspace` alias |
| `inner_bta_skill_tool_creation.yaml` | Use `workspace: {_target_: InferencerWorkspace, ...}` instead of `use_final_deliverables_folder` at BTA level |
| `test_inner_bta_yaml_equivalence.py` | Update tests for new workspace field |

### Phase 2 (FUTURE — separate task)

- Migrate `cache_folder`, `working_dir` to workspace-driven when `workspace` is set
- Add `workspace` to `TerminalInferencerBase` (replace `working_dir` as primary)
- Add `workspace` to `StreamingInferencerBase` (replace `cache_folder`)
- Migrate `PlanThenImplementInferencer` to use `workspace` field
- Add `workspace` support to API inferencers (for logging)

---

## 4. Backward Compatibility

| Existing usage | Impact after change |
|---|---|
| `BTA(workspace_root="/path")` | ✅ Still works — `workspace_root` honored when `workspace=None` |
| `DualInferencer(workspace_root="/path")` | ✅ Still works |
| `agg_inf._workspace = agg_ws` (dynamic set) | ✅ Still works — `_workspace` alias preserved |
| `getattr(inf, '_workspace', None)` | ✅ Still works — `_workspace` synced from `workspace` in `__attrs_post_init__` |
| Existing callers not setting `workspace` | ✅ `workspace=None` by default — no behavior change |

---

## 5. Implementation Order

| Step | Task | File |
|---|---|---|
| 1 | Add `workspace` field + `_workspace` sync to `InferencerBase` | `inferencer_base.py` |
| 2 | Add `InferencerWorkspace` alias to registered_targets | `registered_targets.py` |
| 3 | Update BTA: use `workspace` field, remove `use_final_deliverables_folder` | `breakdown_then_aggregate_inferencer.py` |
| 4 | Update DualInferencer: use `workspace` field | `dual_inferencer.py` |
| 5 | Update YAML config: use `workspace: {_target_: InferencerWorkspace, ...}` | `inner_bta_skill_tool_creation.yaml` |
| 6 | Update tests | `test_inner_bta_yaml_equivalence.py` |
| 7 | Verify all tests pass | — |

---

## 6. Open Questions

1. **`workspace.root` override in YAML**: The workspace `root` is a runtime value (not known at YAML authoring time). Best approach: set `root: ""` in YAML, override at runtime via `load_config(overrides={"workspace.root": str(workspace_root)})`. Does this work with OmegaConf?

2. **Child workspace `use_final_deliverables_folder`**: When BTA creates child workspaces for workers/aggregator (`self._workspace.child("aggregator")`), should `use_final_deliverables_folder` propagate? Likely NO — child workspaces don't need their own `final_deliverables/` subfolder.

3. **`_workspace` vs `workspace`**: Should we rename all internal `_workspace` references to `workspace`? Pro: consistency. Con: bigger diff. Recommendation: leave `_workspace` as internal alias for now; only the declared field is `workspace`.
