# MFDual Workspace Anomalies — Implementation Reference (v4.8)

**Purpose**: Machine-followable implementation guide. Contains ONLY current v4.8 designs — no history, no superseded versions.
**Authority**: Extracted from `mfdual_workspace_layout_anomalies_fix_plan.md` (2273 lines). That plan has the WHY; this file has the WHAT.

---

## Implementation Order

```
1. Fix #9 + #10  (LazyConfigFactory + _ImportFactory alias)    ~4.5h
2. §4            (layered switch_role API + migrations)        ~4h
3. Fix #5A       (diagnostic logging)                          ~15min
4. Fix #7 + #8   (audit hardening + snapshot semantics)        ~2.25h
5. Fix #1        (aggregator workspace via switch_role)        ~30min
6. Fix #2        (no double final_deliverables)                ~45min
7. Fix #3        (round01 placeholder removal)                 ~1h
8. Fix #5B       (recursive sharing detection)                 ~2.25h
                                                     Total:   ~16h
```

---

## Fix #9: LazyConfigFactory (ROOT CAUSE)

### New file: `RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py`

```python
"""LazyConfigFactory — re-instantiate from stored config on each call."""
from __future__ import annotations
import copy, logging
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)

class LazyConfigFactory:
    __slots__ = ("_config_dict", "_injectables", "template_extra_feed")

    def __init__(self, *, config_dict: Dict[str, Any], injectables: Optional[Dict[str, Any]] = None):
        if not isinstance(config_dict, dict):
            raise TypeError(f"config_dict must be dict, got {type(config_dict).__name__}")
        self._config_dict = config_dict
        self._injectables = injectables or {}
        self.template_extra_feed: dict = {}

    def __call__(self) -> Any:
        from rich_python_utils.config_utils._instantiate import instantiate
        from omegaconf import OmegaConf
        config = copy.deepcopy(self._config_dict)
        for k, v in self._injectables.items():
            injectable_key = f"_{k}"
            if injectable_key in config:
                config[injectable_key] = v
        instance = instantiate(OmegaConf.create(config))
        if self.template_extra_feed and hasattr(instance, "template_extra_feed"):
            instance.template_extra_feed.update(self.template_extra_feed)
        return instance

    @property
    def target(self) -> str:
        return self._config_dict.get("_target_", "<unknown>")

    def __repr__(self) -> str:
        return f"LazyConfigFactory(target={self.target!r})"
```

### Modify: `RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py`

**Walker change** (lines 944-972, the `*_factory` block): Make `_factory_configs.append(...)` UNCONDITIONAL:

```python
if "_target_" in val:
    raw = copy.deepcopy(val)
    if _FACTORY_MARKER in val:
        del raw[_FACTORY_MARKER]
        del val[_FACTORY_MARKER]
    if _factory_configs is not None:
        _factory_configs.append((a.name, None, raw, _injectables or {}))  # ALWAYS
    val["_partial_"] = True
```

Same for dict-of-factories branch (lines 960-972).

**Rename** `_apply_import_factory` → `_apply_lazy_factory` (line ~356):

```python
def _apply_lazy_factory(obj, field_name, child_key, raw_config, injectables=None):
    from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory
    container = getattr(obj, field_name, None)
    if container is None: return
    factory = LazyConfigFactory(config_dict=raw_config, injectables=injectables)
    if child_key is None:
        setattr(obj, field_name, factory)
    elif isinstance(container, dict):
        container[child_key] = factory
```

### Modify: `AgentFoundation/.../breakdown_then_aggregate_inferencer.py`

**BTA isinstance** (lines 1504, 1509):

```python
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

if isinstance(factory, (functools.partial, LazyConfigFactory)):
    worker = factory()
else:
    worker = factory(sub_query=query_str, index=i)
```

**Accept**: Two `factory()` calls → distinct nested instances. No shared `id()` across workers.

---

## Fix #10: _ImportFactory Deprecated Alias

### Modify: `RichPythonUtils/.../config_utils/_instantiate.py` (lines 38-65)

```python
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

class _ImportFactory(LazyConfigFactory):
    def __init__(self, config_dict, injectables=None):
        import warnings
        warnings.warn("_ImportFactory deprecated; use LazyConfigFactory", DeprecationWarning, stacklevel=2)
        super().__init__(config_dict=config_dict, injectables=injectables)
```

**Accept**: `isinstance(_ImportFactory(...), LazyConfigFactory)` → True. DeprecationWarning emitted.

---

## §4: Layered switch_role()

### Add to `inferencer_base.py` (~line 240):

```python
_ROLE_RELEVANT_ATTRS: tuple = ("output_is_deliverable", "is_deliverable_boundary")

def switch_role(self, new_role: str, *, workspace=None,
                output_is_deliverable=None, is_deliverable_boundary=None,
                reset_session=True):
    import time
    if workspace is not None:
        self._workspace = workspace
    for attr, val in {"output_is_deliverable": output_is_deliverable,
                      "is_deliverable_boundary": is_deliverable_boundary}.items():
        if val is not None:
            setattr(self, attr, val)
    if reset_session:
        self.reset_session()
    history = getattr(self, "_role_history", None)
    if history is None:
        history = []
        object.__setattr__(self, "_role_history", history)
    changes = {**({"workspace": str(workspace.root)} if workspace else {}),
               **{k: v for k, v in {"output_is_deliverable": output_is_deliverable,
                  "is_deliverable_boundary": is_deliverable_boundary}.items() if v is not None}}
    pending = getattr(self, "_pending_role_changes", None)
    if pending:
        changes.update(pending)
        object.__setattr__(self, "_pending_role_changes", None)
    history.append({"to_role": new_role, "at": time.time(), "changes": changes})
```

### Add to `templated_inferencer_base.py` (~line 365):

```python
_ROLE_RELEVANT_ATTRS = InferencerBase._ROLE_RELEVANT_ATTRS + (
    "template_key", "template_root_space", "template_extra_feed",
    "template_variables", "template_version", "modes",
)

def switch_role(self, new_role, *, template_key=None, template_root_space=None,
                template_extra_feed=None, template_variables=None,
                template_version=None, modes=None, **base_kwargs):
    changes = {}
    for attr, val in {"template_key": template_key, "template_root_space": template_root_space,
                      "template_extra_feed": template_extra_feed, "template_variables": template_variables,
                      "template_version": template_version, "modes": modes}.items():
        if val is not None:
            setattr(self, attr, val)
            changes[attr] = val
    if changes:
        object.__setattr__(self, "_pending_role_changes", changes)
    super().switch_role(new_role, **base_kwargs)
```

### Update `multi_flow_dual_inferencer.py`:

```python
# New attribs (optional overrides):
review_template_key: Optional[str] = attrib(default=None)
review_template_root_space: Optional[str] = attrib(default=None)
followup_template_key: Optional[str] = attrib(default=None)
followup_template_root_space: Optional[str] = attrib(default=None)

def _resolve_role_template(self, role_name):
    from agent_foundation.common.inferencers.template_defaults import (
        REVIEW_TEMPLATE_DEFAULTS, FOLLOWUP_TEMPLATE_DEFAULTS)
    if role_name in ("reviewer", "review_inferencer"):
        return (self.review_template_key or REVIEW_TEMPLATE_DEFAULTS.template_key,
                self.review_template_root_space or getattr(REVIEW_TEMPLATE_DEFAULTS, 'template_root_space', None))
    elif role_name in ("fixer", "fixer_inferencer"):
        return (self.followup_template_key or FOLLOWUP_TEMPLATE_DEFAULTS.template_key,
                self.followup_template_root_space or getattr(FOLLOWUP_TEMPLATE_DEFAULTS, 'template_root_space', None))
    return (None, None)

def _reassign_role_workspace(self, inferencer, role_name):
    if inferencer is None or self._workspace is None: return
    original = getattr(self, f"_{role_name}_original", None)
    if inferencer is original: return
    role_ws = self._workspace.child(role_name)
    role_ws.ensure_dirs()
    new_key, new_root = self._resolve_role_template(role_name)
    inferencer.switch_role(new_role=role_name, workspace=role_ws,
                           template_key=new_key, template_root_space=new_root,
                           output_is_deliverable=(True if role_name == "fixer_inferencer" else None))
```

Remove inline `output_is_deliverable = True` from `_select_reviewer_and_fixer()`.

**Accept**: Fixer `template_key=="followup"`. Reviewer `template_key=="review"`. `_role_history` records.

---

## Fix #1: Aggregator Workspace

### Modify: `multi_flow_inferencer.py` (before aggregator invocation)

```python
if self.aggregator_inferencer is not None and self._workspace is not None:
    expected = self._workspace.child("aggregator").root
    current = getattr(getattr(self.aggregator_inferencer, "_workspace", None), "root", None)
    if current != expected:
        agg_ws = self._workspace.child("aggregator")
        agg_ws.ensure_dirs()
        self.aggregator_inferencer._workspace = agg_ws
```

**Accept**: Both workers' aggregators write to canonical `aggregator/` workspace.

---

## Fix #2: No Double final_deliverables

### Modify: `inferencer_workspace.py` — `surface_outputs_from()` (line ~177)

```python
for root_dir, dirs, files in os.walk(src_root):
    dirs[:] = [d for d in dirs if d != "final_deliverables"]  # v4.8: prevent nesting
    for f in files:
        # ... existing copy logic unchanged
```

**Accept**: No `final_deliverables/final_deliverables/` at any level.

---

## Fix #3: Remove Round01 Placeholder

### Modify: `multi_flow_inferencer.py` — `_propagate_workspace_to_children()` (line ~531)

Option A.1 (preferred): Omit followup_inferencer from propagation entirely:

```python
for slot, suffix in (
    ("initial_inferencer", f"flow_{i}_initial"),
    # followup_inferencer: OMITTED — LWI assigns per-step
):
```

Also in LWI: change guard from `if step_index > 0` to always assign. First step → `_round01`.

**Accept**: No empty `flow_X_round01/` directories. First followup in `_round01`.

---

## Fix #5: Worker Sharing Detection

### Add to `inferencer_base.py` (after `_iter_child_inferencers`):

```python
def _collect_all_descendant_inferencers(self, _seen=None):
    if _seen is None: _seen = set()
    if id(self) in _seen: return
    _seen.add(id(self))
    yield self
    for child in self._iter_child_inferencers():
        yield from child._collect_all_descendant_inferencers(_seen=_seen)
```

### Add to `breakdown_then_aggregate_inferencer.py`:

```python
worker_isolation_check: bool = attrib(default=True, kw_only=True)

def _validate_worker_isolation(self, workers):
    if not self.worker_isolation_check: return
    seen = {}
    for i, w in enumerate(workers):
        if not isinstance(w, InferencerBase): continue
        for inf in w._collect_all_descendant_inferencers():
            iid = id(inf)
            if iid in seen and seen[iid] != i:
                _logger.warning("BTA[%s] workers %d and %d share inferencer %s (id=0x%x)",
                                getattr(self, "name", "?"), i, seen[iid], type(inf).__name__, iid)
            else:
                seen[iid] = i
```

Call at end of `_build_subgraph_spec()`, after worker creation loop.

**Accept**: Shared instances → warning. Zero warnings after Fix #9. `worker_isolation_check=False` skips.

---

## Fix #7 + #8: Audit Hardening

### Modify: `dual_inferencer.py` — `_record_round_audit()` (line ~713)

Add `workspace_root_at_phase: Optional[str] = None` parameter. Use snapshot if provided:

```python
target = workspace_root_at_phase or str(inferencer._workspace.root)

# Check 1: Cross-worker leakage
my_root = str(self._workspace.root).rstrip("/") + "/"
if not target.startswith(my_root):
    _logger.error("Audit: cross-worker leakage at round_%02d/%s: target %s outside %s",
                  round_idx, phase, target, self._workspace.root)
```

Call sites capture snapshot BEFORE `ainfer()`:

```python
ws_snapshot = str(self.review_inferencer._workspace.root) if ... else None
# ... ainfer() ...
self._record_round_audit(..., workspace_root_at_phase=ws_snapshot)
```

**Accept**: Cross-worker targets → ERROR log. Symlinks reflect phase-execution workspace.

---

## File Paths (ALL correct)

| Component | Path |
|-----------|------|
| LazyConfigFactory (NEW) | `CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py` |
| Walker / _ImportFactory | `CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` |
| InferencerBase | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_base.py` |
| TemplatedInferencerBase | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/templated_inferencer_base.py` |
| InferencerWorkspace | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_workspace.py` |
| MFDual | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py` |
| MultiFlow | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_inferencer.py` |
| Dual | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py` |
| BTA | `CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py` |

**Import convention**: ALL code uses `from rich_python_utils.config_utils...` (NOT `python_utils`).
