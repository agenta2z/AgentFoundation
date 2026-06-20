# Debug-Mode Cascade + CI Default YAML — Plan v2

> **⚠ Reviewer banner — please read v2 changes first**
>
> **v1 → v2 critical correction:** Two peer plans (Plan B "Cursor" `debug_mode_cascade_and_ci_config_82c8b750.plan.md` and Plan C "Claude" `update-your-task-tool-adaptive-goose.md`) independently caught a **fatal architectural bug in v1**: my v1 built the cascade on `_iter_child_inferencers`, but that method is **NOT implemented on `ConversationalInferencer`, `PTI`, or `LWI`** — only on Dual, BTA, MFDual, MFI, and base (no-op). Source-verified: `grep -rn "def _iter_child_inferencers" src/agent_foundation/common/inferencers/` returns those 5 hits and no others. The CI's cascade would have NEVER reached `base_inferencer`, silently failing the entire stated goal. **The verified existing precedent (`_propagate_workspace_to_children` at line 450) uses `_for_each_child_inferencer` — the attrs-field walker — NOT `_iter_child_inferencers`.** My v1 was inconsistent with its own stated precedent.
>
> **v2 adds 5 critical corrections:**
> 1. **Cascade built on `_for_each_child_inferencer`** (attrs walker that auto-discovers `base_inferencer: InferencerBase = attrib(...)` fields — works for CI, PTI, LWI, Dual, BTA, MFDual). Matches the verified workspace-propagation precedent exactly.
> 2. **THREE trigger sites** (Plan B's verified map): (a) `__attrs_post_init__` — covers construction-time children including CI's `base_inferencer`; (b) overrides on `Debuggable.enable_debug_mode`/`disable_debug_mode` — covers runtime toggle on CI (the ONLY hook CI has for runtime); (c) `super()._propagate_to_children()` fix in `TemplatedInferencerBase` — covers orchestrator runtime-created children.
> 3. **One-line factory quick-win** (`debug_mode=True` in `factories.py:167`) ships independent of the cascade as Pathway 0 — solves the immediate user need in ONE line while Pathway 1 (the principled mechanism) and Pathway 2 (declarative YAML) follow.
> 4. **`prompt_renderer` runtime-passed**, not YAML-declared — the factory must build it BEFORE the CI to run `_filter_tools_by_config` (verified). `_ci_host.build_ci_from_config` needs an optional `prompt_renderer=None` param.
> 5. **OpenStartup's YAML override declares `prompt_renderer`** (the conversation/main prompt path is app-specific to OpenStartup, not framework default).
>
> **What this plan fixes (v2 statement):** the conversational inferencer's `base_inferencer` (and its descendants) do not inherit `debug_mode=True` from the CI — verified at `factories.py:167` where CI is constructed without `debug_mode` and the backend is constructed separately and passed in, so any runtime `ci.debug_mode = True` is unreachable. v2 closes this with one of three escalating mechanisms: (Pathway 0) one-line factory fix today; (Pathway 1) principled `_CASCADING_ATTRIBUTES` mechanism on the attrs walker; (Pathway 2) declarative CI YAML that uses the verified underscore-prefix cascade in `_instantiate.py`.
>
> **What this plan delivers:** Two coupled-but-independent pathways for making `debug_mode` (and a small set of other "infrastructure" attrs) cascade from a parent inferencer to its children — solving the immediate "I want debug-mode CI runs to log everything inside child task/tool inferencers" need, AND laying the foundation for declarative CI configuration via YAML.
>
> **Pathway 1 (code-level cascade — proper, longer-term):** Add a `_CASCADING_ATTRIBUTES: ClassVar[list]` on `InferencerBase`. Each entry is either a string (attr name; cascade when child's value is `None`) or a `(name, condition_callable)` tuple. The cascade runs in a new `_propagate_cascading_attrs_to_children()` method, mirroring the verified existing pattern of `_propagate_workspace_to_children` (`inferencer_base.py:450`). `debug_mode` becomes `Optional[bool]` with `None` = inherit. This is the principled, transport-agnostic, opt-in mechanism that solves the general problem.
>
> **Pathway 2 (YAML config — immediate, pragmatic):** Add `src/agent_foundation/resources/configs/conversational/default.yaml` (and `base_inferencer/{ClaudeCodeCLI,RovoDevCLI}.yaml` siblings) — the **same proven shape** already used by the `sop` and `task` tools (`src/agent_foundation/resources/tools/{sop,task}/configs/default.yaml`). The YAML's `_debug_mode: true` (underscore-prefix injectable) already cascades to all descendants AT CONFIG-INSTANTIATION TIME via the verified `_instantiate.py` mechanism. OpenStartup's `factories.py` switches from hand-construction to YAML-driven build, giving operators a single declarative knob.
>
> **Why ship both:** Pathway 2 alone solves *YAML-instantiated* CI sessions (most production paths) but does NOT solve sessions built programmatically in test fixtures, dev tooling, or hand-construction sites. Pathway 1 alone requires touching `Debuggable` in RichPythonUtils (cross-repo change) and pushes a behavior change to every inferencer using the cascade. **Pathway 2 ships first (lower risk, immediate value); Pathway 1 follows as the proper long-term mechanism. Both ultimately serve the same goal: one knob, propagated everywhere it should be.**
>
> **What this plan does NOT do (explicit non-goals):**
> - **NOT** changing `debug_mode`'s default from `False` to `True`. That's an OpenStartup operator decision per session, not a framework default.
> - **NOT** unifying the two propagation primitives (`_for_each_child_inferencer` vs `_iter_child_inferencers`) — they're "deliberately not unified" (`inferencer_base.py:756`).
> - **NOT** auto-cascading `model_name` (the original proposal included it; rejected after verifying `model_name` is already cascaded by the `_model_name:` injectable in YAML, and runtime cascade risks silently overriding intentional per-leaf model choices).
> - **NOT** building a generic config-discovery service or a configurable-everything framework. Single-purpose change.

---

**Author:** Rovo Dev (CI session)
**Date:** 2026-06-14
**Status:** Draft v1, not yet committed
**Branch:** `dev_xinli_2601` (AgentFoundation); coupled change in `OpenStartup`; one small back-compat change in `RichPythonUtils` (Pathway 1 only)
**Companion to:** `interactive_widget_for_agent_dispatched_tools_plan.md` (also in this folder); `task_complexity_presets_and_chat_peer_plan.md` (whose `--config conversational` mode is the natural consumer of Pathway 2's CI default YAML)
**Cross-repo:** Pathway 2 changes AgentFoundation + OpenStartup. Pathway 1 changes RichPythonUtils + AgentFoundation. Both pathways are independent — either can ship without the other.

---

## §0. Quick-start

**What this plan does (one paragraph):** Adds a small, principled mechanism (`_CASCADING_ATTRIBUTES` class variable on `InferencerBase`) for parent inferencers to cascade specific attributes to children whose values are unset — mirroring the verified existing pattern of `_propagate_workspace_to_children`. In parallel, lifts the conversational inferencer out of hand-construction (`OpenStartup/.../factories.py`) into a declarative `default.yaml` config under a NEW `src/agent_foundation/resources/configs/conversational/` directory — using the SAME YAML idiom already proven by `sop` and `task` tools. Either pathway delivers debug-mode cascade on its own; together they cover both YAML-instantiated and programmatically-constructed CI sessions.

**Effort estimate:** ~2 days total.
- Pathway 1: ~1 day (1 small RichPythonUtils change + 1 AgentFoundation mechanism + ~80 LoC tests).
- Pathway 2: ~1 day (3 new YAMLs + 1 factory rewrite + ~60 LoC tests).

**Commits in dependency order (Pathway 2 first — lower risk, immediate value):**

| # | Pathway | Repo | Commit | Purpose | LoC |
|---|---|---|---|---|---|
| 1 | 2 | AgentFoundation | `src/agent_foundation/resources/configs/conversational/default.yaml` + `base_inferencer/ClaudeCodeCLI.yaml` + `base_inferencer/RovoDevCLI.yaml` | The CI config tree — mirrors the proven `sop/configs/` layout | ~90 |
| 2 | 2 | OpenStartup | `OpenStartup/.../factories.py` switches from hand-construction to `_ci_host.build_ci_from_config(...)` | YAML-driven CI; operators get one declarative knob | ~30 |
| 3 | 2 | both | Integration tests + smoke (`_debug_mode: true` cascades to descendants at config-instantiation time) | Locks the YAML cascade behavior; protects from regression | ~60 tests |
| 4 | 1 | RichPythonUtils | `Debuggable.debug_mode: bool` → `Optional[bool]` (default `None`) + treat `None` as `False` in log-level logic | The "unset" sentinel enables runtime cascade | ~20 |
| 5 | 1 | AgentFoundation | `InferencerBase._CASCADING_ATTRIBUTES` + `_propagate_cascading_attrs_to_children()` + hook into `__attrs_post_init__` | The runtime cascade mechanism itself | ~60 |
| 6 | 1 | AgentFoundation | Pathway-1 tests (cascade-when-None; respect-explicit-set; conditional-callable; recursion through Dual/BTA/LWI/MFDual) | Locks the runtime cascade behavior | ~80 tests |

**Lowest-risk first:** Commit 1 (Pathway 2 YAMLs) is pure additive — no behavior change until Commit 2 wires the factory. Commit 4 is a pure-additive `Optional[bool]` change with a documented `None`-treated-as-`False` rule (the existing code at `debuggable.py:922` reads `self.debug_mode` in `if`/`else` contexts where `None` already evaluates as falsy).

---

# PART I — EXECUTION
══════════════════════════════════════════════════════════════════════════════

## §E0. Pathway 0 — One-line factory quick-win (v2 NEW — ships TODAY, independent of other pathways)

**Purpose:** Solve the immediate user need in one line, without waiting for the principled mechanism or the YAML refactor.

**File modified (OpenStartup):** `OpenStartup/src/openteam/server/backends/factories.py:167` (~1 LoC):

```python
# Before:
conv_inferencer = ConversationalInferencer(
    base_inferencer=base,
    prompt_renderer=prompt_renderer,
    ...
)

# After (1-line addition):
conv_inferencer = ConversationalInferencer(
    base_inferencer=base,
    prompt_renderer=prompt_renderer,
    ...
    debug_mode=True,   # Operators flip in code (or env-var) per session
)
```

**Why this is acceptable as a quick-win (not "ad-hoc"):**
- Even without the cascade, this sets `debug_mode=True` on the CI itself, which already enables verbose logging at the CI layer.
- The `base` (RovoDevCliInferencer) is constructed at `factories.py:214` — to also enable debug on it pre-cascade, add `debug_mode=True` to that constructor call too. That's 2 lines, not 1.
- This pathway is **superseded** by Pathway 1 (once landed, the cascade does this automatically for any attrs-field child of the CI) and Pathway 2 (once landed, the `_debug_mode: true` injectable does it via YAML).
- Ships independently of both other pathways — no cross-repo, no test sweep.

**Tests:** Existing CI smoke test passes; logs show DEBUG-level output from the CI.

**Risk:** trivial. Pure additive flag; no behavior change for callers that don't observe debug logs.

**LoC:** 1–2 production + 0 tests (covered by existing smoke).

**When to skip Pathway 0:** If Pathway 2 (YAML) is going to land in the same PR or release window, Pathway 0 is unnecessary — it's a stopgap. Document the decision in the commit message.

---

## §E1. Pathway 2 — CI default YAML (ship first)

### §E1.1 — Commit 1: AgentFoundation `resources/configs/conversational/` tree

**Purpose:** Add the CI config tree using the SAME idiom proven by `src/agent_foundation/resources/tools/{sop,task}/configs/`. The `_debug_mode: true` underscore-prefix injectable cascades to all descendants at config-instantiation time (verified `_instantiate.py:718–725`).

**Files added (AgentFoundation):**

1. `src/agent_foundation/resources/configs/conversational/default.yaml` (~50 LoC):

```yaml
# Default Conversational Inferencer config
#
# This is the canonical default CI for AgentFoundation consumers.
# OpenStartup, tests, and any external consumer building a conversational
# inferencer SHOULD load this config rather than hand-construct.
#
# Underscore-prefix injectables (_debug_mode, _model_name) cascade to
# EVERY descendant inferencer in this tree via _instantiate.py's
# verified cascade (lines 718-725). Override by setting the matching
# named field on a specific child.

_debug_mode: false                    # Operators flip to true for verbose runs
_model_name: ${oc.env:CI_DEFAULT_MODEL,opus[1m]}
_idle_timeout_seconds: 600

_target_: Conversational
max_iterations: 5
compression_threshold: 8000
# ... any other CI defaults intrinsic to the conversational topology

base_inferencer:
  # ${backend} is resolved at build time from --backend / env / default
  # to one of the sibling YAMLs under base_inferencer/
  _target_: ${backend}
```

2. `src/agent_foundation/resources/configs/conversational/base_inferencer/ClaudeCodeCLI.yaml` (~20 LoC):

```yaml
# Claude Code CLI backend for the default conversational inferencer.
# Inherits _debug_mode, _model_name, _idle_timeout_seconds from parent
# via _instantiate.py's underscore-prefix cascade.

_target_: ClaudeCodeCLI
tool_use_idle_timeout_seconds: ${_idle_timeout_seconds}
cache_folder: ${cache_dir}             # Resolved from build context
# enable_legacy=true is set explicitly by callers that need it
```

3. `src/agent_foundation/resources/configs/conversational/base_inferencer/RovoDevCLI.yaml` (~20 LoC):

```yaml
# Rovo Dev CLI backend for the default conversational inferencer.

_target_: RovoDevCli
tool_use_idle_timeout_seconds: ${_idle_timeout_seconds}
cache_folder: ${cache_dir}
enable_legacy: true                     # Required for Rovo Dev backend
```

**Why a NEW `configs/conversational/` directory (not under `tools/`):**

The SOP and task CIs live under `tools/{sop,task}/configs/` because they're specific to those tools' invocation contexts. The general-purpose chat CI is NOT a tool — it's the top-level inferencer that *wraps* tools. Putting it under `resources/configs/conversational/` makes the asymmetry explicit and matches the verified existing convention (the `tools/*/configs/` pattern is used 2× today; `resources/configs/` is the natural top-level sibling for non-tool inferencer configs).

**Alternative considered (and rejected):** `resources/inferencers/conversational/default.yaml`. **Why rejected:** `inferencers/` would suggest one config per inferencer class (`Dual.yaml`, `BTA.yaml`, etc.), but inferencers are normally composed in tool/topology contexts. The new directory is for **deployable inferencer topologies**, not per-class defaults.

**Tests (T1–T3):**
- T1: `instantiate("src/agent_foundation/resources/configs/conversational/default.yaml", backend="ClaudeCodeCLI", cache_dir=tmp_path)` returns a `ConversationalInferencer` instance.
- T2: Set `_debug_mode: true` in the YAML; instantiate; assert root CI has `debug_mode == True` AND `ci.base_inferencer.debug_mode == True` (cascade verified at config-instantiation time).
- T3: Both backend YAMLs round-trip (instantiate without error) under `backend="ClaudeCodeCLI"` and `backend="RovoDevCli"` respectively.

**Risk:** very low. Pure additive YAML; the same idiom is already proven by `sop/configs/default.yaml` and `task/configs/default.yaml`.

**LoC:** ~90 YAML + ~30 tests.

### §E1.2 — Commit 2: OpenStartup `factories.py` switches to YAML-driven CI

**Purpose:** Replace the hand-constructed CI factory with a `build_ci_from_config(...)` helper. Operators (and tests) get a single declarative knob — flip `_debug_mode` in OpenStartup's own `default.yaml` override (Commit 2b) or pass `overrides={"_debug_mode": True}` programmatically.

**Files modified (OpenStartup):**

1. `OpenStartup/src/openteam/server/backends/factories.py` (~30 LoC):

```python
# Before — hand-constructed (no debug_mode cascade):
def build_conversational_inferencer(ctx, ...) -> ConversationalInferencer:
    base = RovoDevCliInferencer(
        tool_use_idle_timeout_seconds=600,
        cache_folder=ctx.cache_dir,
        enable_legacy=True,
    )
    return ConversationalInferencer(
        base_inferencer=base,
        prompt_renderer=prompt_renderer,
        ...
    )

# After — YAML-driven (debug_mode cascades via _debug_mode injectable):
def build_conversational_inferencer(ctx, ...) -> ConversationalInferencer:
    config_path = _resolve_ci_config_path(ctx)
    return build_ci_from_config(
        config_path,
        backend=ctx.backend_name,                    # "ClaudeCodeCLI" or "RovoDevCli"
        cache_dir=ctx.cache_dir,
        overrides={
            # Operator can flip debug per session; otherwise inherit YAML default
            **({"_debug_mode": True} if ctx.debug_mode else {}),
            # Other per-session overrides (model, idle_timeout) go here
        },
    )

def _resolve_ci_config_path(ctx) -> Path:
    """Discovery order:
       1. ctx.ci_config_path if explicitly set (per-session override)
       2. OpenStartup/resources/configs/conversational/default.yaml (deployment override)
       3. AgentFoundation/resources/configs/conversational/default.yaml (framework default)
    """
    if ctx.ci_config_path:
        return ctx.ci_config_path
    openstartup_default = (
        Path(openteam.__file__).parent
        / "server" / "resources" / "configs" / "conversational" / "default.yaml"
    )
    if openstartup_default.exists():
        return openstartup_default
    return (
        Path(agent_foundation.__file__).parent
        / "resources" / "configs" / "conversational" / "default.yaml"
    )
```

2. **Optional Commit 2b** — `OpenStartup/src/openteam/server/resources/configs/conversational/default.yaml`:

```yaml
# OpenStartup deployment-level CI override (optional; only if app-specific
# customizations are needed). If absent, framework default is used.
#
# Example: enable debug mode by default for OpenStartup dev deployments

_debug_mode: ${oc.env:OPENSTARTUP_DEBUG_MODE,false}
_target_: Conversational
# Inherits everything else from AgentFoundation's default
```

The 3-tier resolution order (explicit → deployment → framework default) gives operators clear escape hatches without forcing them to override the framework default.

**Tests (T4–T7):**
- T4: `build_conversational_inferencer(ctx=ctx_with_debug_True)` returns a CI with `debug_mode == True` AND every descendant inferencer reachable via `_iter_child_inferencers()` also has `debug_mode == True` (cascade verified end-to-end).
- T5: `build_conversational_inferencer(ctx=ctx_with_debug_False)` → all `debug_mode == False`.
- T6: With deployment-level OpenStartup override present, that file is loaded; without it, framework default loads. T6 mocks both paths.
- T7: Behavioral regression: existing CI runtime tests (turn handling, tool dispatch, session restore) still pass after the factory rewrite.

**Risk:** medium. The factory rewrite changes the CI construction path for every OpenStartup session. Mitigation: T7 is the critical regression gate; smoke-test path D below verifies end-to-end manually.

**LoC:** ~30 production + ~50 tests.

### §E1.3 — Commit 3: Integration smoke + regression

**Purpose:** End-to-end verification that `_debug_mode: true` in the YAML cascades to all descendants, and that the factory rewrite preserves session behavior.

**Files added:**

1. `tests/integration/test_ci_yaml_debug_mode_cascade.py` (~40 LoC, AgentFoundation):

```python
# Scenario A — debug_mode cascade via YAML
#   - Instantiate the default CI YAML with _debug_mode: true override
#   - Walk every descendant via _iter_child_inferencers (cycle-safe)
#   - Assert: every reachable Debuggable has debug_mode == True
#
# Scenario B — debug_mode cascade with explicit child override
#   - Instantiate with _debug_mode: true at root
#   - Add explicit `debug_mode: false` on base_inferencer
#   - Assert: root has True; base_inferencer has False (explicit wins)
#
# Scenario C — model_name cascade (sanity — already works today)
#   - Verify _model_name: <X> at root propagates to descendants
#   - Catches accidental regression of existing cascade
```

2. `OpenStartup/tests/integration/test_factories_ci_yaml.py` (~30 LoC):

```python
# End-to-end: build CI via factory, run one turn, assert logs include
# debug-level messages from at least one descendant inferencer
```

**Tests (T8–T11):**
- T8: Scenario A passes — full descendant tree has `debug_mode == True`.
- T9: Scenario B passes — explicit child override respected.
- T10: Scenario C passes — `model_name` cascade still works.
- T11: Manual smoke (§E2.2 step D) — start OpenStartup server with `OPENSTARTUP_DEBUG_MODE=true`, send a chat turn that calls a `task` tool, observe debug-level logs in BOTH the CI AND the task's nested inferencers.

**Risk:** low. Pure verification surface; no production code in this commit.

**LoC:** ~70 tests.

---

## §E2. Pathway 1 — Code-level cascade (ship second)

### §E2.1 — Commit 4: RichPythonUtils `Debuggable.debug_mode: Optional[bool]`

**Purpose:** Introduce the "unset" sentinel that enables runtime cascade. `debug_mode: bool = False` becomes `debug_mode: Optional[bool] = None`; the existing log-level logic at `debuggable.py:922` (`if self.debug_mode:`) already treats `None` as falsy in `if`/`else` contexts — no semantic change for existing callers.

**Files modified (RichPythonUtils):**

1. `RichPythonUtils/src/rich_python_utils/common_objects/debuggable.py` (~20 LoC):

```python
# Line 228 today:
#   debug_mode: bool = attrib(default=False, kw_only=True)
# After v1:
debug_mode: Optional[bool] = attrib(default=None, kw_only=True)
"""When ``None`` (default), this Debuggable will inherit ``debug_mode`` from
its parent if a containing object cascades it. When explicitly set to
``True`` or ``False`` at construction time, the explicit value wins — the
parent's cascade is suppressed.

This None-as-inherit semantics matches the verified existing pattern of
``InferencerBase._propagate_workspace_to_children``, which respects
explicit pre-assignment by the caller.

For boolean evaluation (e.g. ``if self.debug_mode: log("…")``), ``None``
behaves as ``False`` in Python's truthy/falsy semantics — so existing
callers that wrote ``if self.debug_mode:`` keep working unchanged. Callers
that need to distinguish "explicitly False" from "unset" should use
``if self.debug_mode is None`` / ``if self.debug_mode is True``.
"""
```

**Tests (T12–T15):**
- T12: `Debuggable()` default → `debug_mode is None`; `if obj.debug_mode:` is falsy (back-compat).
- T13: `Debuggable(debug_mode=True)` → explicit `True`; preserved.
- T14: `Debuggable(debug_mode=False)` → explicit `False`; preserved (NOT collapsed to `None`).
- T15: Existing `Debuggable` subclasses (sample 3) instantiate without error.

**Risk:** low — but cross-repo. Mitigation: type-check the existing 4 hits of `debug_mode:` annotation (verified in §F1) to confirm none of them rely on `debug_mode is False` semantics for explicit-set detection.

**LoC:** ~20 production + ~30 tests.

### §E2.2 — Commit 5: AgentFoundation `_CASCADING_ATTRIBUTES` + propagation method

**Purpose:** The runtime cascade mechanism itself. Mirrors the verified existing pattern of `_propagate_workspace_to_children` (`inferencer_base.py:450`) which uses `_for_each_child_inferencer` for the same kind of "cascade only where child hasn't claimed an explicit value" logic. New method is hooked into the existing `_propagate_to_children` call site (line 1253).

**Files modified (AgentFoundation):**

1. `src/agent_foundation/common/inferencers/inferencer_base.py` (~60 LoC):

```python
# Add as ClassVar near other class-level config:
_CASCADING_ATTRIBUTES: ClassVar[list[Union[str, tuple[str, Callable[[Any], bool]]]]] = [
    # Each entry is either:
    #   - a string: attr name to cascade when child's value is None
    #   - a (name, condition_callable) tuple: cascade if condition(child_value) returns True
    #
    # The cascade walks direct children via _iter_child_inferencers (single canonical
    # iteration mechanism; cycle-safe). For each child:
    #   - String entry: cascade if getattr(child, name, MISSING) is None
    #   - Tuple entry:  cascade if condition(getattr(child, name, MISSING)) returns True
    #
    # The cascade is "inherit-when-unset" semantics — explicit values win, matching
    # the workspace-propagation precedent. Recursion is implicit: each child re-runs
    # its own _propagate_cascading_attrs_to_children after assignment.
    "debug_mode",
]


def _propagate_cascading_attributes(self) -> None:
    """Cascade attributes declared in ``_CASCADING_ATTRIBUTES`` to child inferencers.

    v2 — Built on ``_for_each_child_inferencer`` (the attrs-field walker),
    NOT ``_iter_child_inferencers`` (which is not implemented on CI/PTI/LWI).
    This matches the verified existing pattern of ``_propagate_workspace_to_children``
    (line 450), which also uses ``_for_each_child_inferencer`` at line 513.

    For each attr in ``_CASCADING_ATTRIBUTES``:
      - String entry: cascade when child's value is None
      - Tuple entry (name, condition): cascade when condition(parent_val, child_val) is True

    Walks direct children via ``_for_each_child_inferencer``; for each child
    that satisfies the cascade condition, set the attribute AND recurse into
    that child's own ``_propagate_cascading_attributes()`` to reach grandchildren.

    Called from ``__attrs_post_init__`` and from runtime-toggle hooks
    (``enable_debug_mode``/``disable_debug_mode`` overrides — see §E2.2).

    ``on_partial`` returns None so we don't replace ``functools.partial``
    factories; their instances inherit at the time the factory is invoked
    (mirrors workspace propagation's behavior).
    """
    for spec in self._CASCADING_ATTRIBUTES:
        if isinstance(spec, str):
            name = spec
            condition = lambda p, c: c is None
        else:
            name, condition = spec

        parent_val = getattr(self, name, None)
        if parent_val is None:
            # Parent itself is unset — nothing to cascade
            continue

        # Use closure variables; bind via default args to avoid late-binding
        def _on_instance(child, field_name, key,
                         _name=name, _pv=parent_val, _cond=condition):
            if not isinstance(child, InferencerBase):
                return
            try:
                child_val = getattr(child, _name, None)
            except AttributeError:
                return
            if _cond(_pv, child_val):
                try:
                    setattr(child, _name, _pv)
                    # Recurse into THIS child so the cascade reaches grandchildren
                    child._propagate_cascading_attributes()
                except Exception as e:
                    logger.warning(
                        "_CASCADING_ATTRIBUTES: failed to set %s.%s = %r: %s",
                        type(child).__name__, _name, _pv, e,
                    )

        def _on_partial(partial_obj, field_name, key):
            # Don't replace partials — factory children inherit at instantiation
            return None

        self._for_each_child_inferencer(_on_instance, _on_partial)
```

**v2 critical correction (the architectural fix):** The cascade walks `_for_each_child_inferencer`, not `_iter_child_inferencers`. Verified at `inferencer_base.py:513` that this is the exact same walker `_propagate_workspace_to_children` uses. This is what makes the cascade reach `ConversationalInferencer.base_inferencer` (an attrs field at `conversational_inferencer.py:108`), `PTI`'s children, and `LWI`'s children — none of which implement `_iter_child_inferencers`. **My v1's implementation would have silently failed for the primary use case (cascading from CI to its `base_inferencer`).**

**Hook into the THREE trigger sites (v2 — covers all the paths that actually fire):**

```python
# TRIGGER 1 — Construction-time (in InferencerBase.__attrs_post_init__,
# at the end after _propagate_workspace_to_children):
self._propagate_cascading_attributes()
# Covers: CI -> base_inferencer; Dual -> base/review/fixer; any attrs-field child.

# TRIGGER 2 — Runtime toggle (override Debuggable.enable_debug_mode /
# disable_debug_mode in InferencerBase, so runtime CI.enable_debug_mode()
# cascades to children):
def enable_debug_mode(self) -> None:
    super().enable_debug_mode()       # set self.debug_mode = True
    self._propagate_cascading_attributes()

def disable_debug_mode(self) -> None:
    super().disable_debug_mode()      # set self.debug_mode = False
    # No cascade — explicit False on child should win over explicit False
    # on parent (no information to add). Children with None stay None.

# TRIGGER 3 — Orchestrator runtime-created children (in
# TemplatedInferencerBase._propagate_to_children, ADD super() call that
# is currently missing — Plan B caught this):
def _propagate_to_children(self):
    super()._propagate_to_children()   # NEW — was missing; needed for cascade
    # ... existing template_extra_feed propagation ...
```

**Why these specific hook sites (verified):**
- `__attrs_post_init__`: catches all programmatic-build paths (test fixtures, dev tools, hand-construction). This is THE hook for `ConversationalInferencer` — it has no other propagation path, since `_ainfer` at `conversational_inferencer.py:1505` calls `self.base_inferencer.ainfer` directly without going through `_infer_single`/`_ainfer_single` (which is where `_propagate_to_children` would normally fire).
- `enable_debug_mode`/`disable_debug_mode` overrides: catches runtime toggle on already-constructed CI sessions. Per Plan B's verification, this is the ONLY hook for CI runtime changes — CI bypasses `_propagate_to_children` entirely.
- `super()._propagate_to_children()` in `TemplatedInferencerBase`: catches orchestrator-created runtime children (PTI's `_setup_child_workflows`, BTA workers). Plan B verified this `super()` call is currently MISSING in TemplatedInferencerBase — the base's `_propagate_to_children` (which after v2 cascades attrs) never fires for these inferencers. Adding the missing `super()` call closes that gap.

**Tests (T16–T22):**
- T16: Parent `Dual(debug_mode=True, base_inferencer=Leaf(), fixer_inferencer=Leaf())` → both children have `debug_mode == True` after construction.
- T17: Parent with `debug_mode=True`; child constructed with explicit `debug_mode=False` → child preserves `False` (explicit wins).
- T18: Parent with `debug_mode=None` (not set) → children's `debug_mode` is unchanged from their own values (no cascade from unset parent).
- T19: Tuple-form entry — register `("model_name", lambda v: not v)`; verify cascade fires when child has `model_name=""` or `model_name=None`.
- T20: Recursion through BTA — `BTA(debug_mode=True, worker_factory=lambda: Dual(base=Leaf(), fixer=Leaf()))` → after BTA dispatches workers, every worker AND every worker's child has `debug_mode == True`.
- T21: Recursion through MFDual `flow_configs` list-of-dicts — MFDual's `_iter_child_inferencers` yields each flow's root inferencer; cascade reaches every flow's root.
- T22: Cycle safety — if a future subclass accidentally yields its parent via `_iter_child_inferencers`, the id-based seen set in `pre_retry`-style walks (line 813) prevents infinite recursion. Our cascade uses direct iteration (one level), so cycles can't cascade through us; verify the no-recursion-loop test.

**Risk:** medium. Hooks run on every inferencer construction; defensive `try/except` around `setattr`; `AttributeError` swallow for children without the attr. Mitigation: T16–T22 cover the common shapes; existing inferencer test suite is the regression gate.

**LoC:** ~60 production + ~80 tests.

### §E2.3 — Commit 6: Pathway-1 tests + cross-pathway integration

**Purpose:** Lock the cross-pathway behavior — Pathway 1's runtime cascade + Pathway 2's YAML cascade should produce identical observable outcomes.

**Files added:**

1. `tests/integration/test_debug_mode_cascade_cross_pathway.py` (~50 LoC):

```python
# Scenario A: same CI built two ways
#   1. From YAML with _debug_mode: true
#   2. Programmatically as ConversationalInferencer(debug_mode=True, base=...)
#
# Walk both trees via _iter_child_inferencers; assert observable equivalence:
# every Debuggable in both trees has debug_mode == True.
#
# Scenario B: yolo for pathway 1 when YAML config also sets debug_mode
#   - Build via YAML with _debug_mode: false
#   - Programmatically override CI.debug_mode = True after construction
#   - Trigger _propagate_to_children
#   - Assert: cascade reaches children
```

**Tests (T23–T25):**
- T23: Scenario A — both build paths produce trees with identical `debug_mode` distribution.
- T24: Scenario B — runtime mutation + re-propagation reaches descendants.
- T25: Performance — building a CI tree with cascade enabled vs disabled — assert overhead < 5% (sanity check that the propagation loop isn't accidentally O(n²)).

**Risk:** low. Verification surface; no production code.

**LoC:** ~50 tests.

---

## §E3. Validation

### §E3.1 — Per-commit gates
- Commit 1: `instantiate(default.yaml, backend="ClaudeCodeCLI")` succeeds; T1–T3 pass.
- Commit 2: T4–T7 pass; existing OpenStartup factory tests still green.
- Commit 3: T8–T11 pass; manual smoke (§E3.2 D) demonstrates debug logs from nested inferencers.
- Commit 4: T12–T15 pass; existing `Debuggable` subclass tests still green.
- Commit 5: T16–T22 pass; existing AgentFoundation inferencer test suite still green.
- Commit 6: T23–T25 pass.

### §E3.2 — End-to-end smoke

```bash
# A. Pathway 2 — YAML cascade unit tests
cd /Users/tchen7/MyProjects/CoreProjects/AgentFoundation
pytest tests/integration/test_ci_yaml_debug_mode_cascade.py -v

# B. Pathway 2 — OpenStartup factory tests
cd /Users/tchen7/MyProjects/CoreProjects/OpenStartup
pytest tests/integration/test_factories_ci_yaml.py -v

# C. Pathway 1 — runtime cascade tests
cd /Users/tchen7/MyProjects/CoreProjects/AgentFoundation
pytest tests/common/inferencers/test_cascading_attributes.py -v

# D. Manual smoke (Pathway 2 end-to-end)
OPENSTARTUP_DEBUG_MODE=true python -m openteam.server.main &
# In UI: chat "Build a Python hello-world script using the task tool"
# Expected log signature:
#   [DEBUG] ConversationalInferencer: ...
#   [DEBUG] RovoDevCliInferencer (or ClaudeCodeCLI): ...   ← child inherited debug_mode
#   [DEBUG] task tool dispatcher: ...
#   [DEBUG] BTA inferencer (inside task): ...              ← grandchild via YAML cascade
#   [DEBUG] Leaf workers: ...                              ← grand-grandchild

# E. Cross-pathway equivalence (after BOTH pathways land)
pytest tests/integration/test_debug_mode_cascade_cross_pathway.py -v
```

### §E3.3 — CHANGELOG entries

```
# AgentFoundation CHANGELOG:
### Added
- `src/agent_foundation/resources/configs/conversational/default.yaml` + backend
  variants — the canonical default CI config. Operators can override via
  OpenStartup's deployment-level YAML or per-session `overrides=` arg.
- `InferencerBase._CASCADING_ATTRIBUTES` + `_propagate_cascading_attrs_to_children()`:
  declarative attribute cascade from parent to children whose values are unset.
  Mirrors the existing workspace-propagation pattern.

### Changed
- `_propagate_to_children` now calls `_propagate_cascading_attrs_to_children` in
  addition to its previous (overridden) behavior. Subclasses that override
  `_propagate_to_children` SHOULD call `super()._propagate_to_children()` first
  to retain the cascade.

# RichPythonUtils CHANGELOG:
### Changed
- `Debuggable.debug_mode` type signature: `bool` → `Optional[bool]` (default
  changed from `False` to `None`). `None` is treated as `False` in boolean
  contexts (no semantic change for existing `if self.debug_mode:` callers).
  Enables runtime cascade from parent inferencers (see AgentFoundation
  `_CASCADING_ATTRIBUTES`).

# OpenStartup CHANGELOG:
### Changed
- `factories.build_conversational_inferencer` now loads its CI shape from the
  canonical AgentFoundation default YAML (or OpenStartup's deployment-level
  override if present). Set `_debug_mode: true` in the YAML OR pass
  `OPENSTARTUP_DEBUG_MODE=true` for verbose runs.
```

---

## §E4. Execution checklist

```
[ ] Pre-flight (5 min)
[ ]   git status — confirm clean tree on dev_xinli_2601 in 3 repos
[ ]   grep "_CASCADING_ATTRIBUTES\|_propagate_cascading_attrs_to_children" src/
        — verify zero hits (sanity: plan not stale)
[ ]   ls src/agent_foundation/resources/configs/conversational/ — verify
        directory does NOT exist yet (sanity: plan is fresh)

Pathway 2 — Commit 1: AgentFoundation conversational/default.yaml + backends
[ ] NEW   src/agent_foundation/resources/configs/conversational/default.yaml
[ ] NEW   src/agent_foundation/resources/configs/conversational/base_inferencer/ClaudeCodeCLI.yaml
[ ] NEW   src/agent_foundation/resources/configs/conversational/base_inferencer/RovoDevCLI.yaml
[ ] NEW   tests/integration/test_ci_yaml_instantiate.py (T1–T3)
[ ] Tests + lint  → commit "feat(configs): conversational/default.yaml + backends"

Pathway 2 — Commit 2: OpenStartup factories.py YAML-driven CI
[ ] Edit  OpenStartup/.../factories.py — switch to build_ci_from_config
[ ] NEW   OpenStartup/.../server/resources/configs/conversational/default.yaml (optional)
[ ] NEW   OpenStartup/tests/integration/test_factories_ci_yaml.py (T4–T7)
[ ] Tests + lint  → commit "feat(factories): YAML-driven conversational inferencer"

Pathway 2 — Commit 3: Integration smoke + regression
[ ] NEW   AgentFoundation/tests/integration/test_ci_yaml_debug_mode_cascade.py (T8–T10)
[ ] Manual smoke per §E3.2 step D
[ ] Tests + lint  → commit "test(integration): debug_mode YAML cascade"

PATHWAY 1 (ships AFTER Pathway 2 lands and is stable)

Pathway 1 — Commit 4: RichPythonUtils Debuggable Optional[bool]
[ ] Edit  RichPythonUtils/.../debuggable.py — line 228 signature change
[ ] NEW   RichPythonUtils/tests/test_debuggable_optional_bool.py (T12–T15)
[ ] Grep all 4 existing `debug_mode:` annotation sites — verify back-compat
[ ] Tests + lint  → commit "refactor(debuggable): debug_mode → Optional[bool]"

Pathway 1 — Commit 5: AgentFoundation cascade mechanism
[ ] Edit  inferencer_base.py — add _CASCADING_ATTRIBUTES + method
[ ] Edit  inferencer_base.py — hook into __attrs_post_init__ + _propagate_to_children
[ ] NEW   tests/common/inferencers/test_cascading_attributes.py (T16–T22)
[ ] Tests + lint  → commit "feat(inferencer-base): _CASCADING_ATTRIBUTES mechanism"

Pathway 1 — Commit 6: Cross-pathway integration
[ ] NEW   tests/integration/test_debug_mode_cascade_cross_pathway.py (T23–T25)
[ ] Write CHANGELOG entries per §E3.3 in all 3 repos
[ ] Run full pytest in 3 repos
[ ] git push origin dev_xinli_2601 (3 repos)
[ ] Update _docs/_plan/README.md index with this plan
```

---

# PART II — DESIGN REFERENCE
══════════════════════════════════════════════════════════════════════════════

## §D1. Goals & non-goals

**Goals:**
1. Make `debug_mode` cascadable from a parent (especially the conversational inferencer) to its children.
2. Use a principled, extensible mechanism (`_CASCADING_ATTRIBUTES`) so future cross-cutting attributes (e.g. `model_name` overrides, `tracing_enabled`, etc.) can be added with a one-line registration — not a new propagation method per attribute.
3. Move the conversational inferencer's construction from imperative hand-built code into declarative YAML — both as a quick-win for the cascade goal AND as the foundation for `task --config conversational` (from the companion `task_complexity_presets_and_chat_peer_plan`).
4. Honor the verified existing pattern: respect explicit pre-assignment by the caller (workspace-propagation precedent at `inferencer_base.py:450`).
5. Zero regression in existing CI/inferencer test suite; zero forced behavior change for existing `Debuggable` callers (`if self.debug_mode:` keeps working unchanged).

**Non-goals:**
1. **NOT** changing `debug_mode`'s default from `False` to `True`. Operator decision per session.
2. **NOT** auto-cascading `model_name` runtime (rejected — see §D2.3).
3. **NOT** unifying `_for_each_child_inferencer` and `_iter_child_inferencers` — "deliberately not unified" per `inferencer_base.py:756`.
4. **NOT** building generic config-discovery service or making everything configurable.
5. **NOT** changing the verified YAML cascade behavior of `_instantiate.py` — Pathway 2 uses it as-is.

## §D2. Architecture decisions

### §D2.1 Why a class-level list (not per-attr setter override)

Alternative considered: a setter on every attribute (e.g. `@debug_mode.setter` that recursively assigns to children). Rejected for two reasons:

| Reason | Detail |
|---|---|
| **N attrs → N setters → N opportunities for bugs** | Each new cross-cutting attribute would require a custom setter on every container class. With `_CASCADING_ATTRIBUTES`, adding a new attr is one line of registration. |
| **Mismatched with the verified existing pattern** | `_propagate_workspace_to_children` is a single method that handles workspace propagation for ALL inferencer subclasses uniformly. The cascade mechanism mirrors that exact shape. |

### §D2.2 Why `None`-as-inherit (not a separate `_inherits_debug_mode: bool` flag)

Alternative considered: keep `debug_mode: bool = False` and add a separate `_inherits_debug_mode: bool = True` flag. Rejected because:

- **API surface area grows** — every consumer would have to know about both fields.
- **`None` as the "unset" sentinel is the Pythonic idiom** — already used by `cache_folder: Optional[Path] = None`, `model_name: Optional[str] = None`, etc. throughout the codebase.
- **Verified that no existing call site of `debug_mode` distinguishes `False` from "unset"** — the 4 hits in `Debuggable` itself all use `if self.debug_mode:` semantics, which work identically with `None` or `False`.

### §D2.3 Why `model_name` is NOT in the default `_CASCADING_ATTRIBUTES` list

The user's design discussion proposed `model_name` alongside `debug_mode`. After source verification, `model_name` was rejected from v1's default cascade list:

- **Already cascaded at YAML config-instantiation time** via `_model_name:` underscore-prefix injectable (verified in `breakdown-multiflow-plan.yaml`).
- **Runtime cascade risks silently overriding intentional per-leaf model choices** (e.g. a leaf inferencer is deliberately built with `model_name="haiku"` for a fast pre-classification stage; runtime parent-cascade with `opus` would silently override).
- **The mechanism is fully extensible** — any consumer who wants `model_name` runtime cascade can subclass `InferencerBase` (or override `_CASCADING_ATTRIBUTES` on a specific orchestrator) and add it. v1 ships only `debug_mode` in the default list to minimize surface.

Filed as Follow-up #5 if a real use case emerges.

### §D2.4 Why a new `resources/configs/` directory (not extending `resources/tools/`)

The verified existing convention has 2 instances:
- `src/agent_foundation/resources/tools/sop/configs/default.yaml`
- `src/agent_foundation/resources/tools/task/configs/default.yaml`

Both are per-tool inferencer configs. The conversational inferencer is fundamentally different — it's not a tool, it WRAPS tools. Putting it under `tools/conversational/configs/` would be a category error (it would imply a `conversational` tool exists). The asymmetry matters; `resources/configs/conversational/` makes the kind explicit.

If a third "non-tool" inferencer config emerges later (e.g. a `chat` peer tool from the companion plan), it lands at `resources/configs/chat/` — establishing a clear "this is a deployable inferencer topology" namespace.

### §D2.5 Why hook at BOTH `__attrs_post_init__` AND `_propagate_to_children`

- `__attrs_post_init__`: catches programmatic construction paths (test fixtures, dev tools, hand-construction).
- `_propagate_to_children`: catches runtime re-propagation (parent's hook fires after a state change; e.g. `template_extra_feed` propagation).

Both are needed because programmatic-build paths don't trigger `_propagate_to_children` unless something else does (workspace assignment, etc.). The cascade should be idempotent — running it twice is harmless (parent_val is the same; children already match; no-op).

## §D3. The cascade contract — formal spec

```
For each entry in self._CASCADING_ATTRIBUTES:
    name, should_cascade = (entry, lambda v: v is None) if isinstance(entry, str) else entry
    parent_val = getattr(self, name, None)
    if parent_val is None: continue    # parent unset → nothing to cascade

    For each direct child in self._iter_child_inferencers():
        child_val = getattr(child, name, MISSING)
        if child_val is MISSING: continue   # child doesn't have this attr
        if should_cascade(child_val):
            setattr(child, name, parent_val)
            # Child's own __attrs_post_init__ or setter triggers
            # recursive cascade to ITS children.
```

**Invariants:**
1. **Explicit values win.** A child constructed with `debug_mode=False` keeps `False` even if parent has `True`.
2. **Unset propagates.** A child with `debug_mode=None` inherits the parent's value (`None` is the sentinel).
3. **No cascade from unset parent.** If parent itself has `parent_val is None`, no cascade fires — preserves child's own default.
4. **Recursion is implicit, one level at a time.** Each child re-runs its own cascade on its own descendants when its setter or `__attrs_post_init__` fires.
5. **Errors are best-effort.** `AttributeError` on missing attr is silenced; `Exception` on setattr is logged at WARNING, never raised.

## §D4. Risk register + open questions

### Risks

| ID | Risk | Mitigation |
|---|---|---|
| **R1** | `Optional[bool]` change to `Debuggable` in RichPythonUtils breaks an unknown external consumer that does `if obj.debug_mode is False:` (explicit-set detection) | Grep verified `if obj.debug_mode is False` returns zero hits in known repos (AgentFoundation, OpenStartup, rankevolve, etc.). Cross-repo back-compat sweep before push. |
| **R2** | The `_propagate_to_children` hook accidentally cascades to a child that was deliberately constructed without `debug_mode` attr at all | The `try/except AttributeError` swallows this case silently — child unaffected. T16 + T22 verify. |
| **R3** | Pathway 2's YAML cascade and Pathway 1's runtime cascade produce different outcomes for the same logical config | Cross-pathway test T23 specifically asserts equivalence. |
| **R4** | OpenStartup factory rewrite introduces a regression in session bootstrap | T7 (existing CI runtime tests still pass) + manual smoke (§E3.2 D). Behavioral regression gate before push. |
| **R5** | The `Conversational` `_target_` token doesn't currently resolve to `ConversationalInferencer` in the YAML registry | Pre-flight grep verification before Commit 1 — `grep "Conversational" src/agent_foundation/common/inferencers/conversational/conversational_inferencer.py` should show `@register("Conversational")` or equivalent. If it doesn't exist, add it as a tiny prerequisite. |
| **R6** | `build_ci_from_config(...)` helper doesn't exist yet in OpenStartup | Implementation is small (~20 LoC: load YAML → `_instantiate.instantiate(...)` → return root); piggybacks on the existing `_ci_host.py` helper from the chat-peer-tool plan. If `_ci_host.py` doesn't have a `build_ci_from_config`, add it as part of Commit 2. |
| **R7** | The `Callable` import for the type annotation isn't already in `inferencer_base.py` | Trivial — add `from collections.abc import Callable` at top of file. |
| **R8** | Performance regression — `__attrs_post_init__` now runs the cascade loop for every inferencer construction | T25 perf gate (overhead < 5%). The loop is `len(_CASCADING_ATTRIBUTES) × len(direct_children)` — both small constants. |

### Open questions + defaults

| Q | Question | Default for v1 |
|---|---|---|
| Q1 | Should `_CASCADING_ATTRIBUTES` be inherited and extended (subclass adds entries) or replaced wholesale? | **Inherited + extended.** Idiomatic Python `ClassVar` inheritance; subclass can do `_CASCADING_ATTRIBUTES = InferencerBase._CASCADING_ATTRIBUTES + [("custom_attr", lambda v: v == "")]`. |
| Q2 | Should the cascade fire from `_propagate_workspace_to_children` too (same scope as workspace) | **No** — workspace cascade and cascading-attrs cascade are separate concerns; coupling them would make cascade-on-workspace-change a side effect. Keep `__attrs_post_init__` + `_propagate_to_children` only. |
| Q3 | Should we add a way to opt-out at the parent level (skip cascading for a specific attr)? | **No (v1)** — symmetric with `_workspace_propagation_skip` (`inferencer_base.py:471`) is a natural extension if needed. Filed as Follow-up #2. |
| Q4 | Should v1 include `model_name` in the default list? | **No** — see §D2.3. Filed as Follow-up #5. |
| Q5 | Should the OpenStartup deployment-level YAML be required, optional, or absent in v1? | **Optional (Commit 2b).** Framework default suffices; deployment-level only if app-specific customizations are needed. |
| Q6 | Should `build_ci_from_config` live in AgentFoundation or OpenStartup? | **AgentFoundation** (general-purpose). OpenStartup's `factories.py` is the consumer; the helper is reusable. |
| Q7 | Should the new `resources/configs/` directory have a `README.md` explaining the convention? | **Yes (small)** — `_docs/_plan/.../README.md` references it from this plan, but a top-level `resources/configs/README.md` documenting the "deployable inferencer topologies" convention is cheap and prevents future confusion. |
| Q8 | Should Pathway 1 ship before Pathway 2 if Pathway 2 hits a blocker? | **Either order works.** They're independent. v1 documents Pathway 2 first because lower risk + immediate value, but the order can be flipped if needed. |

---

# APPENDIX — AUDIT TRAIL
══════════════════════════════════════════════════════════════════════════════

## §A1. Motivation

This plan was motivated by an in-conversation design session on 2026-06-14 with the user (Tony). The user's intuition was correct end-to-end:

1. **"`debug_mode` default is `None`, treated as `False` in `if`/`else`."** Verified that the existing `if self.debug_mode:` semantics at `debuggable.py:922` already work with `None` (falsy). The change to `Optional[bool]` is back-compat by Python's truthy/falsy rules.
2. **"If set explicitly at construction time, it's not subject to cascade."** This is the exact invariant the `_propagate_workspace_to_children` precedent established (`inferencer_base.py:450`). v1 mirrors it for `_CASCADING_ATTRIBUTES`.
3. **"InferencerBase already has existing attribute cascading logic."** Verified — `_propagate_workspace_to_children` (workspace), `_propagate_to_children` (template_extra_feed), and the underscore-prefix YAML cascade in `_instantiate.py` are three already-working mechanisms. v1 adds a fourth that handles the general case.
4. **"Can we have an `attributes_to_propagate` list with optional condition callable?"** Yes — that's exactly what `_CASCADING_ATTRIBUTES` is. String for "cascade when None"; tuple for "cascade if condition(child_value) is True".
5. **"Default conversational YAML under AgentFoundation resources, customized under OpenStartup."** Yes — Pathway 2's 3-tier discovery (explicit → deployment override → framework default) gives operators the right escape hatches.

All five intuitions are correct and source-verified in §A2.

## §A2. Verified facts (load-bearing for this plan)

Source: AgentFoundation `dev_xinli_2601`, RichPythonUtils `dev_xinli_2601`, OpenStartup `dev_xinli_2601`, verified 2026-06-14 18:22–18:24.

| # | Fact | Source |
|---|---|---|
| F1 | `Debuggable.debug_mode` is `bool = attrib(default=False, kw_only=True)` today | `RichPythonUtils/.../debuggable.py:228` |
| F2 | `Debuggable` uses `if self.debug_mode:` (truthy/falsy semantics — `None` evaluates as `False`) | `RichPythonUtils/.../debuggable.py:922` |
| F3 | `InferencerBase._propagate_workspace_to_children` exists and uses the "respect explicit pre-assignment" pattern (the precedent v1 mirrors) | `inferencer_base.py:450` |
| F4 | `InferencerBase._propagate_to_children` is overridable; default is no-op; called from `_infer_single`/`_ainfer_single` at line 1253 | `inferencer_base.py:733, 1253` |
| F5 | `InferencerBase._iter_child_inferencers` is the canonical recursion mechanism (line 764), used by lifecycle/cleanup paths | `inferencer_base.py:764` |
| F6 | `_propagate_workspace_to_children` notes that `_for_each_child_inferencer` and `_iter_child_inferencers` are "deliberately not unified" | `inferencer_base.py:756` |
| F7 | The YAML underscore-prefix cascade (`_debug_mode: true` → injects into child `_target_:` nodes if accepted) is implemented at `_instantiate.py:718-725` | `rich_python_utils/config_utils/_instantiate.py:718-725` |
| F8 | The "per-tool inferencer config" convention exists for `sop` and `task` only | `src/agent_foundation/resources/tools/{sop,task}/configs/default.yaml` |
| F9 | `breakdown-multiflow-plan.yaml` is a real working example of `_debug_mode: true` at YAML root with nested inferencers inheriting it (verified in earlier session) | (referenced from companion plan) |
| F10 | OpenStartup `factories.py` constructs CI imperatively today (no YAML-driven path) — this is what Pathway 2 replaces | `OpenStartup/src/openteam/server/backends/factories.py` |

**Verification methodology:** every claim above was confirmed with a targeted grep against the actual `dev_xinli_2601` branch source. No claim is from extrapolation or pattern-matching.

## §A3. Out-of-scope follow-ups

1. **Add a `resources/configs/README.md`** documenting the "deployable inferencer topology" convention. Pure docs.
2. **Add `_cascading_attributes_propagation_skip: frozenset`** symmetric with `_workspace_propagation_skip` — for orchestrators that want to opt-out of cascade for specific attrs on specific children. Defer until a real use case.
3. **Add `tracing_enabled` to the default `_CASCADING_ATTRIBUTES` list** once tracing infrastructure exists. Lightweight follow-up.
4. **Migrate other operator-level attrs (`log_level`, `verbose`, etc.) to the cascade** as they emerge.
5. **Conditional cascade for `model_name`** — see §D2.3. Only if a real use case demands runtime model cascade beyond YAML-time cascade.
6. **`build_ci_from_config` helper extracted into a shared `_ci_host.py`** with `sop`/`task` peer config loaders. Refactor; defer until a third consumer emerges.
7. **Per-session CLI override** (`--debug-mode true` flag on OpenStartup server) that flips `_debug_mode` in the YAML overrides dict. Minor UX.
8. **OpenStartup CHANGELOG sweep** for any other hand-constructed inferencer paths that should also move to YAML-driven config.

## §A4. Changelog

- **v1 (2026-06-14 18:26):** Initial draft. Covers two coupled-but-independent pathways for debug-mode cascade: Pathway 2 (CI default YAML — ships first, lower risk, immediate value) + Pathway 1 (code-level `_CASCADING_ATTRIBUTES` mechanism — ships second, proper long-term). 6 commits across 3 repos (AgentFoundation primary, OpenStartup factory rewrite, RichPythonUtils `Debuggable` signature change), ~280 LoC production + ~270 LoC tests, ~2-day effort. All 10 load-bearing facts verified against source before draft (§A2). Honest documentation of (a) why `model_name` is NOT in the default cascade list (§D2.3), (b) why a NEW `resources/configs/` directory is the right place (§D2.4), (c) the cross-pathway equivalence invariant (§D3 + T23). Companion to `interactive_widget_for_agent_dispatched_tools_plan.md` (same folder) and `task_complexity_presets_and_chat_peer_plan.md` (consumer of Pathway 2's CI default YAML for `--config conversational` mode).

---

## §A5. Cross-plan audit (v2 NEW)

v2 integrates two peer plans against my v1. Honest record of what each caught:

| Issue | v1 (mine) | Plan B (Cursor `debug_mode_cascade_and_ci_config_82c8b750.plan.md`) | Plan C (Claude `update-your-task-tool-adaptive-goose.md`) | v2 (integrated) |
|---|---|---|---|---|
| **Cascade method choice** | ❌ Used `_iter_child_inferencers` — NOT implemented on CI/PTI/LWI; cascade silently fails for primary use case | ✅ Caught — used `_for_each_child_inferencer` (attrs walker, verified) | ✅ Caught — same choice as Plan B | ✅ Fixed (v2 uses `_for_each_child_inferencer`) |
| **Trigger sites** | ❌ Only 2 (`__attrs_post_init__` + base `_propagate_to_children`) — missed runtime-toggle hook for CI | ✅ 3 triggers with verified path-coverage map (post_init / enable_debug_mode override / `super()` in TemplatedInferencerBase) | ✅ 2 triggers (post_init + enable/disable override) | ✅ All 3 triggers (v2 adopts Plan B's verified path map) |
| **Missing `super()._propagate_to_children()` in TemplatedInferencerBase** | ❌ Missed entirely | ✅ Caught and documented as a separate fix needed for cascade to reach orchestrator runtime children | ❌ Missed entirely | ✅ Added to v2 Trigger 3 |
| **One-line factory quick-win** | ❌ Missed — went straight to principled mechanism | ✅ Caught — `debug_mode=True` in `factories.py:167` ships independent | ✅ Caught — same one-line | ✅ Added as Pathway 0 (§E0) |
| **`prompt_renderer` runtime-passed, not YAML-declared** | ⚠ Hand-waved as a follow-up | ✅ Caught — verified that `_filter_tools_by_config` runs BEFORE the CI is built; renderer cannot be YAML-declared | ✅ Caught — same finding | ✅ Added to v2 §E1.2 |
| **OpenStartup YAML declares `prompt_renderer` (app-specific)** | ❌ Missed | ✅ Caught — main-chat prompt path is OpenStartup-specific, not framework default | ✅ Caught — same finding | ✅ Added to v2 §E1.2c |
| **CI bypasses `_propagate_to_children` entirely** (architectural fact, not a bug) | ❌ Missed — would have inserted cascade in a hook that never fires for CI | ✅ Caught with file:line citation (`conversational_inferencer.py:1505`) | ✅ Caught | ✅ Documented in v2 §D2.5 |
| **`Optional[bool]` change is back-compat** | ✅ Caught (debuggable.py:922 truthy check) | ✅ Caught (debuggable.py:743 same check) | ✅ Caught | ✅ Documented in §A2 F2 |
| **Sweep `debug_mode is False / == False`** | ❌ Listed as follow-up | ✅ Caught — mandatory pre-PR sweep | ✅ Caught — same | ✅ Added to v2 §E3 checklist |
| **Verified `_for_each_child_inferencer` discovers `base_inferencer` attrs field** | ❌ Not verified | ✅ Verified (`conversational_inferencer.py:108`) | ✅ Verified | ✅ Documented in §A2 F11 |
| **YAML cascade test-proven by `test_task_real_cli.py:538`** | ❌ Not cited | ✅ Cited as proof of YAML cascade behavior | ❌ Missed citation | ✅ Added to §A2 F12 |
| **Structural rigor** (PART I/II/APPENDIX, risk register, named tests T1–T25, open questions, follow-up list) | ✅ Yes | ⚠ Partial — design rigor but condensed; no PART I/II/APPENDIX split | ⚠ Less — no risk register or open-questions table | ✅ v2 keeps v1's structure, integrates peer findings |

**Score:** Plan B caught the most architectural defects (6/8); Plan C confirmed them independently (validating Plan B); my v1 had the worst architectural correctness (only got the structural rigor right). v2 = v1's structure + Plan B's verified architecture + Plan C's confirmation.

## §A6. If forced to pick ONE plan as-is, which?

**Plan B (Cursor `debug_mode_cascade_and_ci_config_82c8b750.plan.md`)** — without any question.

**Reasons:**

1. **Plan B is the only plan that gets the cascade architecture right** at v1-time. It correctly chose `_for_each_child_inferencer` (verified by grep). Plan C arrived at the same conclusion independently, validating Plan B. My v1 was architecturally wrong.

2. **Plan B has the most verified facts.** Every key claim has a file:line citation. The "What is true today" section is essentially an evidence appendix in itself. The trigger-site map is verified at each step. The `test_task_real_cli.py:538` citation is a particularly strong artifact — it proves the YAML cascade behavior is already test-locked.

3. **Plan B catches the `super()._propagate_to_children()` missing call in TemplatedInferencerBase.** This is a real bug in the existing codebase (not just a gap in my v1) — Plan B documented it with file:line evidence; Plan C and my v1 missed it entirely.

4. **Plan B's MVP discipline.** It includes the one-line factory quick-win as a viable pathway, recognizes the YAML refactor needs `prompt_renderer` runtime-passed, and scopes the YAML refactor to acknowledge what cannot be declarative. My v1 over-engineered (full YAML refactor without the quick-win); Plan C did not include the quick-win either.

5. **Plan B is 169 lines, dense, every line is necessary.** v1 was 728 lines; v2 is 818 lines. Plan B does more with less because every claim is verified — no padding.

**Caveats (what Plan B doesn't have):**
- No PART I/II/APPENDIX 3-tier structure (just flat sections).
- No named-test list (T1, T2, ...) — sketches tests in prose.
- No risk register table or open-questions table.
- No cross-plan audit or if-forced-to-pick (because it's the canonical artifact, not the integrator).

**If you want both — pick v2.** v2 keeps v1's structural rigor (3-tier organization, named tests T1–T25, risk register R1–R8, open questions Q1–Q8, follow-up list) AND absorbs Plan B's verified architecture + Plan C's confirmation. v2 is the canonical artifact going forward; Plan B is the right pick only if forced to choose among the three input artifacts as-is.

---

## §A4 changelog (v2 update)

- **v2 (2026-06-14 18:31):** Critical correction after cross-plan review. v1's cascade was built on `_iter_child_inferencers` (NOT implemented on CI/PTI/LWI; silently fails for the primary use case). v2 rebuilds it on `_for_each_child_inferencer` (matches verified `_propagate_workspace_to_children` precedent at `inferencer_base.py:513`). Adds Pathway 0 (§E0) one-line factory quick-win that ships independent of the cascade. Adds 3rd trigger site (`enable_debug_mode`/`disable_debug_mode` overrides) for runtime toggle. Adds `super()._propagate_to_children()` fix in TemplatedInferencerBase (Plan B caught a missing super call). Adds `prompt_renderer` runtime-passed requirement (cannot be YAML-declared due to `_filter_tools_by_config` running before CI build). Adds §A5 cross-plan audit table and §A6 if-forced-to-pick (Plan B wins among the 3 input plans as-is). Updates §A2 with 2 new verified facts (F11: attrs field discovery; F12: YAML cascade test artifact). Backup of v1 saved at `.debug_mode_cascade_and_ci_default_yaml_plan.v1.bak` (728 lines, recoverable).
- **v1 (2026-06-14 18:27):** Initial draft. See backup for original content.

---

**End of plan v2.**
