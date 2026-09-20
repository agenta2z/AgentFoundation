# Inferencer Axes Refactor — Integrated v4 Plan

**Author:** Tony Chen (integrating Rovo Dev v3 + Claude's updated integrated plan)
**Date drafted:** 2026-05-16 07:32
**Status:** Ready for review and implementation
**Supersedes:**
- `inferencer_axes_INTEGRATED_v3_plan.md` (Rovo Dev v3, 786 lines)
- `/Users/tchen7/.claude/plans/let-s-create-an-integrated-lively-pearl.md` (Claude updated, 292 lines)

> **Why v4 exists.** Both v3 and Claude's plan converged on the same architecture (decouple SIB/TIB from TemplatedIB; add 3 MI convenience classes; guarded clobber). Round-2 critical reading found Claude's plan made **one design constraint explicit that v3 left implicit**: `target_path` MUST default to `None` (never `os.getcwd()`) for the orchestrator-safety guard to work. v4 elevates that constraint, adopts Claude's `target_path or working_dir` migration pattern for RovoDev, adds Claude's Deferred Work section, and **fixes one new gap both plans missed: a potential `pre_exec_scripts` double-execution risk in Phase 4 that needs explicit reconciliation.**

> **If forced to pick ONE plan today:** **Claude's updated plan**, because the "target_path defaults to None — load-bearing!" insight is the right primary design constraint, and the prose is tighter. My v3 has better operational scaffolding (risk registers, rollback matrix, slot test code, DO-NOT-DELETE markers) but those are additions on top of the right architecture. v4 below combines both.

---

## 1. Target architecture

```
InferencerBase                                              — generic / orchestrators
├── TemplatedInferencerBase                                 — template rendering (standalone axis)
├── StreamingInferencerBase(InferencerBase)                 — streaming + recovery cache (DECOUPLED)
├── TerminalInferencerBase(InferencerBase)                  — subprocess + target_path (DECOUPLED)
│
├── TerminalSessionInferencerBase(TIB, SIB)                 — terminal + streaming  (MI diamond #1)
├── TerminalTemplatedInferencerBase(TIB, TemplatedIB)       — terminal + templates  (MI diamond #2)
├── TerminalSessionTemplatedInferencerBase(TSIB, TemplatedIB) — all three            (MI diamond #3)
│
│   Concrete CLI leaves (templated):
├── ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase)
├── KiroCliInferencer(TerminalSessionTemplatedInferencerBase)
├── DevmateCliInferencer(TerminalSessionTemplatedInferencerBase)
├── RovoDevCliInferencer(TerminalSessionTemplatedInferencerBase)
│
│   Concrete CLI leaf (no templates needed):
├── MetamateCliInferencer(TerminalSessionInferencerBase)
│
│   Concrete SDK/API/Tool leaves (streaming-only, no subprocess, no templates):
├── ClaudeCodeSdkInferencer(StreamingInferencerBase)
├── DevmateSDKInferencer(StreamingInferencerBase)
├── MetamateSDKInferencer(StreamingInferencerBase)
├── OpenClawInferencer(StreamingInferencerBase)
├── PlugboardApiInferencer(StreamingInferencerBase)
├── AgClaudeApiInferencer(StreamingInferencerBase)
├── RovoChatInferencer(StreamingInferencerBase)
├── RovoDevServeInferencer(StreamingInferencerBase)
├── ToolAsInferencer(StreamingInferencerBase)
│
└── Orchestrators(InferencerBase)                            — unchanged (BTA, MFDual, LWI, Dual, etc.)
```

**MROs (C3-verified):**
- `TerminalSessionTemplatedInferencerBase` → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC
- `TerminalTemplatedInferencerBase` → TIB → TemplatedIB → IB → Debuggable → Resumable → ABC
- `TerminalSessionInferencerBase` → TIB → SIB → IB → Debuggable → Resumable → ABC

---

## 2. The path contract — three roles, one writer each

| Field | Role | Lives on | Field-level default | Resolved by |
|---|---|---|---|---|
| `workspace` (`Optional[InferencerWorkspace]`) | Artifact storage (outputs, logs, cache_folder) | `InferencerBase` (existing) | `None` | User opt-in |
| `target_path` (`Optional[str]`) | CLI agent operating directory (the repo being worked on) | **`TerminalInferencerBase` (NEW)** | **`None` — NEVER `os.getcwd()`** | Leaf's `__attrs_post_init__` may set leaf-specific default (e.g., `~/fbsource` for Claude/Devmate) |
| `working_dir` (`str`) | Subprocess `cwd=` for `subprocess.run` / `asyncio.create_subprocess_*` | `TerminalInferencerBase` (existing; deduplicated from TSIB) | `None` | TIB.`__attrs_post_init__`: `working_dir = target_path or os.getcwd()` |

### 2.1 ⚠️ Load-bearing design constraint: `target_path` defaults to `None`, never `os.getcwd()`

This is the **single most important constraint** in the entire refactor. Spell it out at every layer:

- **TIB field declaration:** `target_path: Optional[str] = attrib(default=None)`
- **TIB `__attrs_post_init__`:** does NOT default `target_path`. Only defaults `working_dir` from `target_path or os.getcwd()`.
- **Leaf-level defaulting (Claude/Devmate `~/fbsource`):** runs *before* `super().__attrs_post_init__()`. Leaf is free to set `target_path` to a leaf-specific default. Leaves that don't set it (Kiro, RovoDev) intentionally leave it `None`.

**Why this is load-bearing:** the `_configure_for_workspace` guard discriminates "user wanted explicit target" from "no explicit target" using `target_path is None`. If TIB ever defaulted `target_path` to `os.getcwd()`, then:
- Standalone construction: `target_path == os.getcwd()` (non-None) → guard says "user-explicit" → clobber skips → `working_dir == os.getcwd()` → orchestrator scenario broken (cwd is wherever the framework was launched, not the per-iteration workspace).
- Specifically, BTA-spawned children would launch in the framework startup cwd instead of `parent_ws.child(...)`. Hidden runtime failure mode.

The constraint is enforced by:
1. Field-level default in TIB (`default=None`).
2. A permanent regression test (§12): `test_target_path_field_default_is_None` — asserts `attr.fields(TerminalInferencerBase).target_path.default is None`.
3. Comment in TIB's `__attrs_post_init__` and the field docstring stating "DO NOT default this to os.getcwd() — see plan §2.1."

### 2.2 The default-resolution algorithm

In `TerminalInferencerBase.__attrs_post_init__`:

```python
def __attrs_post_init__(self):
    """Resolve working_dir from target_path; do NOT default target_path itself.

    Order matters:
      1. Leaf's own __attrs_post_init__ has already run (it may have set
         target_path to a leaf-specific default like ~/fbsource; or left
         it None on purpose, e.g., Kiro and RovoDev).
      2. We default working_dir = target_path if set, else os.getcwd().
         target_path itself stays None if the leaf didn't set it — this
         is load-bearing for the _configure_for_workspace guard (§2.1).
      3. super() runs InferencerBase.__attrs_post_init__ → triggers
         _configure_for_workspace(workspace) IF workspace was provided.
         The guard (§3) only clobbers working_dir when target_path is None.
    """
    if self.working_dir is None:
        self.working_dir = self.target_path or os.getcwd()
    super().__attrs_post_init__()
```

### 2.3 `_configure_for_workspace` after Phase 1

```python
def _configure_for_workspace(self, workspace):
    import os
    if hasattr(self, "working_dir"):
        # Guarded clobber: auto-set working_dir from workspace.root ONLY
        # when no explicit target_path. Required for orchestrator-spawned
        # children (target_path=None → subprocess MUST use workspace.root
        # as cwd). When user supplies target_path, leave working_dir alone.
        target = getattr(self, "target_path", None)
        if target is None:
            new_wd = str(workspace.root)
            # Windows CreateProcessW enforces MAX_PATH=260 on lpCurrentDirectory
            if sys.platform != "win32" or len(new_wd) < 240:
                self.working_dir = new_wd
    if hasattr(self, "cache_folder"):
        self.cache_folder = os.path.join(
            str(workspace.root), "_runtime", "inferencer_cache"
        )
    # ...logger redirection unchanged...
```

### 2.4 Correctness table (all four scenarios)

| Scenario | `target_path` after TIB post-init | `working_dir` after TIB post-init | After `_configure_for_workspace` (if workspace) |
|---|---|---|---|
| Standalone, explicit `target_path="/repo"` | `/repo` | `/repo` | `/repo` (guard sees target non-None → skip) ✅ |
| Standalone, explicit `working_dir="/explicit"`, no target_path | None | `/explicit` (user-set) | `/explicit` (guard fires: target is None; but working_dir is already explicit — overwrite happens unless we also gate on `working_dir is None`) ⚠️ — see §2.5 |
| Standalone, no target_path, no workspace | None | `os.getcwd()` | (no workspace assignment; not called) ✅ |
| Orchestrator-spawned child (target_path=None, workspace assigned later via setter) | None | `os.getcwd()` initially | Setter fires → guard sees target None → working_dir = workspace.root ✅ |

### 2.5 Edge case: explicit `working_dir` without `target_path`

Row 2 above exposes a subtle case. If user constructs `Leaf(working_dir="/explicit", workspace=ws)` and *doesn't* set target_path, the guard fires (target is None) and overwrites the user's explicit `working_dir` with `workspace.root`. **This is arguably wrong.** The user explicitly set `working_dir` — they should win.

**v4 fix:** add a sentinel field on TIB to track whether `working_dir` was user-supplied:

```python
working_dir: Optional[str] = attrib(default=None)
_working_dir_user_set: bool = attrib(default=False, init=False, repr=False)
target_path: Optional[str] = attrib(default=None)
```

In TIB's `__attrs_post_init__`, set `_working_dir_user_set = (self.working_dir is not None)` *before* defaulting.

In `_configure_for_workspace`, extend the guard:
```python
target = getattr(self, "target_path", None)
wd_user_set = getattr(self, "_working_dir_user_set", False)
if target is None and not wd_user_set:
    # ... clobber as before
```

This makes the table consistent: explicit `target_path` OR explicit `working_dir` both prevent the clobber. Documents user intent more precisely without breaking the orchestrator case.

**Trade-off:** adds one sentinel attrib. Worth it for correctness; the alternative (silently overwriting the user's `working_dir`) is the kind of bug that takes hours to debug when it happens. Both Claude's plan and v3 missed this; v4 fixes it.

---

## 3. Phased rollout

| Phase | Title | Files | Risk | Reversible alone? | Tests added |
|---|---|---|---|---|---|
| 0 | Pre-flight: RED tests + BUCK/YAML/config grep verification | 1 (test) + 1 (verification log) | Lowest | Yes | 10 |
| 1 | Guarded clobber in `_configure_for_workspace` + working_dir sentinel | 1 (inferencer_base.py) | Low | Yes | 3 |
| 2 | Decouple `StreamingInferencerBase` from `TemplatedInferencerBase` | 1 (streaming_inferencer_base.py) | Low–Medium | Yes | 3 |
| 3 | Decouple `TerminalInferencerBase`; add `target_path`; promote streaming output fields; add `_wrap_parse_output` hook; add `TerminalTemplatedInferencerBase` | 1 (terminal_inferencer_base.py) | Medium | Yes | 5 |
| 4 | Re-parent `TerminalSessionInferencerBase` via MI; reconcile `pre_exec_scripts` execution path; add `TerminalSessionTemplatedInferencerBase` | 1 (terminal_session_inferencer_base.py) | Medium | With Phase 3 | 6 |
| 5 | Migrate 5 CLI leaves; DevMate `repo_path` mirror; RovoDev `target_path or working_dir` fallback | 5 (one per CLI leaf) | Medium | Per-leaf | 4 |
| 6 | Update `__init__.py` exports + 3 docstrings | 3 | Trivial | Yes | 0 |
| 7 | Permanent regression invariants suite | 1 (test file) | Trivial | N/A | 5 |

**Total:** ~36 tests, 14 files touched. Roughly 50 hours / 1.5 engineer-weeks.

---

## 4. Phase 0 — Pre-flight tests + verification grep

### 4.1 New test file
`test/agent_foundation/common/inferencers/test_inferencer_axes_contract.py`

All 10 tests marked `xfail(strict=True)` with `# WILL_PASS_AFTER_PHASE_N` markers.

### 4.2 The 10 RED tests

| # | Test | Pre-fix | Lands |
|---|------|---------|-------|
| 1 | `test_target_path_survives_workspace_assignment` — KiroCli(target_path="/repo", workspace=ws) → working_dir == "/repo" | FAIL | Phase 3 + 5 |
| 2 | `test_explicit_working_dir_survives_workspace_assignment` — Leaf(working_dir="/x", workspace=ws) → working_dir == "/x" | FAIL | Phase 1 (sentinel) |
| 3 | `test_target_path_field_default_is_None` — `attr.fields(TIB).target_path.default is None` | FAIL (attrib absent) | Phase 3 |
| 4 | `test_working_dir_defaults_from_target_path` — TIB.post_init resolves working_dir = target_path or os.getcwd() | FAIL | Phase 3 |
| 5 | `test_session_inherits_terminal_features` — TSIB has `timeout`, `env_vars`, `post_exec_scripts` | FAIL | Phase 4 |
| 6 | `test_session_subprocess_uses_target_path` — patch asyncio shell, assert cwd == target_path | FAIL | Phase 3 + 5 |
| 7 | **`test_orchestrator_spawned_child_uses_workspace_root` (regression guard)** — BTA child (target_path=None) → working_dir == per-iter workspace.root | passes today | Phase 1 (must NOT regress) |
| 8 | **`test_orchestrator_does_not_override_explicit_target_path`** — BTA child (target_path="/explicit") → working_dir == "/explicit" | FAIL today | Phase 1 + 3 |
| 9 | **`test_devmate_repo_path_kwarg_still_accepted`** — DevmateCliInferencer(repo_path="/x") constructs; target_path == "/x" | passes today | Phase 5 (mirror) |
| 10 | **`test_pre_exec_scripts_runs_exactly_once_on_session_sync_path`** — TSIB subclass calls `_infer` (sync path), assert pre-exec scripts run exactly once (not twice) | likely FAIL after Phase 4 if pre_exec_scripts double-execution is not reconciled | Phase 4 (see §8.5) |

### 4.3 Pre-flight verification grep (records baseline assumptions)

Run before any source change; save output as `test/agent_foundation/common/inferencers/PREFLIGHT_GREP_BASELINE.txt`:

```bash
# 1. Confirm no YAML/BUCK config references base class names
grep -rn "StreamingInferencerBase\|TerminalInferencerBase\|TemplatedInferencerBase" \
    --include="*.yaml" --include="*.yml" --include="*.json" --include="BUCK" --include="TARGETS" \
    CoreProjects/AgentFoundation/

# 2. Confirm no functools.partial bakes template fields into inferencer factories
grep -rn "functools.partial.*template_manager\|partial.*template_key" \
    --include="*.py" CoreProjects/AgentFoundation/

# 3. Confirm exhaustive count of repo_path kwarg sites
grep -rn "repo_path *=" --include="*.py" CoreProjects/AgentFoundation/ | wc -l

# 4. Confirm exhaustive count of StreamingInferencerBase subclass declarations
grep -rn "class \w*(\w*StreamingInferencerBase\w*)" --include="*.py" \
    CoreProjects/AgentFoundation/src/

# 5. Confirm exhaustive count of TerminalInferencerBase / TerminalSessionInferencerBase subclasses
grep -rn "class \w*(\w*TerminalInferencerBase\w*)" --include="*.py" CoreProjects/AgentFoundation/
grep -rn "class \w*(\w*TerminalSessionInferencerBase\w*)" --include="*.py" CoreProjects/AgentFoundation/

# 6. Confirm isinstance check locations
grep -rn "isinstance.*TemplatedInferencerBase" --include="*.py" CoreProjects/AgentFoundation/
grep -rn "isinstance.*StreamingInferencerBase\|isinstance.*TerminalInferencerBase" \
    --include="*.py" CoreProjects/AgentFoundation/

# 7. Confirm template_manager is unused outside guarded SIB call sites
grep -rn "self.template_manager" --include="*.py" CoreProjects/AgentFoundation/src/
```

Commit the baseline output. If any count changes during implementation, the refactor's assumption set has shifted; pause and re-audit.

### 4.4 Phase 0 ships its own PR

Single test file + verification log. Zero source changes. RED tests run as `xfail(strict=True)` and turn GREEN as subsequent phases land. CI dashboard becomes the progress tracker.

---

## 5. Phase 1 — Guarded clobber + `working_dir` sentinel

**File:** `src/agent_foundation/common/inferencers/inferencer_base.py`

### 5.1 Edits

1. Apply the guarded clobber from §2.3.
2. Add the working_dir-user-set sentinel logic from §2.5. Since this sentinel lives on TIB (not InferencerBase), Phase 1 has only the `_configure_for_workspace` edit; the sentinel attrib is added in Phase 3 along with `target_path`. The guard reads `getattr(self, "_working_dir_user_set", False)` and `getattr(self, "target_path", None)` — both `getattr` calls return safe defaults (False / None) before Phase 3 ships.

### 5.2 Why Phase 1 is a pure no-op before Phase 3

At Phase 1 ship time:
- `target_path` attrib doesn't exist anywhere → `getattr(self, "target_path", None)` returns `None`.
- `_working_dir_user_set` attrib doesn't exist anywhere → `getattr(self, "_working_dir_user_set", False)` returns `False`.
- Guard `if target is None and not wd_user_set:` → `if True and not False:` → True → clobber fires.
- **Identical behavior to today's unconditional clobber.**

Phase 3's addition of `target_path` and `_working_dir_user_set` to TIB then *activates* both guards. Staged activation by design.

### 5.3 Tests landing in Phase 1

- New regression test: `test_phase1_preserves_orchestrator_clobber_behavior` — pre-Phase-3 leaf without target_path attrib, assign workspace via setter, assert `working_dir == workspace.root`.
- New regression test: `test_phase1_no_effect_when_target_path_attrib_absent` — assert no behavior change vs main for non-terminal inferencers.
- Phase 0 test #7 must continue to PASS after Phase 1 (regression guard).

### 5.4 Phase 1 risk register

| Risk | Mitigation |
|---|---|
| The rejected "delete entirely" approach would have broken orchestrators with `NotADirectoryError`. | v4 uses guarded clobber. Test #7 (regression guard) + new tests in §5.3. |
| BTA's `_configure_for_workspace` override at `breakdown_then_aggregate_inferencer.py:1135` calls super() expecting working_dir propagation. | Preserved: BTA's breakdown_inferencer is constructed with target_path=None → guard fires as before. Add `test_bta_breakdown_child_working_dir`. |
| `_propagate_workspace_to_children` comment at line 349 documents the clobber was load-bearing. | Same — guard preserves load-bearing behavior. Update comment to describe the conditional rule. |
| A user explicitly sets `target_path=os.getcwd()` and expects the guard to honor it. | Guard fires on `target is None`. `target_path=os.getcwd()` is non-None → guard skips → user wins. No degenerate case. |

---

## 6. Phase 2 — Decouple `StreamingInferencerBase` from `TemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/streaming_inferencer_base.py`

### 6.1 Edits

1. **Parent (line 103):**
   ```python
   # OLD: class StreamingInferencerBase(TemplatedInferencerBase):
   class StreamingInferencerBase(InferencerBase):
   ```

2. **Imports (lines 46–48):** swap `TemplatedInferencerBase` for `InferencerBase`.

3. **Duck-type `template_manager`** in two places:
   ```python
   # Line 225 (__attrs_post_init__):
   def __attrs_post_init__(self):
       super().__attrs_post_init__()
       tm = getattr(self, "template_manager", None)
       if self.use_default_prompt_templates and tm is not None:
           tm.add_template_root(_DEFAULT_RECOVERY_DIR, priority=TemplateRootPriority.LOWEST)

   # Line 246 (_render_recovery_prompt):
   def _render_recovery_prompt(self, mode, prompt, partial_output):
       key = f"{self.fallback_recovery_template_key}/{mode.value}"
       tm = getattr(self, "template_manager", None)
       if tm is not None:
           return tm(key, active_template_type="", prompt=prompt,
                     partial_output=partial_output)
       elif self.use_default_prompt_templates:
           return render_recovery_prompt(...)  # module-level fallback already in code
   ```

4. **Keep** `use_default_prompt_templates: bool = attrib(default=True)` and `fallback_recovery_template_key: str = "recovery"`. Configure recovery regardless of whether `template_manager` is present.

### 6.2 Impact on 10 streaming subclasses

All 10 use zero template features (verified by §4.3 grep #7). After Phase 2:
- They lose template attribs from `__init__` (correct — they never used them).
- `isinstance(x, TemplatedInferencerBase)` returns False (correct).
- No leaf code changes required.

### 6.3 Tests landing in Phase 2

- `test_sib_no_longer_inherits_templated` — `assert not issubclass(StreamingInferencerBase, TemplatedInferencerBase)`.
- `test_sib_recovery_fallback_when_no_template_manager` — instantiate a SIB-only subclass without `template_manager`; trigger `_render_recovery_prompt`; assert the module-level fallback fires.
- `test_sib_recovery_uses_template_manager_when_present` — instantiate a templated subclass (post-Phase-5 ClaudeCode); trigger `_render_recovery_prompt`; assert it routes through `template_manager`.

### 6.4 Phase 2 risk register

| Risk | Mitigation |
|---|---|
| A streaming-only subclass implicitly relied on `template_manager`. | Verified zero. Test 6.3 #2 pins fallback. |
| Module-level `render_recovery_prompt` requires template_manager too. | Verified self-contained: uses Jinja directly from `_DEFAULT_RECOVERY_DIR`. |
| Removing TemplatedIB from MRO breaks isinstance checks. | The 2 known checks (dual:1602, mfdual:460) discriminate templated-vs-not — SHOULD return False for SIB-only. Correct behavior, not break. |
| YAML/BUCK configs reference template fields on SIB-only subclasses. | §4.3 grep #1 confirms zero. |

---

## 7. Phase 3 — Decouple `TerminalInferencerBase`; add `target_path`; promote streaming fields; add `_wrap_parse_output` hook; add `TerminalTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/terminal_inferencer_base.py`

### 7.1 Edits

1. **Parent (line 17):**
   ```python
   # OLD: class TerminalInferencerBase(TemplatedInferencerBase):
   class TerminalInferencerBase(InferencerBase):
   ```

2. **Imports (lines 12–14):** swap `TemplatedInferencerBase` for `InferencerBase`.

3. **Add `target_path` field** before `working_dir` (after line 41):
   ```python
   # The directory the CLI agent operates on (e.g., the repo it edits).
   # Distinct from:
   #   - workspace.root: where THIS inferencer stores its artifacts.
   #   - working_dir:    the subprocess cwd= for command execution.
   # Defaulting: leaves may set this in their __attrs_post_init__ BEFORE
   #   calling super(). TIB does NOT default this to os.getcwd() — see
   #   plan §2.1 for why (load-bearing for the _configure_for_workspace
   #   guard against orchestrator scenarios).
   target_path: Optional[str] = attrib(default=None)
   ```

4. **Add `_working_dir_user_set` sentinel:**
   ```python
   _working_dir_user_set: bool = attrib(default=False, init=False, repr=False)
   ```

5. **Promote streaming output fields** to `attrib(init=False)`:
   ```python
   _last_streaming_output: str = attrib(default="", init=False, repr=False)
   _last_streaming_return_code: int = attrib(default=0, init=False, repr=False)
   ```

6. **Update `__attrs_post_init__`:**
   ```python
   def __attrs_post_init__(self):
       # Track whether working_dir was user-supplied (for §2.5 guard).
       self._working_dir_user_set = (self.working_dir is not None)
       # Default working_dir from target_path; do NOT default target_path
       # itself — see plan §2.1.
       if self.working_dir is None:
           self.working_dir = self.target_path or os.getcwd()
       super().__attrs_post_init__()
   ```

7. **Add `_wrap_parse_output` hook** for return-type reconciliation:
   ```python
   def _wrap_parse_output(self, parsed):
       """Hook for subclasses to wrap parse_output result. Default: identity."""
       return parsed
   ```
   Update TIB's `_infer` to call `return self._wrap_parse_output(self.parse_output(...))`.

8. **Add `TerminalTemplatedInferencerBase` convenience class** at bottom of file:
   ```python
   @attrs
   class TerminalTemplatedInferencerBase(TerminalInferencerBase, TemplatedInferencerBase):
       """Sync terminal + templates (no streaming/session).
       MRO: TTIB → TIB → TemplatedIB → IB → Debuggable → Resumable → ABC.
       """
       pass
   ```

### 7.2 Tests landing in Phase 3

- Phase 0 tests #1, #2, #3, #4, #6, #8 turn GREEN.
- `test_tib_no_longer_inherits_templated` — `assert not issubclass(TerminalInferencerBase, TemplatedInferencerBase)`.
- `test_tib_target_path_attribute_exists` — assert `"target_path"` in `attr.fields(TerminalInferencerBase)`.
- `test_tib_target_path_default_is_None` — `attr.fields(TIB).target_path.default is None` (the load-bearing constraint from §2.1).
- `test_tib_streaming_output_fields_are_attribs` — verify `_last_streaming_output` and `_last_streaming_return_code` are `init=False`.
- `test_ttib_mro` — assert TTIB MRO matches documented.

### 7.3 Phase 3 risk register

| Risk | Mitigation |
|---|---|
| 5 Terminal-only test stubs lose template attribs from `__init__`. | They never used them. Confirmed by audit. |
| Promoting `_last_streaming_output` changes initial-value semantics. | Default `""` / `0` matches pre-set state. Pre-set state unobservable (set in `_execute_command_streaming` before any read). |
| Future contributor defaults `target_path` to `os.getcwd()`. | Permanent regression test (§12 #5) pins the field default. Comment in field docstring + `__attrs_post_init__` body explicitly forbids it. |
| `_wrap_parse_output` default is identity → Terminal-only stubs see no behavior change. | Explicit design. Document. |

---

## 8. Phase 4 — Re-parent TSIB via MI; **reconcile `pre_exec_scripts` execution path**; add `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/terminal_inferencers/terminal_session_inferencer_base.py`

### 8.1 Edits

1. **Parent (line 56):**
   ```python
   # OLD: class TerminalSessionInferencerBase(StreamingInferencerBase):
   class TerminalSessionInferencerBase(TerminalInferencerBase, StreamingInferencerBase):
       """Terminal exec + streaming/recovery.
       MRO: TSIB → TIB → SIB → IB → Debuggable → Resumable → ABC.
       """
   ```

2. **Add import** for `TerminalInferencerBase`.

3. **Remove duplicated attribs** (now inherited):
   ```python
   # DELETED (inherited from TIB):
   # working_dir: Optional[str] = attrib(default=None)     (line 71)
   # pre_exec_scripts: Optional[List[str]] = attrib(...)   (line 72)
   # _last_streaming_output                                 (line 89)
   # _last_streaming_return_code                            (line 91)
   ```

4. **Remove duplicate method** `_resolve_subprocess_cwd` (lines 136–156) — identical to TIB's.

5. **Delete TSIB's `_infer` override.** Inherit TIB's richer version (with timeout, env_vars, post_exec_scripts, _save_output). Override `_wrap_parse_output` only:
   ```python
   def _wrap_parse_output(self, parsed):
       return TerminalInferencerResponse.from_dict(parsed)
   ```

6. **Keep** TSIB's async streaming machinery (`_ainfer_streaming`, `_read_stdout_with_exit_detection`, `_poll_process_exit`, `_force_close_pipes`, `_kill_process_group`, `_safe_process_cleanup`, `_build_full_command`, `_ainfer` accumulation) — unchanged.

7. **Keep** `_last_streaming_stderr` (TSIB-only field; not promoted to TIB because TIB doesn't run streaming).

8. **Add convenience class at bottom of file:**
   ```python
   @attrs
   class TerminalSessionTemplatedInferencerBase(
       TerminalSessionInferencerBase, TemplatedInferencerBase,
   ):
       """Terminal + streaming + templates.
       MRO: TSTIB → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC.

       Use this for CLI inferencers that need all three axes
       (ClaudeCode, Kiro, Devmate, RovoDev). Use TerminalSessionInferencerBase
       directly if you don't want templates (Metamate).
       """
       pass
   ```

### 8.2 The `__attrs_post_init__` MRO chain (worked example)

For `ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase)`:

```
ClaudeCode.__attrs_post_init__
  1. set target_path = "~/fbsource" if None  (Claude-specific default)
  2. super() → TSTIB (no post_init)
       → TSIB (no post_init)
         → TIB.__attrs_post_init__
              - _working_dir_user_set = (self.working_dir is not None)  # False
              - self.working_dir = self.target_path or os.getcwd()      # "~/fbsource"
              - super() → SIB.__attrs_post_init__
                   - super() → TemplatedIB (no post_init)
                        → IB.__attrs_post_init__
                             - sets _workspace from workspace if provided
                             - _configure_for_workspace fires
                               - guard sees target_path = "~/fbsource" → SKIP clobber ✅
                               - sets cache_folder, logger
                   - duck-typed: tm = getattr(self, "template_manager", None)
                     → present (Claude has it via TemplatedIB)
                     → add recovery template root
  3. (back in ClaudeCode) _resolve_claude_command()  (no path deps — F11 verified)
```

Every layer runs exactly once. MRO is C3-deterministic. attrs field collection deduplicates (working_dir, pre_exec_scripts appear once in the field list from TIB).

### 8.3 Tests landing in Phase 4

- Phase 0 test #5 turns GREEN.
- `test_tsib_mro_is_documented` — assert `TerminalSessionInferencerBase.__mro__` matches.
- `test_tstib_mro_is_documented` — same for `TerminalSessionTemplatedInferencerBase`.
- `test_tsib_inherits_timeout_env_vars_post_exec` — assert all three attribs present.
- `test_tsib_no_duplicate_working_dir_attrib` — `len([f for f in attr.fields(TSIB) if f.name == "working_dir"]) == 1`.
- `test_tsib_pipe_hang_detection_still_works` — regression for MCP hang fix.
- `test_pre_exec_scripts_runs_exactly_once_on_session_sync_path` — see §8.5 below.

### 8.4 Phase 4 risk register

| Risk | Mitigation |
|---|---|
| Diamond MRO produces unexpected attrs field order. | All 5 classes use legacy `@attrs` (slots=False) — verified. C3 deterministic. Phase 0 test #5 + Phase 4 MRO assertions catch drift. |
| TIB's stronger `_infer` breaks subclasses with overrides that don't expect timeout/env_vars/post_exec. | Audit: ClaudeCodeCli and KiroCli both override `_infer` and don't call `super()._infer`. No break. |
| `_wrap_parse_output` hook semantics differ between sync (`_infer` via TIB) and async (`_ainfer` via TSIB). | TIB's `_infer` calls `_wrap_parse_output`. TSIB's `_ainfer` accumulator calls `_wrap_parse_output` after final-parse. Both routes funnel through the same hook. Add `test_wrap_parse_output_called_on_both_sync_and_async`. |
| **`pre_exec_scripts` double-execution risk** (new in v4 — see §8.5). | Explicit reconciliation design in §8.5. Phase 0 test #10 pins single-execution. |
| `large_input_mode` (TSIB-specific) interacts with TIB's `_execute_command`. | `large_input_mode` is consumed only by TSIB's `_ainfer_streaming`. TIB's `_execute_command` untouched. No interaction. |

### 8.5 ⚠️ NEW v4 finding — `pre_exec_scripts` execution path reconciliation

**The risk both prior plans missed:**

Today's pre-refactor state:
- `TerminalInferencerBase._infer` calls `self._execute_scripts(self.pre_exec_scripts)` and then `self._execute_command(...)` — two separate subprocess invocations. Pre-scripts run in their own shell (env vars set by them DO NOT propagate to the main command).
- `TerminalSessionInferencerBase._ainfer_streaming` calls `self._build_full_command(...)` which chains `pre_scripts && main_cmd` into a single shell — env vars from pre-scripts DO propagate.

Post-Phase-4 state (TSIB inherits TIB's `_infer`):
- A TSIB subclass that triggers the sync path (e.g., via `infer()` → falls back to `_infer` rather than `_ainfer`) would now run pre_exec_scripts via TIB's `_execute_scripts` (separate shell) — different env semantics from the async path.
- Worse: if a leaf overrides `_infer` to call `super()._infer()` AND its `_ainfer` separately runs `_build_full_command` somewhere, pre-scripts could execute *twice*.

**Reconciliation design (v4):**

Add an explicit `_run_pre_exec_scripts_in_subprocess_shell` strategy hook on TIB:

```python
# In TerminalInferencerBase:
def _run_pre_exec_scripts_in_subprocess_shell(self) -> bool:
    """Whether pre_exec_scripts should be chained into the main subprocess
    shell (True, env propagates) or run as a separate pre-step (False, env
    does NOT propagate). Default False = current TIB behavior.
    """
    return False

# In _infer:
if self.pre_exec_scripts and not self._run_pre_exec_scripts_in_subprocess_shell():
    self._execute_scripts(self.pre_exec_scripts)
# If True, _execute_command is responsible for prefixing pre_exec_scripts
# into its shell (similar to _build_full_command).
```

```python
# In TerminalSessionInferencerBase, override:
def _run_pre_exec_scripts_in_subprocess_shell(self) -> bool:
    """Session subclasses chain pre-scripts via && in the main shell
    (env-vars propagate). Disables TIB's separate _execute_scripts call
    to avoid double-execution.
    """
    return True
```

This ensures:
- TIB-only subclasses: pre-scripts run as separate step (today's behavior, preserved).
- TSIB subclasses: pre-scripts run inline in main shell (today's behavior, preserved).
- No code path runs pre-scripts twice.
- Phase 0 test #10 (`test_pre_exec_scripts_runs_exactly_once_on_session_sync_path`) pins this.

**Why this is elegant, not a hack:** the hook documents the asymmetry that already existed pre-refactor. Without it, the asymmetry was implicit in two different `_infer` implementations. With it, the asymmetry is explicit, named, and testable.

---

## 9. Phase 5 — Migrate the 5 CLI leaves

### 9.1 `ClaudeCodeCliInferencer` → `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`

> **Audit F11 verdict (✅ SAFE):** `_resolve_claude_command()` (lines 125–186) has zero references to `self.target_path`, `self.working_dir`, or `self._workspace`. Safe to call before or after `super().__attrs_post_init__()`.

1. **Parent (line 31):** `class ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase):`
2. **Remove local `target_path` declaration (line 92):** inherited from TIB.
3. **Remove `self.working_dir = self.target_path`** from post-init (line 121): TIB handles it.
4. **Keep** the Claude-specific `target_path = ~/fbsource` default, marked DO-NOT-DELETE:
   ```python
   def __attrs_post_init__(self) -> None:
       # ─── DO NOT DELETE: Claude-specific default target ─────────────
       if self.target_path is None:
           self.target_path = os.path.expanduser("~/fbsource")
       # ─── End must-preserve ─────────────────────────────────────────
       super().__attrs_post_init__()
       # NOTE: working_dir = target_path now handled by TIB post-init.
       # ─── DO NOT DELETE: claude command path resolution (F11 ✅) ────
       self._resolve_claude_command()
       # ─── End must-preserve ─────────────────────────────────────────
   ```

### 9.2 `KiroCliInferencer` → `TerminalSessionTemplatedInferencerBase`

1. **Parent (line 25):** swap.
2. **Remove local `target_path` declaration (line 73).**
3. **Remove `self.working_dir = self.target_path`** from post-init (line 93).
4. **Note on Kiro's default behavior:** Kiro previously set `target_path = os.getcwd()`. Under v4 this is **optional** — if Kiro leaves target_path = None, TIB defaults working_dir = os.getcwd() anyway. Leaving target_path = None is **preferred** because it allows orchestrator-spawned Kiro children to be steered by `_configure_for_workspace` instead of being pinned to the original cwd.
5. **Keep** model normalization business logic, marked DO-NOT-DELETE:
   ```python
   def __attrs_post_init__(self) -> None:
       from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.common import (
           resolve_model_tag,
       )
       # ─── DO NOT DELETE: model normalization ────────────────────────
       if self.model_name and self.model_name != "auto":
           self.model_name = resolve_model_tag(self.model_name)
       # ─── End must-preserve ─────────────────────────────────────────
       super().__attrs_post_init__()
   ```

### 9.3 `DevmateCliInferencer` → `TerminalSessionTemplatedInferencerBase`

> **⚠️ CRITICAL — `repo_path` MUST stay as an attrib.** The naïve `@property` approach breaks attrs (`__init__` no longer accepts `repo_path=` kwarg). Verified: **45+ test sites** pass `repo_path=` as kwarg.

1. **Parent (line 63):** swap.
2. **Keep** `repo_path: Optional[str] = attrib(default=None)` (line 163) — UNCHANGED.
3. **Update post-init** to mirror `repo_path → target_path` BEFORE `super()`:
   ```python
   def __attrs_post_init__(self):
       """Devmate-specific: ~/fbsource default; cd-into-repo pre-exec."""
       # ─── DO NOT DELETE: Devmate-specific default (~/fbsource) ──────
       if self.repo_path is None:
           self.repo_path = os.path.expanduser("~/fbsource")
       # ─── End must-preserve ─────────────────────────────────────────

       # Mirror repo_path → target_path BEFORE super() so TIB's working_dir
       # defaulting sees the correct value. The mirror is one-way at construction
       # time; if a user post-construction reassigns repo_path, target_path
       # does NOT auto-update (would require @repo_path.setter which @attrs
       # doesn't generate). Document this in field docstring.
       if self.target_path is None:
           self.target_path = self.repo_path

       # ─── DO NOT DELETE: cd-into-repo pre-exec script ───────────────
       cd_script = f'cd "{self.repo_path}" || exit 1'
       if self.pre_exec_scripts is None:
           self.pre_exec_scripts = [cd_script]
       elif cd_script not in self.pre_exec_scripts:
           self.pre_exec_scripts.insert(0, cd_script)
       # ─── End must-preserve ─────────────────────────────────────────

       super().__attrs_post_init__()
   ```

All 45+ existing `repo_path=` test sites continue to pass unchanged.

### 9.4 `RovoDevCliInferencer` → `TerminalSessionTemplatedInferencerBase`

> **Audit F10 verdict (✅ SAFE):** Exhaustive trace of all 6 `self.working_dir` reads confirms they expect "directory the CLI agent operates on" = target_path semantics.

1. **Parent (line 75):** swap.
2. **Remove `if self.working_dir is None: self.working_dir = os.getcwd()`** (lines 144–146): TIB handles it.
3. **Update 4 session call sites** (lines 623, 628, 677, 683) using Claude's safe fallback pattern:
   ```python
   # OLD: workspace_path=self.working_dir
   # NEW: workspace_path=self.target_path or self.working_dir
   ```
   The `or` fallback preserves today's behavior (working_dir always non-None after post-init) AND uses target_path semantics when explicitly set.
4. **Leave target_path = None by default** (RovoDev doesn't have a leaf-specific default).

### 9.5 `MetamateCliInferencer` — stays on `TerminalSessionInferencerBase`

- **No parent change.** Metamate confirmed not using templates; non-templated branch is correct fit.
- **Remove** any local `target_path` / `working_dir` initialization.
- **Pre-merge mini-audit recommended:** confirm Metamate has no hidden working_dir dependencies (analogous to F10/F11).

### 9.6 Tests landing in Phase 5

- Phase 0 tests #6, #9 turn GREEN.
- `test_each_cli_leaf_isinstance_of_templated` for {Claude, Kiro, Devmate, RovoDev} → True; for Metamate → False.
- `test_devmate_repo_path_mirrors_to_target_path` — `DevmateCliInferencer(repo_path="/x"); assert inst.target_path == "/x"`.
- `test_metamate_does_not_inherit_templated` — `assert not isinstance(MetamateCliInferencer(), TemplatedInferencerBase)`.

### 9.7 Phase 5 risk register

| Risk | Mitigation |
|---|---|
| DevMate's `repo_path` and `target_path` drift if user reassigns post-construction. | Document "use either name consistently; do not mix" in field docstring. Test sites construct fresh instances. |
| MetamateCli loses some feature by not inheriting from templated branch. | Verified: not using templates. No loss. |
| A leaf's `__attrs_post_init__` does business logic AFTER super() that depends on TIB's post-init having run (reads `working_dir`). | F11 verified Claude's `_resolve_claude_command()` has no path deps. Others have nothing after super(). |
| dual_inferencer.py:1602 / mfdual:460 isinstance checks change behavior. | Verified: TSTIB subclasses → True; Metamate (TSIB-only) → False. Correct discrimination per axes design. |
| RovoDev's session-finding `workspace_path=` parameter semantics drift. | The `target_path or working_dir` fallback preserves today's behavior in all cases. |

---

## 10. Phase 6 — Update exports + docstrings

### 10.1 `src/agent_foundation/common/inferencers/terminal_inferencers/__init__.py`

```python
from .terminal_inferencer_base import (
    TerminalInferencerBase,
    TerminalTemplatedInferencerBase,
)
from .terminal_inferencer_response import TerminalInferencerResponse
from .terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
    TerminalSessionTemplatedInferencerBase,
)

__all__ = [
    "TerminalInferencerBase",
    "TerminalTemplatedInferencerBase",
    "TerminalInferencerResponse",
    "TerminalSessionInferencerBase",
    "TerminalSessionTemplatedInferencerBase",
]
```

### 10.2 `src/agent_foundation/common/inferencers/templated_inferencer_base.py` (docstring lines 6–15)

> "After 2026-05-16 axes refactor: `TemplatedInferencerBase` is one of three orthogonal axes (templating, streaming, terminal-exec). Leaves opt into templating via direct inheritance OR via the convenience MI classes `TerminalTemplatedInferencerBase` and `TerminalSessionTemplatedInferencerBase`. The cascade-injection of `_template_manager` walks every descendant that has `template_manager` as a constructor param; SIB and TIB no longer inherit it, so streaming-only and terminal-only leaves are no longer accidental recipients of cascaded template state."

### 10.3 `src/agent_foundation/common/inferencers/streaming_inferencer_base.py` (top docstring)

> "Streaming + cache-based recovery base. Inherits from `InferencerBase`. Recovery-prompt rendering uses `template_manager` if present (duck-typed via `getattr`) and falls back to a module-level Jinja-only renderer otherwise. See plan §6 for the decoupling rationale."

---

## 11. Phase 7 — Permanent regression invariants suite

**File (new):** `test/agent_foundation/common/inferencers/test_inferencer_axes_invariants.py`

These tests are **permanent** — they pin invariants the entire refactor depends on. Future contributor breaking them gets immediate CI failure.

### 11.1 Five permanent tests

1. **`test_axes_isinstance_matrix`** — for each of the 5 CLI leaves + 10 streaming-only leaves, assert expected `isinstance` results against `(TerminalInferencerBase, StreamingInferencerBase, TemplatedInferencerBase, TerminalSessionInferencerBase, TerminalSessionTemplatedInferencerBase)`. Contract grid for the axes design.

2. **`test_three_diamond_mros_documented`** — assert all 3 diamond MROs (TSIB, TTIB, TSTIB) match documented orders.

3. **`test_diamond_attrs_slots_consistency`** — assert all classes in the 3 diamond MROs have `slots=False` (legacy `@attrs` API):
   ```python
   def test_diamond_attrs_slots_consistency():
       classes = [
           InferencerBase, TemplatedInferencerBase,
           StreamingInferencerBase, TerminalInferencerBase,
           TerminalSessionInferencerBase,
           TerminalTemplatedInferencerBase,
           TerminalSessionTemplatedInferencerBase,
       ]
       for cls in classes:
           slots = getattr(cls, "__slots__", None)
           assert slots is None or slots == (), (
               f"{cls.__name__} has non-empty __slots__={slots!r} — "
               "diamond inheritance with mixed slots will break MRO-based "
               "field resolution. Stay on the legacy `from attr import attrs` "
               "API for all classes in the inferencer axes diamonds."
           )
   ```

4. **`test_no_duplicate_fields_under_diamond`** — for each diamond class, assert `attr.fields(cls)` contains no duplicate field names.

5. **`test_tib_target_path_field_default_is_None`** (new v4 — the load-bearing constraint from §2.1):
   ```python
   def test_tib_target_path_field_default_is_None():
       """target_path MUST default to None (NEVER os.getcwd()).
       If this fails, the _configure_for_workspace guard becomes broken
       for orchestrator-spawned children. See plan §2.1.
       """
       assert attr.fields(TerminalInferencerBase).target_path.default is None
   ```

---

## 12. Comparison with source plans (Round 2)

| Aspect | v3 (Rovo Dev) | Claude (updated 07:25) | v4 (this plan) |
|---|---|---|---|
| `target_path` field default | Claimed None in code but didn't highlight | **Explicit: "defaults to None NEVER os.getcwd() — load-bearing!"** | **Adopt Claude's emphasis (§2.1)** |
| `_configure_for_workspace` gate | `target != os.getcwd()` (over-engineered) | `target is None` (clean) | **Claude's gate** |
| Explicit `working_dir` w/o target_path | Silently overwritten | Silently overwritten | **v4 NEW: sentinel `_working_dir_user_set` prevents overwrite (§2.5)** |
| `pre_exec_scripts` execution path | Mentioned asymmetry in risk register | Mentioned briefly | **v4 NEW: explicit `_run_pre_exec_scripts_in_subprocess_shell` hook (§8.5)** |
| RovoDev session-finding migration | "rename direct to target_path" | `target_path or working_dir` fallback | **Claude's safer fallback** |
| Deferred Work section | Missing | Present | **Adopt Claude's section (§13)** |
| Phase 0 RED tests | 9 tests | 5 tests | **v4: 10 tests (added #10 for pre-exec single-execution)** |
| Pre-flight verification grep | Missing | Missing | **v4 NEW: §4.3 baseline grep + commit-as-evidence** |
| Per-phase risk registers | Yes (3) | No | **Adopt v3's 6 risk registers + add Phase 5's** |
| Rollback per-phase matrix | Yes | No | **Adopt v3's matrix (§14)** |
| DO-NOT-DELETE leaf markers | Yes | No | **Adopt v3's markers (§9)** |
| `_wrap_parse_output` hook | Yes (Phase 2b) | Yes (Phase 3) | **Adopt both — define in TIB, override in TSIB** |
| `_last_streaming_output` promotion | Not addressed | Yes | **Adopt Claude's promotion (§7.1 #5)** |
| MetamateCli classification | Ambiguous | Explicit (stays on TSIB) | **Claude's explicit classification** |
| BTA `_configure_for_workspace` override location | Documented line 1135 | Mentioned without line | **Adopt v3's precision** |
| F10 (RovoDev working_dir trace) | Done ✅ | Cited | **Adopted** |
| F11 (ClaudeCode resolve cmd) | Done ✅ | Cited | **Adopted** |
| Permanent regression suite | 4 tests | None | **v4: 5 tests (added field-default invariant)** |
| Plan length | 786 lines | 292 lines | ~1050 lines target |

### 12.1 If we had to pick ONE plan today, which?

**My honest answer: Claude's updated plan.**

Three reasons:
- **Architecturally it remains the right answer** (decouple 3 bases; 3 MI convenience classes), and it now matches v3's operational rigor on most fronts.
- **The "target_path defaults to None — load-bearing!" insight is the single most important design constraint in the whole refactor**, and Claude makes it front-and-center while v3 buried it. A reader of Claude's plan walks away with the right mental model. A reader of v3 has to reconstruct it.
- **The `target_path or working_dir` fallback for RovoDev is a smarter migration pattern** than v3's rename-directly approach — preserves today's behavior in all cases.

What v3 has that Claude's still lacks: per-phase risk registers, rollback per-phase matrix, DO-NOT-DELETE leaf markers, the explicit BTA override line number, the slot-consistency test code, and the explicit pre-flight verification grep. These are *important* operational additions but they're scaffolding around the architecture — they don't change structural decisions.

**Bottom line:** if forced to ship from Claude's plan alone today, you'd get a correct refactor and you'd add the missing operational scaffolding during implementation. If forced to ship from v3 alone, you'd ship the right architecture but have to *reconstruct* the load-bearing design constraint during reviews when someone asks "why target_path = None and not os.getcwd()?". Picking Claude's costs less.

### 12.2 Genuine new gaps v4 adds (beyond either source plan)

1. **Sentinel `_working_dir_user_set`** (§2.5) — closes the row-2 edge case in the correctness table.
2. **`pre_exec_scripts` reconciliation hook** (§8.5) — closes the sync/async asymmetry that becomes a real bug post-Phase-4.
3. **`test_tib_target_path_field_default_is_None`** (§11.1 #5) — permanent regression for the load-bearing constraint.
4. **Pre-flight verification grep** (§4.3) — records assumption baseline that can be diffed against implementation.
5. **Worked MRO chain example** (§8.2) — shows reviewers the per-line behavior of the diamond's `__attrs_post_init__` chain.

---

## 13. Deferred work (explicit non-scope)

(Adopted from Claude's plan §7.)

- **`ApiInferencerBase` / `RemoteInferencerBase` decoupling from `TemplatedInferencerBase`.** Both still inherit TemplatedIB. Can be decoupled in a follow-up for consistency. Not blocking.
- **`StreamingTemplatedInferencerBase(SIB, TemplatedIB)` convenience class.** Not needed today (no SIB subclass uses templates). Add if demand arises.
- **Renaming `StreamingInferencerBase` → `TemplatedStreamingInferencerBase`.** Obsoleted by decoupling — once SIB no longer inherits TemplatedIB, the name is honest. Do not pursue.
- **Migrating all bases to `@attrs.define` (slots=True).** Would break the 3 diamonds (slot inheritance conflicts). Permanent regression test §11.1 #3 enforces "stay on legacy `@attrs`".
- **Removing `repo_path` from DevMate entirely.** Possible long-term if all 45+ call sites migrate to `target_path`. Not done in this refactor (would require a 6-month migration window).
- **CoreProjects vs atlassian_packages tree synchronization.** Both trees have copies of these files; this plan applies only to `CoreProjects/AgentFoundation`. The `atlassian_packages/rovoteam/AgentFoundation` copy is separately tracked.

---

## 14. Migration & rollback

### 14.1 Branch & PR strategy

- **Single feature branch:** `refactor/inferencer-axes-decoupling-v4`.
- **8 commits, one per phase.** Each commit independently revertable.
- **PR strategy:**
  - **PR-1 (Phase 0):** RED tests + preflight grep baseline. Lands first.
  - **PR-2 (Phase 1):** Guarded clobber + sentinel reads. Pure no-op until Phase 3.
  - **PR-3 (Phases 2–4):** Structural core. Reviewed as one PR (Phases 3+4 co-dependent).
  - **PR-4 (Phase 5):** Leaf migrations. Optional split into 5 sub-commits.
  - **PR-5 (Phases 6–7):** Exports + permanent regression suite.

### 14.2 Rollback per phase

| Revert | Consequence |
|---|---|
| Phase 0 only | Tests vanish; no functional change |
| Phase 1 only | Returns to unconditional clobber; safe |
| Phase 2 only (after 1) | SIB re-inherits TemplatedIB; streaming leaves regain template attribs in `__init__` (unused but present) |
| Phase 3 only (after 2) | target_path / sentinel vanish; guarded clobber reverts to no-op; TIB re-inherits TemplatedIB |
| Phase 4 only (after 3) | TSIB returns to single-inheritance from SIB; convenience classes vanish; CLI leaves break (reference TSTIB) → revert Phase 5 first |
| Phase 5 only (after 4) | Leaves return to TSIB parent; lose templating |
| Phase 6 only | Imports break wherever TSTIB is referenced externally |
| Phase 7 only | Loses permanent regression tests |

**Safe rollback combinations:**
- {Phase 7} — pure test removal.
- {Phases 4, 5, 6, 7} — undoes MI refactor, leaves decoupling in place.
- {Phases 1–7} — full revert to today's state.

### 14.3 Cross-team notification

- **Orchestrator owners** (BTA, MFDual, LWI, Dual) — guarded clobber preserves their behavior; smoke-test on feature branch.
- **DevMate consumers** — `repo_path` semantics unchanged; now also reflects into `target_path`. Internal docs update.
- **Streaming-only leaf owners** — no notification needed (zero observable change).

### 14.4 Estimated effort

| Phase | Impl | Tests | Review | Total (h) |
|---|---|---|---|---|
| 0 | 0 | 5 | 1 | 6 |
| 1 | 1 | 2 | 1 | 4 |
| 2 | 1 | 2 | 1 | 4 |
| 3 | 3 | 3 | 2 | 8 |
| 4 | 5 | 5 | 4 | 14 |
| 5 | 3 | 4 | 3 | 10 |
| 6 | 1 | 0 | 1 | 2 |
| 7 | 1 | 3 | 1 | 5 |
| **Total** | **15** | **24** | **14** | **53** |

Roughly **1.6 engineer-weeks**. Higher than v3 (49h) because of Phase 4's pre_exec_scripts reconciliation work and Phase 0's expanded test count.

---

## 15. Design principles applied (12)

1. **Single Responsibility per field.** `workspace`, `target_path`, `working_dir` — three names, three roles, one writer each.
2. **Orthogonal axes, composable via MI.** Templating, streaming, terminal-exec are orthogonal capabilities; leaves opt in via narrow convenience MI classes.
3. **Names tell the truth.** `StreamingInferencerBase` is honest once decoupled. No rename needed.
4. **Locality of behavior.** Default-resolution logic lives in exactly one place (TIB.`__attrs_post_init__`).
5. **Explicit > implicit.** `target_path` MUST default to `None` (never `os.getcwd()`) — explicit user intent vs. silent defaulting.
6. **User intent always wins.** Sentinel `_working_dir_user_set` ensures explicit `working_dir` is never silently overwritten.
7. **Asymmetries documented as explicit hooks.** `_wrap_parse_output`, `_run_pre_exec_scripts_in_subprocess_shell` — what was implicit becomes explicit, named, and testable.
8. **Backwards compatibility via mirroring.** DevMate's `repo_path` stays as an attrib (preserves 45+ test sites) and mirrors into `target_path`.
9. **Diamond inheritance accepted only when both parents trace to the same root.** All 3 diamonds resolve to `InferencerBase` via C3.
10. **Test-first via Phase 0.** 10 RED tests pin the contract before any source change.
11. **Phased rollout with independent revertability.** Each of 8 phases shippable and revertable.
12. **Permanent regression tests pin load-bearing invariants.** §11's 5 tests stay in CI forever.

---

*End of integrated v4 plan. Reviewers: please challenge §2.1 (the load-bearing target_path constraint), §2.5 (the new sentinel), §8.5 (the new pre_exec_scripts reconciliation), and §12.1 ("pick Claude's if forced to one") most carefully.*







