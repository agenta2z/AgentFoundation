# Terminal Inferencer Axes Refactor + StreamingInferencerBase Rename — Unified End-to-End Plan

> **✅ AUDIT-INTEGRATED v2 (2026-05-16 06:59):**
> This plan has been audited and the 8 binding corrections (C1–C8) from `terminal_inferencer_axes_AUDIT_FINDINGS.md` are now **integrated in-line below**. The audit document is preserved as a historical record. Follow-up audits F10 (RovoDev `working_dir` exhaustive trace) and F11 (ClaudeCodeCli `_resolve_claude_command`) both returned ✅ SAFE — no further blockers. Implementation may proceed.
>
> **What changed in v2 vs. v1:**
> - **Phase 1** (§6) — redesigned: the clobber is NOT deleted; it is made *conditional* on whether the user explicitly set `target_path`. The unconditional deletion was load-bearing for orchestrator-spawned children (would have caused `NotADirectoryError`).
> - **Phase 1.5** (§6A) — the back-compat alias is now permanent (pickle compatibility); added isinstance-loop test; documented `__getattr__` cache semantics.
> - **Phase 3 Devmate** (§8.1) — redesigned: `repo_path` stays as an `attrib` and is *mirrored* into `target_path` in post-init. The original `@property` approach was an `attrs` API violation that would have raised `TypeError` on every existing test site.
> - **Phase 3 Kiro/Claude/RovoDev/Meta** (§8.2–8.5) — added explicit DO-NOT-DELETE markers around business logic in post-init.
> - **Phase 0** (§5) — added 3 orchestrator-scenario tests (now 9 tests, was 6).
> - **Phase 2b** (§7.2) — added permanent slot-compatibility regression test.
> - **§13 Appendix** — corrected DevMate test-site count (45+, not 15); added BTA `_configure_for_workspace` override at line 1135; documented slot-compatibility verification.

**Author:** Tony Chen (with Rovo Dev assistance)
**Date drafted:** 2026-05-15 (rev. 2026-05-16 06:59 — audit-integrated v2)
**Status:** Ready for review and implementation
**Codebase root:** `CoreProjects/AgentFoundation`
**Scope:**
- `src/agent_foundation/common/inferencers/terminal_inferencers/*` — re-parent Session to Terminal; add `target_path` to Terminal
- `src/agent_foundation/common/inferencers/inferencer_base.py` — surgical removal of the `working_dir = workspace.root` clobber line in `_configure_for_workspace`
- `src/agent_foundation/common/inferencers/streaming_inferencer_base.py` — rename `StreamingInferencerBase` → `TemplatedStreamingInferencerBase` (the class genuinely uses `self.template_manager(...)` in its recovery subsystem; the name should reflect that)
- All 5 production CLI leaves + 10 streaming-base leaves + test stubs + examples + isinstance call sites

---

## 1. Problem statement

Three closely-related architectural problems in the terminal inferencer hierarchy interact to create real bugs and reader confusion. They are:

### P1 — `working_dir` is overloaded across two semantically distinct axes
A single `working_dir` field today carries **two** concepts:

1. **Inferencer artifact home** — written by `InferencerBase._configure_for_workspace()` (`inferencer_base.py:374–384`): `self.working_dir = str(workspace.root)`.
2. **CLI agent operating target** — written by individual leaves (e.g., `DevmateCliInferencer.__attrs_post_init__` line 204 sets `self.working_dir = self.repo_path`; `KiroCliInferencer.__attrs_post_init__` does the same with `target_path`).

Both writers run in `__attrs_post_init__` and the ordering is fragile:
- The leaf sets `working_dir = target_path` *before* `super().__attrs_post_init__()`.
- The base's `__attrs_post_init__` (line 443) then triggers `_configure_for_workspace`, which **overwrites** `working_dir` with `workspace.root`.
- Net effect: when a leaf is constructed with **both** `workspace=ws` and `target_path=/repo`, `working_dir` ends up at `workspace.root`, not `/repo`. The CLI is launched in the wrong directory.

The comment at `inferencer_base.py:380` already names the latent concept: *"leave working_dir at its previously-resolved value (typically target_path)"* — but `target_path` is not formalized as a base-level concept, only a leaf-local one.

### P2 — `TerminalSessionInferencerBase` does not inherit from `TerminalInferencerBase`
The two bases live side-by-side but have separate hierarchies:

```
TemplatedInferencerBase
├── TerminalInferencerBase          (synchronous; full-featured _infer)
└── StreamingInferencerBase
    └── TerminalSessionInferencerBase   (async-streaming; weaker _infer)
```

This produces:

| Concern | TerminalInferencerBase | TerminalSessionInferencerBase |
|---|---|---|
| `working_dir` attrib | declared (line 42) | re-declared (line 71) — duplicate |
| `pre_exec_scripts` | declared (line 49) | re-declared (line 72) — duplicate |
| `_resolve_subprocess_cwd` (Windows MAX_PATH fix) | implemented (lines 60–80) | re-implemented line-for-line (lines 136–156) |
| `_infer` features | timeout, env_vars, pre/post-scripts, output file, fail-on-error | bare `subprocess.run`; **none of those features** |
| `_ainfer_streaming` async pipe-hang detection | absent | sophisticated (lines 281–423) — must be preserved |
| `parse_output` return type | `Any` | `Dict[str, Any]` (wrapped in `TerminalInferencerResponse`) |

Five production CLI leaves (Claude Code, Kiro, Devmate, Metamate, RovoDev) all inherit from the **session** base, so they silently lose Terminal's `timeout`/`env_vars`/`post_exec_scripts`/`_save_output` features. The duplicated `_resolve_subprocess_cwd` is also a maintenance hazard: any Windows fix has to be applied in two places.

### P3 — Templating-axis vs Terminal-axis confusion
`TerminalInferencerBase` currently inherits from `TemplatedInferencerBase`. While defensible (every production CLI leaf wants templating), the inheritance bundles two orthogonal capabilities. A future "CLI leaf that does not want templating" can't easily opt out. Conversely, an inferencer that wants templating without subprocess execution today must pick a different branch of the hierarchy.

We will untangle this with **mixin-style composition** in the leaves (no convenience class proliferation), while preserving the cascade-injection semantics that put template fields on `TemplatedInferencerBase` in the first place.

### P4 — `StreamingInferencerBase` name actively misleads readers
The class is named for what it superficially does ("yield chunks") but its actual implementation **invokes `self.template_manager(...)` as part of its core behavior** (`streaming_inferencer_base.py` line 246, inside `_render_recovery_prompt`). Recovery from partial caches — used on *every* `infer()`/`ainfer()` call through `_try_resume_from_cache` (line 331) and `_atry_resume_from_cache` (line 353), plus the framework's standard `_infer_recovery`/`_ainfer_recovery` (lines 989/927) — depends on rendering Jinja templates. Three of the class's attribs (`use_default_prompt_templates`, `fallback_recovery_template_key`, `fallback_infer_mode`) exist solely to configure that templated recovery subsystem.

The current name hides this dependency. New contributors reasonably assume "Streaming" and "Templated" are orthogonal and try to split them — exactly the conversation that prompted this plan. We will rename the class to `TemplatedStreamingInferencerBase` so the dependency is visible at the class declaration, while preserving back-compat via a module-level alias (`StreamingInferencerBase = TemplatedStreamingInferencerBase`) that emits a `DeprecationWarning`.

The rename is targeted: we do **not** rename `TerminalInferencerBase` even though it also inherits from `TemplatedInferencerBase`. The asymmetry is intentional — `TerminalInferencerBase` does not itself call `self.template_manager(...)`; its templating is for subclass benefit (cascade injection + isinstance dispatch). The current name is conventional, not misleading. `StreamingInferencerBase`'s name *is* misleading, so only it gets renamed.

---

## 2. Goals and non-goals

### Goals
1. **Eliminate the workspace-overwrites-target_path bug** so that `Leaf(workspace=ws, target_path=/repo)` reliably runs the CLI in `/repo` while storing artifacts under `ws.root`.
2. **Unify the terminal hierarchy** so `TerminalSessionInferencerBase` inherits the full feature set of `TerminalInferencerBase` (timeout, env_vars, post-exec scripts, output-file saving, etc.).
3. **Formalize `target_path` as a first-class field on `TerminalInferencerBase`** — not on `InferencerBase` (which would pollute API/orchestrator subtree).
4. **Cleanly separate the templating axis from the terminal axis** so that leaves opt into both via composition, while all current production behavior is preserved.
5. **Preserve session-resumption semantics** for RovoDev / Claude / Kiro — their session metadata is per-CWD, so we must pass the *actual* CLI cwd (which after the refactor is `target_path`) to `find_latest_session_id()`.
6. **Backwards-compatible migration path** — DevMate's `repo_path` survives via a property alias; all 15+ test sites continue to pass unchanged.
7. **Integration tests that pin the post-init ordering** so future drift is caught immediately.

### Non-goals
- We do **not** add `target_path` to `InferencerBase`. It is not a universal concept (API/cloud/orchestrator inferencers have no filesystem cwd).
- We do **not** introduce convenience classes `TerminalTemplatedInferencerBase` / `TerminalSessionTemplatedInferencerBase`. Since every production CLI leaf already wants templating, we keep templating in the existing chain (via `TemplatedInferencerBase` ancestry of the session base) and let any future non-templated CLI leaf opt out via composition.
- We do **not** rename `working_dir` (it remains the subprocess `cwd` field). Only its **default-derivation** changes.
- We do **not** refactor the *behavior* of `StreamingInferencerBase` or `TemplatedInferencerBase` themselves. The only change to the streaming base is a **rename** (`StreamingInferencerBase` → `TemplatedStreamingInferencerBase`) with a back-compat alias. Implementation, fields, methods, MRO order — all preserved.
- We do **not** rename `TerminalInferencerBase` (its name is conventional, not misleading — see §1 P4).
- We do **not** split `StreamingInferencerBase` into a "pure streaming" base + a "templated recovery" mixin. That was considered and rejected — the recovery subsystem is intrinsic to streaming's value-add (see §11 Open Questions, Q6).

---

## 3. Target architecture

### 3.1 Final hierarchy

```
InferencerBase
└── TemplatedInferencerBase                       (unchanged)
    ├── TemplatedStreamingInferencerBase          (RENAMED — was StreamingInferencerBase)
    │       └── ...10 streaming leaves (RovoChat, ToolAs, Plugboard, AgClaude,
    │              OpenClaw, RovoDevServe, ClaudeCodeSdk, DevmateSDK,
    │              MetamateSDK + TerminalSessionInferencerBase)
    └── TerminalInferencerBase                    (gets new target_path attrib)
        └── TerminalSessionInferencerBase         (NEW PARENT — was StreamingInferencerBase)
            ├── ClaudeCodeCliInferencer
            ├── KiroCliInferencer
            ├── DevmateCliInferencer
            ├── MetamateCliInferencer
            └── RovoDevCliInferencer
```

**Back-compat alias (zero-disruption rename):**
```python
# In streaming_inferencer_base.py, at module bottom:
import warnings as _warnings

def __getattr__(name: str):
    if name == "StreamingInferencerBase":
        _warnings.warn(
            "StreamingInferencerBase is deprecated; use "
            "TemplatedStreamingInferencerBase. The old name will be "
            "removed in a future release.",
            DeprecationWarning, stacklevel=2,
        )
        return TemplatedStreamingInferencerBase
    raise AttributeError(name)
```

The module-level `__getattr__` is preferred over a simple `StreamingInferencerBase = TemplatedStreamingInferencerBase` assignment because it lets us emit a `DeprecationWarning` *only* when the old name is actually imported, rather than at module load time.

**Critical wrinkle:** `TerminalSessionInferencerBase` needs *both* `TerminalInferencerBase`'s execution machinery AND `TemplatedStreamingInferencerBase`'s streaming/cache mechanics. We resolve this with **multiple inheritance** at exactly one point:

```python
class TerminalSessionInferencerBase(TerminalInferencerBase, TemplatedStreamingInferencerBase):
    """Multiple inheritance is intentional and safe here:
    - Both parents ultimately derive from TemplatedInferencerBase → InferencerBase
      (a clean diamond on TemplatedInferencerBase — see MRO below).
    - Neither overrides __attrs_post_init__ in a conflicting way:
      TerminalInferencerBase only sets working_dir/target_path defaults;
      TemplatedStreamingInferencerBase's post-init only registers a
      template root for recovery prompts (idempotent).
    - MRO is:
        TerminalSessionInferencerBase
        → TerminalInferencerBase
        → TemplatedStreamingInferencerBase
        → TemplatedInferencerBase
        → InferencerBase
        → (Debuggable, Resumable, ABC, object)
    """
```

This is the **only** diamond in the plan. We deliberately accept it because:
- Both parents derive from `TemplatedInferencerBase`, so the diamond resolves cleanly with C3 linearization.
- `attrs` accepts diamonds; field order follows MRO (verified by prototype in Phase 4).
- The alternative — extracting a `_TerminalExecMixin` — would require renaming/aliasing fields and breaks `isinstance(x, TerminalInferencerBase)` checks elsewhere.
- The rename of `StreamingInferencerBase` → `TemplatedStreamingInferencerBase` (Phase 1.5) makes the diamond *more* legible: both parents are now named honestly, so readers immediately see why the MRO converges on `TemplatedInferencerBase` exactly once.

### 3.2 The `target_path` / `working_dir` / `workspace` contract

After the refactor, the three fields have **strictly disjoint roles**:

| Field | Owns | Lives on | Default rule |
|---|---|---|---|
| `workspace` (`InferencerWorkspace`) | Artifact storage — outputs, logs, cache, checkpoints | `InferencerBase` (already) | `None` (opt-in) |
| `target_path` (`str`) | CLI agent operating directory — where the CLI "operates" semantically | `TerminalInferencerBase` (NEW) | Falls back per the rule below |
| `working_dir` (`str`) | Subprocess `cwd=` for `subprocess.run` / `asyncio.create_subprocess_*` | `TerminalInferencerBase` (already; deduplicated from Session) | Defaults to `target_path` |

### 3.3 Default-resolution rule (deterministic, single owner)

In `TerminalInferencerBase.__attrs_post_init__`:

```
Step 1 (target_path resolution):
    if target_path is None:
        target_path = os.getcwd()
    # Note: do NOT auto-set target_path from workspace.root. They are
    # semantically distinct. If a leaf wants to bind them, it should do
    # so explicitly.

Step 2 (working_dir resolution):
    if working_dir is None:
        working_dir = target_path

Step 3 (call super().__attrs_post_init__()):
    This triggers InferencerBase.__attrs_post_init__, which triggers
    _configure_for_workspace IF workspace was provided. After our edit
    (see Phase 1), that method NO LONGER overwrites working_dir.
```

### 3.4 What `_configure_for_workspace` does after the edit

```python
def _configure_for_workspace(self, workspace):
    # REMOVED: self.working_dir = str(workspace.root)
    # working_dir is now owned by TerminalInferencerBase and defaults to
    # target_path. Workspace controls artifact paths only.

    if hasattr(self, "cache_folder"):
        self.cache_folder = os.path.join(
            str(workspace.root), "_runtime", "inferencer_cache"
        )

    # Logger redirection — unchanged.
    logger_val = getattr(self, "logger", None)
    if isinstance(logger_val, str) and logger_val == "auto":
        self._normalize_loggers()
    elif getattr(self, "_logger_awaiting_workspace", False):
        self._logger_awaiting_workspace = False
        self._add_workspace_logger(workspace)
    elif isinstance(logger_val, dict):
        self._redirect_loggers_to_workspace(workspace)
```

---

## 4. Phased implementation plan

The work is split into **six phases**, each independently shippable. Phases 0–2 are mandatory; Phase 1.5 (rename) is mandatory but order-flexible (can land before or after Phase 1 — they touch disjoint files); Phase 3 is the convenience cleanup; Phase 4 is the validation harness.

| Phase | Title | Risk | Lines changed (est.) | Tests added |
|---|---|---|---|---|
| 0 | Pre-flight: integration test that pins post-init ordering (RED) | low | +120 (test only) | 6 |
| 1 | Surgical fix to `_configure_for_workspace` | medium | -2 / +12 src; +30 tests | 4 |
| **1.5** | **`StreamingInferencerBase` → `TemplatedStreamingInferencerBase` rename + back-compat alias** | **low-medium** | **~+60 src across ~25 files; +15 tests** | **3** |
| 2 | `target_path` on `TerminalInferencerBase` + Session re-parenting (under the new name) | medium-high | ~+150 src; ~+200 tests | 8 |
| 3 | Leaf migrations (Devmate `repo_path` → alias; Kiro/Claude/Rovodev/Metamate) | low-medium | ~+80 src; ~+60 tests | 5 |
| 4 | Session-resumption regression suite + final validation | low | +180 tests | 6 |

**Ordering note:** Phase 1.5 may land *before* Phase 1 if reviewers prefer the smallest surgical edit (the rename) to ship first as a confidence-builder. Phase 2 *must* land after Phase 1.5 because the Phase 2 diamond declaration uses the new name. Phase 0 tests should be authored using the new name so they don't need to be edited when Phase 1.5 ships — if Phase 1.5 hasn't landed yet, the tests import via the alias and document the expected post-rename name in a comment.

---

## 5. Phase 0 — Pre-flight tests (RED)

**Purpose:** Before changing any source, write tests that *pin the current buggy behavior as a failing assertion*. This guarantees we will know exactly when each subsequent phase fixes what it claims to fix, and provides a regression net.

**New test file:**
`test/agent_foundation/common/inferencers/terminal_inferencers/test_target_path_workspace_contract.py`

**Nine required test cases (v2 — added 3 orchestrator-scenario tests per audit C6; all expected to start RED, turn GREEN as phases land):**

| # | Test | Pre-fix expectation | Post-refactor expectation |
|---|------|----------------------|----------------------------|
| 1 | `test_target_path_survives_workspace_assignment` — construct `KiroCliInferencer(target_path="/tmp/repo", workspace=InferencerWorkspace(root="/tmp/ws"))`, assert `inf.working_dir == "/tmp/repo"`. | **FAIL** (working_dir == "/tmp/ws") | PASS (after Phase 2a — needs `target_path` field) |
| 2 | `test_working_dir_explicit_user_wins` — construct with `working_dir="/explicit"`, assert it's preserved through workspace assignment. | FAIL | PASS |
| 3 | `test_target_path_defaults_to_cwd_when_no_workspace` — no workspace, no target_path → target_path == os.getcwd(); working_dir == target_path. | depends | PASS |
| 4 | `test_workspace_only_controls_artifacts_not_cwd` — workspace set, target_path unset → cache_folder under workspace.root; working_dir is the leaf's defaulted value (target_path or workspace.root via orchestrator rule). | depends | PASS |
| 5 | `test_session_inherits_terminal_features` — assert `TerminalSessionInferencerBase` has `timeout`, `env_vars`, `post_exec_scripts` attribs (inherited). | FAIL (attribs don't exist) | PASS |
| 6 | `test_session_inferencer_subprocess_uses_target_path_not_workspace` — patch `asyncio.create_subprocess_shell`, instantiate `KiroCliInferencer(target_path="/repo", workspace=ws)`, run, assert `cwd="/repo"`. | FAIL | PASS |
| **7** | **`test_orchestrator_propagated_workspace_sets_child_working_dir` (audit C6)** — construct `BreakdownThenAggregateInferencer(workspace=root_ws)` with a child terminal inferencer that has `target_path=None`. Run. Assert child's runtime `working_dir == child_workspace.root`. | passes today (clobber fires) | **PASS** (the v2 conditional clobber preserves this) |
| **8** | **`test_orchestrator_propagated_workspace_does_not_override_explicit_target_path` (audit C6)** — same setup but child has `target_path="/explicit"`. Assert child's runtime `working_dir == "/explicit"` (NOT `child_workspace.root`). | **FAIL** today (clobber overrides) | PASS (the v2 conditional clobber respects explicit value) |
| **9** | **`test_devmate_repo_path_kwarg_still_accepted_as_init_arg` (audit C6)** — `DevmateCliInferencer(repo_path="/x")` constructs successfully (no `TypeError`); asserts `inst.repo_path == "/x"` and `inst.target_path == "/x"` (mirror). | passes today (repo_path is attrib) | PASS (v2 keeps repo_path as attrib + mirrors) |

**Acceptance criterion for Phase 0:** All 9 tests merged in `xfail(strict=True)` (so they alert when they unexpectedly pass) or with explicit `# WILL_PASS_AFTER_PHASE_N` markers. CI must run them.

**Why tests #7 and #8 matter (and were missing in v1):** The v1 plan only tested standalone construction. Test #7 is the regression guard that would have caught the v1 mistake of unconditional-clobber-deletion. Test #8 is the positive case for the v2 conditional logic. Together they form the contract that Phase 1 must honor.

---

## 6. Phase 1 — Surgical fix to `_configure_for_workspace`

**Goal:** Remove the workspace-overwrites-working_dir clobber as a stand-alone change. This is the smallest possible diff and validates the core hypothesis (that `working_dir` shouldn't be touched by the workspace setter) before we restructure the hierarchy.

### 6.1 File: `src/agent_foundation/common/inferencers/inferencer_base.py`

> **⚠️ CRITICAL — v2 redesign (audit correction C1):** The v1 plan said "delete the clobber unconditionally." That was **wrong**. The clobber is *load-bearing* for orchestrators (BTA, MFDual, LWI, Dual) that spawn terminal inferencers and call `child._workspace = parent_ws.child(...)` at runtime. Lines 345–353 of `inferencer_base.py` document the failure mode explicitly: removing the clobber unconditionally would cause `NotADirectoryError` on subprocess launch when the child was constructed with no `target_path` (it defaulted to `os.getcwd()` at config-load time, which is not the per-iteration workspace).
>
> **The v2 fix:** make the clobber *conditional* on whether the user explicitly set `target_path`. Explicit `target_path` wins (fixes the latent bug from §1 P1). No explicit `target_path` → preserve the orchestrator-spawning behavior.

**Edit `_configure_for_workspace` (lines 366–398):**

Replace:
```python
if hasattr(self, "working_dir"):
    new_wd = str(workspace.root)
    if sys.platform != "win32" or len(new_wd) < 240:
        self.working_dir = new_wd
```

With:
```python
if hasattr(self, "working_dir"):
    # CONDITIONAL clobber (audit-corrected 2026-05-16):
    # The original unconditional clobber overwrote user-specified
    # target_path with workspace.root. The original "delete entirely"
    # fix broke orchestrator scenarios where children are constructed
    # at config-load time (target_path defaults to os.getcwd()) and
    # later assigned a per-iteration workspace via setter — in those
    # cases the subprocess MUST use the new workspace.root as cwd.
    #
    # New rule: only auto-set working_dir from workspace.root if the
    # user did NOT explicitly set target_path. The discriminator is
    # whether target_path differs from os.getcwd() (the default).
    #   - target_path EXPLICIT (!= cwd default) → leave working_dir alone (user wins)
    #   - target_path IMPLICIT (None or == cwd default) AND workspace
    #     provided → set working_dir = workspace.root (orchestrator path)
    target_path = getattr(self, "target_path", None)
    target_was_explicit = (
        target_path is not None
        and target_path != os.getcwd()
    )
    if not target_was_explicit:
        new_wd = str(workspace.root)
        # Windows CreateProcessW enforces MAX_PATH=260 on lpCurrentDirectory
        # regardless of LongPathsEnabled. When workspace.root exceeds the
        # limit, leave working_dir at its prior value; framework writes
        # use workspace.root explicitly via resolve_output_path().
        if sys.platform != "win32" or len(new_wd) < 240:
            self.working_dir = new_wd
```

Note: this requires `import os` at the top of the function (already present at line 372 today).

**Why this v2 logic is correct:**
- **Standalone construction with explicit target_path:** User wins. `working_dir` stays at `target_path`. Fixes the §1 P1 bug.
- **Standalone construction without target_path + with workspace:** Behaves as today — `working_dir` = `workspace.root`. No regression.
- **Orchestrator child spawned with `child._workspace = parent_ws.child(...)`:** Child was constructed at config-load time with `target_path=None` (defaulted to `os.getcwd()` later in `TerminalInferencerBase.__attrs_post_init__`). When the setter fires, `target_path == os.getcwd()` (the default), so `target_was_explicit == False`, so the clobber fires. **Subprocess launches in the new workspace.root.** Identical to today's behavior. No `NotADirectoryError`.
- **Trade-off:** if a user *explicitly* passes `target_path=os.getcwd()`, we can't distinguish from the default. Acceptable degenerate case (both behaviors are reasonable).

**Why the conditional is checked at workspace-assignment time (not construction time):**
- `_configure_for_workspace` runs every time the `_workspace` setter is called, both during construction (via `__attrs_post_init__` at line 443) and during orchestrator runtime propagation (via `_propagate_workspace_to_children` at line 354). The conditional fires correctly in both contexts.
- The `target_path` is set during `TerminalInferencerBase.__attrs_post_init__` (Phase 2a) *before* `super().__attrs_post_init__()` triggers the first workspace assignment. So by the time `_configure_for_workspace` reads `getattr(self, "target_path", None)`, it always sees the post-defaulted value.

**Why this is still safe to ship alone (before Phase 2):**
- Phase 1 ships before Phase 2a, so `target_path` doesn't exist as an attrib yet. `getattr(self, "target_path", None)` returns `None`, so `target_was_explicit` is always `False`, so the clobber always fires — **identical to today's unconditional behavior**. No behavior change in standalone Phase 1.
- The new behavior only activates once Phase 2a adds `target_path` to `TerminalInferencerBase`. This is intentional: Phase 1 is a refactor-prep change that becomes meaningful after Phase 2a.
- `cache_folder` and logger redirection are unchanged → artifact storage continues working.

### 6.2 Update `_propagate_workspace_to_children` (lines 301–360)

The propagation logic at line 354 (`child._workspace = child_ws`) will now NOT clobber child working_dir. This is the desired behavior. The pre-existing comment at line 348 about "ensure dirs before assigning so cwd=workspace.root works" becomes stale — update to:

```python
# Critical: create the on-disk dirs before assigning so artifact paths
# (cache, logs, outputs) exist when the child writes to them. Note:
# working_dir is NO LONGER set from workspace.root (post-2026-05-XX
# axes refactor). Child terminal inferencers default working_dir to
# their own target_path; orchestrators that need a child CLI to run
# in workspace.root should pass target_path=child_ws.root explicitly.
```

### 6.3 Tests landing in Phase 1

> **v2 note:** Phase 1's behavior change only activates once Phase 2a adds `target_path` to `TerminalInferencerBase` (because the conditional reads `getattr(self, "target_path", None)`). So the Phase 0 tests that pass `target_path=` (tests #1, #2, #6) only turn GREEN after Phase 2a lands. Phase 1 by itself preserves today's behavior; Phase 1's value is the *infrastructure* (the conditional) that Phase 2a activates.

Two of the Phase 0 tests turn GREEN (after Phase 2a):
- `test_target_path_survives_workspace_assignment` (#1)
- `test_working_dir_explicit_user_wins` (#2)

Plus 4 new positive tests in the same file (land with Phase 1):
- `test_workspace_still_configures_cache_folder` — assert `cache_folder == workspace.root/_runtime/inferencer_cache`.
- `test_workspace_still_configures_logger` — assert "auto" logger resolves to `workspace.logs_dir`.
- `test_workspace_still_sets_working_dir_when_no_target_path` — **regression test for the orchestrator scenario**. Construct a terminal leaf with no `target_path`, assign workspace → assert `working_dir == workspace.root`. This protects against accidental regression to the broken "unconditional delete" version.
- `test_old_behavior_can_be_opted_into_explicitly` — `Leaf(workspace=ws, working_dir=ws.root)` still works.

**One additional integration test (audit C6):**
- `test_orchestrator_spawned_terminal_inferencer_runs_in_workspace_root` — use a real `BreakdownThenAggregateInferencer` with a child `KiroCliInferencer` (target_path=None). Patch the subprocess. Assert child's subprocess `cwd` equals the per-iteration workspace child path.

### 6.4 Phase 1 risk register (v2 — post-audit)

| Risk | Mitigation |
|---|---|
| **The v1 "delete unconditionally" plan would have broken orchestrators with `NotADirectoryError`.** Audit caught this; v2 logic preserves orchestrator behavior. | v2 conditional clobber + `test_workspace_still_sets_working_dir_when_no_target_path` regression test + orchestrator integration test (C6). |
| A user passes `target_path=os.getcwd()` explicitly, expecting it to be honored, but the conditional treats it as the default. | Degenerate case — both behaviors are reasonable (workspace.root or cwd-when-they-equal). Document in field docstring. If this becomes a real complaint, add a `_target_path_user_set: bool = attrib(default=False, init=False)` companion flag. |
| `_propagate_workspace_to_children` semantics silently change for nested orchestrator hierarchies. | Identical behavior preserved by v2 conditional (when target_path is None/cwd-default, working_dir still tracks workspace.root). Add the BTA integration test (C6). |
| Tests that *implicitly* relied on the clobber by passing only `workspace=` and expecting CLI to run there. | v2 preserves this behavior. Tests continue to pass. |
| BTA's own `_configure_for_workspace` override (`breakdown_then_aggregate_inferencer.py:1135`) calls `super()` and depends on it setting working_dir for the breakdown child. | v2 preserves this behavior. The breakdown child is spawned with `target_path=None`, so the conditional fires and working_dir tracks workspace.root as before. Add explicit test: `test_bta_breakdown_child_working_dir_matches_breakdown_workspace`. |

---

## 6A. Phase 1.5 — `StreamingInferencerBase` → `TemplatedStreamingInferencerBase` rename

**Goal:** Rename the class so its name reflects the templating dependency it actually has (see §1 P4). Keep `StreamingInferencerBase` working as a deprecated alias so no downstream consumer is broken on day one.

### 6A.1 File: `src/agent_foundation/common/inferencers/streaming_inferencer_base.py`

**Change the class declaration (line 103):**

```python
# OLD:
# @attrs
# class StreamingInferencerBase(TemplatedInferencerBase):

# NEW:
@attrs
class TemplatedStreamingInferencerBase(TemplatedInferencerBase):
    """Streaming inferencer base with **templated** recovery prompts.

    This class is named "Templated…" because its recovery subsystem
    (`_render_recovery_prompt`, `_try_resume_from_cache`,
    `_atry_resume_from_cache`, `_infer_recovery`, `_ainfer_recovery`)
    calls `self.template_manager(...)` directly to build continuation
    prompts from partial caches. Without a `template_manager`, those
    paths fall through to a module-level fallback or return None;
    behavior degrades gracefully but the templating dependency is
    intrinsic to the class's value-add.

    See §1 P4 of terminal_inferencer_axes_and_streaming_rename_plan.md
    for the full rationale.
    """
    # ... body unchanged ...
```

**No other behavior changes inside this class.** Field set, methods, attribs, MRO contributions — all preserved.

**Add module-level back-compat alias at the bottom of the file:**

```python
# ── PERMANENT back-compat alias (added 2026-05-XX) ────────────────────
# The class was renamed from StreamingInferencerBase to
# TemplatedStreamingInferencerBase to reflect its actual dependency on
# the templating subsystem (used in _render_recovery_prompt). The old
# name is exported via module __getattr__ (PEP 562, Python 3.7+) so
# existing imports continue to work.
#
# *** THIS ALIAS IS PERMANENT (audit correction C4) ***
# Pickled instances of (renamed) TemplatedStreamingInferencerBase from
# before the rename store __qualname__='StreamingInferencerBase';
# unpickling them resolves the class via this __getattr__. Removing
# the alias would break unpickling of historical artifacts. DO NOT
# REMOVE this alias in a future cleanup PR.
#
# The DeprecationWarning is informational only (does NOT signal future
# removal of the alias itself — only encourages migration of new code
# to use the canonical name).
import warnings as _warnings

def __getattr__(name: str):
    if name == "StreamingInferencerBase":
        _warnings.warn(
            "StreamingInferencerBase has been renamed to "
            "TemplatedStreamingInferencerBase to reflect its templating "
            "dependency. New code should use the canonical name. "
            "(The alias itself is permanent for pickle compatibility — "
            "this warning is migration guidance, not a removal notice.) "
            "See terminal_inferencer_axes_and_streaming_rename_plan.md §6A.",
            DeprecationWarning, stacklevel=2,
        )
        return TemplatedStreamingInferencerBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Note on `__getattr__` caching semantics:** Module `__getattr__` is invoked only when a *direct attribute lookup* on the module fails. Once a name is bound locally via `from x import Name`, subsequent uses of `Name` are local-namespace lookups and do **not** re-trigger `__getattr__`. This means:
- `from ... import StreamingInferencerBase` → 1 warning (at import time).
- `isinstance(x, StreamingInferencerBase)` in a loop → 0 additional warnings (the name is a local binding).
- `sys.modules['...streaming_inferencer_base'].StreamingInferencerBase` → 1 warning *per access* (this anti-pattern is grep-checked).

A test (§6A.4) pins this behavior so a future contributor can't accidentally introduce the anti-pattern.

### 6A.2 Update all in-tree imports (eager migration)

Even though the alias preserves back-compat for *external* consumers, we **eagerly migrate every in-tree import** in this same PR. This avoids leaving the codebase in a state where some sites use the new name and others use the deprecated one. Specifically:

- **10 production subclasses** — update `class Foo(StreamingInferencerBase):` → `class Foo(TemplatedStreamingInferencerBase):`:
  - `agentic_inferencers/external/devmate/devmate_sdk_inferencer.py:26`
  - `agentic_inferencers/tool_inferencers/tool_as_inferencer.py:93`
  - `agentic_inferencers/external/metamate/metamate_sdk_inferencer.py:50`
  - `agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py:30`
  - `agentic_inferencers/external/openclaw/openclaw_inferencer.py:81`
  - `api_inferencers/plugboard/plugboard_api_inferencer.py:23`
  - `api_inferencers/ag/ag_claude_api_inferencer.py:82`
  - `agentic_inferencers/external/rovochat/rovochat_inferencer.py:104`
  - `agentic_inferencers/external/rovodev/rovodev_serve_inferencer.py:43`
  - `terminal_inferencers/terminal_session_inferencer_base.py:57` *(also re-parented in Phase 2b)*

- **6 example/test stubs:**
  - `examples/.../recovery/example_04_custom_recovery_templates.py:92`
  - `examples/.../recovery/example_02_cache_based_recovery.py:99`
  - `examples/.../recovery/example_03_fallback_inferencer_chain.py:83`
  - `examples/.../recovery/example_01_basic_retry_recovery.py:75`
  - `test/.../test_streaming_inferencer_dual_timer.py:25`
  - `test/.../test_streaming_recovery.py:75`
  - `test/.../test_inferencer_resumable_integration.py:36`
  - `test/.../test_inferencer_resumable.py:58`
  - `test/.../test_get_final_output.py:61, 278`

- **isinstance / hasattr usages:** grep `StreamingInferencerBase` across the tree; rename each call site.

### 6A.3 Filename rename (deferred to follow-up PR)

The filename `streaming_inferencer_base.py` stays unchanged in Phase 1.5 for review-noise reasons. A follow-up PR (post-Phase-4) may rename the file to `templated_streaming_inferencer_base.py` and remove the deprecation alias once external consumers have migrated.

### 6A.4 Tests landing in Phase 1.5

1. `test_templated_streaming_alias_import_emits_deprecation_warning` — `pytest.warns(DeprecationWarning, match="StreamingInferencerBase has been renamed")` when importing the old name.
2. `test_old_name_resolves_to_new_class` — `from ...streaming_inferencer_base import StreamingInferencerBase; assert StreamingInferencerBase is TemplatedStreamingInferencerBase`.
3. `test_isinstance_check_works_for_both_names` — instantiate a subclass; assert `isinstance(inst, StreamingInferencerBase)` AND `isinstance(inst, TemplatedStreamingInferencerBase)` both return True.
4. **`test_isinstance_loop_does_not_spam_warnings` (audit C3)** — pins the `__getattr__` caching contract:
   ```python
   def test_isinstance_loop_does_not_spam_warnings():
       from agent_foundation.common.inferencers.streaming_inferencer_base import (
           StreamingInferencerBase,  # one warning here (import-time)
       )
       inst = _SomeStreamingSubclass()
       with warnings.catch_warnings(record=True) as records:
           warnings.simplefilter("always")
           for _ in range(1000):
               assert isinstance(inst, StreamingInferencerBase)
       # Local binding caches the resolved name; isinstance does NOT
       # re-trigger module __getattr__. Expect ZERO additional warnings
       # inside the loop (the import-time warning fired before this block).
       relevant = [r for r in records if "StreamingInferencerBase has been renamed" in str(r.message)]
       assert len(relevant) == 0, (
           f"Expected 0 warnings inside the isinstance loop, got {len(relevant)} — "
           "module __getattr__ is being re-triggered. Check for sys.modules access patterns."
       )
   ```
5. **`test_pickle_round_trip_after_rename`** — `pickle.loads(pickle.dumps(inst))` succeeds and returns an instance whose class is `TemplatedStreamingInferencerBase`. Documents the pickle-compat guarantee.

### 6A.5 Phase 1.5 risk register

| Risk | Mitigation |
|---|---|
| External consumers of `AgentFoundation` (other repos under `CoreProjects/`) import `StreamingInferencerBase` directly. | Module `__getattr__` alias preserves the import; `DeprecationWarning` (not `ImportError`) lets them migrate at their own pace. Document in `MIGRATION_NOTES.md`. |
| Pickled instances persisted with `type(self).__name__ == "StreamingInferencerBase"` can't be unpickled to the new class. | The `__name__` of the *new* class is `TemplatedStreamingInferencerBase`, but pickled instance state is keyed by the qualified class path, not the name. Since the alias preserves the import path, unpickling still works. Add a regression test using `pickle.loads(pickle.dumps(inst))`. |
| `output_manifest.json` (line 1003 of `inferencer_base.py`) records `produced_by: type(self).__name__`. Old manifests will read `"StreamingInferencerBase"`; new ones will read `"TemplatedStreamingInferencerBase"`. | Cosmetic only — no programmatic consumer depends on the string equality. Document in `MIGRATION_NOTES.md`. |
| Logger names that embed class name might change (e.g. `f"[{type(self).__name__}]"`). | Cosmetic; log output strings change but no parser depends on them. |
| YAML configs that reference the class by string (`_target_: agent_foundation....StreamingInferencerBase`). | The Hydra/instantiate machinery resolves via `importlib.import_module` + `getattr` — both still work via the alias. Search configs anyway: `grep -r "StreamingInferencerBase" --include="*.yaml" --include="*.yml"`. Migrate hits in the same PR. |
| The `DeprecationWarning` floods CI logs with noise during the migration. | Make the warning fire **once per import path** (Python's default DeprecationWarning behavior). Configure pytest with `filterwarnings = ["once::DeprecationWarning"]` for this PR. |
| Sphinx / autodoc references the old name. | Sphinx will pick up the new class name automatically. Update any hand-written `:class:` cross-references via grep. |

### 6A.6 Why this is the right rename, and the only rename

The asymmetry with `TerminalInferencerBase` (which we are *not* renaming) is intentional and worth a one-line summary for reviewers:

| Class | Does it call `self.template_manager(...)` in its own body? | Rename? |
|---|---|---|
| `StreamingInferencerBase` | **Yes** (line 246 `_render_recovery_prompt`). | **Yes — name was hiding the dependency.** |
| `TerminalInferencerBase` | No (templating exists only for cascade injection + leaf benefit). | No — current name is honest. |

---

## 7. Phase 2 — `target_path` on `TerminalInferencerBase` + Session re-parenting

This is the structural phase. It depends on Phase 1.5 having landed (so the new class name is available) and Phase 1 (so `_configure_for_workspace` no longer clobbers `working_dir`). It is split into two **independently mergeable sub-phases**:

- **Phase 2a:** Add `target_path` to `TerminalInferencerBase` (no Session changes).
- **Phase 2b:** Re-parent `TerminalSessionInferencerBase` to multi-inherit from `TerminalInferencerBase` + `TemplatedStreamingInferencerBase`.

Both must be on the same branch for production leaves to benefit, but they can be reviewed as separate commits.

### 7.1 Phase 2a — `target_path` on `TerminalInferencerBase`

**File:** `src/agent_foundation/common/inferencers/terminal_inferencers/terminal_inferencer_base.py`

**Add new attrib (after `working_dir` at line 42):**

```python
# === Operating target (CLI agent's semantic working directory) ===
# The directory the CLI agent operates on (e.g., the repo it edits).
# Distinct from:
#   - workspace.root: where THIS inferencer stores its artifacts
#                     (cache, logs, outputs).
#   - working_dir:    the subprocess cwd= for command execution.
# Defaults to os.getcwd() if unset. working_dir defaults to target_path
# unless explicitly overridden.
#
# Why on TerminalInferencerBase (and not InferencerBase): only terminal
# inferencers shell out to a CLI with a meaningful operating directory.
# API/cloud/orchestrator inferencers have no analogous concept.
target_path: Optional[str] = attrib(default=None)
```

**Replace `__attrs_post_init__` (lines 54–58):**

```python
def __attrs_post_init__(self):
    """Resolve target_path and working_dir BEFORE calling super.

    Order matters:
      1. target_path defaults to cwd if unset.
      2. working_dir defaults to target_path if unset.
      3. super() runs InferencerBase.__attrs_post_init__ which triggers
         _configure_for_workspace (which after the 2026-05-XX edit
         no longer touches working_dir — see plan §6).
    """
    if self.target_path is None:
        self.target_path = os.getcwd()
    if self.working_dir is None:
        self.working_dir = self.target_path
    super().__attrs_post_init__()
```

**No other changes to `TerminalInferencerBase` in Phase 2a.** All existing methods (`_execute_command`, `_execute_scripts`, `_infer`, `_save_output`, `_resolve_subprocess_cwd`) are untouched.

**Tests landing in Phase 2a:**
- `test_target_path_default_is_cwd`
- `test_working_dir_defaults_to_target_path`
- `test_target_path_explicit_overrides_cwd_default`
- `test_target_path_independent_of_workspace` (full triangle: workspace, target_path, working_dir all different).

### 7.2 Phase 2b — Re-parent `TerminalSessionInferencerBase`

**File:** `src/agent_foundation/common/inferencers/terminal_inferencers/terminal_session_inferencer_base.py`

**Change parent class** (line 56) — assumes Phase 1.5 has landed:
```python
# OLD:
# class TerminalSessionInferencerBase(StreamingInferencerBase):

# NEW:
class TerminalSessionInferencerBase(
    TerminalInferencerBase, TemplatedStreamingInferencerBase,
):
    """Async-streaming terminal inferencer with session management.

    Multiple inheritance is intentional:
      - TerminalInferencerBase contributes subprocess execution machinery
        (working_dir, target_path, pre/post_exec_scripts, env_vars,
        timeout, _execute_command, _resolve_subprocess_cwd, _save_output).
      - TemplatedStreamingInferencerBase (renamed from StreamingInferencerBase
        in Phase 1.5) contributes the streaming/cache scaffolding
        (ainfer_streaming, idle_timeout, partial-output cache, _ainfer
        accumulation, active_session_id property) AND the templated
        recovery subsystem (_render_recovery_prompt + cache resume).

    Both parents derive from TemplatedInferencerBase → InferencerBase, so
    the diamond resolves via C3 MRO with no duplicate field issues
    (verified in test_attrs_field_order_under_diamond).

    __attrs_post_init__ chain (each layer calls super exactly once):
      1. TerminalInferencerBase: sets target_path/working_dir defaults.
      2. TemplatedStreamingInferencerBase: registers recovery template
         root (idempotent if already registered or template_manager is None).
      3. TemplatedInferencerBase: no override (inherits InferencerBase's).
      4. InferencerBase: workspace sync + post-processor + warnings.

    MRO:
      TerminalSessionInferencerBase
      → TerminalInferencerBase
      → TemplatedStreamingInferencerBase
      → TemplatedInferencerBase
      → InferencerBase
      → (Debuggable, Resumable, ABC, object)
    """
```

**Remove duplicated attribs (lines 71–72):**
```python
# DELETED — now inherited from TerminalInferencerBase:
#   working_dir: Optional[str] = attrib(default=None)
#   pre_exec_scripts: Optional[List[str]] = attrib(default=None)
# target_path is also inherited (added in Phase 2a).
```

**Remove duplicated method (lines 136–156):**
```python
# DELETED — now inherited from TerminalInferencerBase._resolve_subprocess_cwd.
```

**Decide on `_infer` strategy.** Session's current `_infer` (lines ~474–498) is a thin `subprocess.run` wrapper. Terminal's `_infer` (lines ~474–555) is the rich orchestrated version with pre/post-scripts, timeout, env_vars, output file. **We adopt Terminal's `_infer` and delete Session's override.** Justification:

- All session leaves go through `ainfer_streaming` for normal operation. `_infer` is the synchronous fallback / one-shot path.
- Terminal's version is strictly more capable. Inheriting it gives Session subclasses access to features they didn't have.
- The only concern is `parse_output` return type. Session's `_infer` wraps in `TerminalInferencerResponse.from_dict(...)`. Terminal's returns `Any`. To preserve Session leaves' behavior, we add a hook:

```python
# In TerminalSessionInferencerBase, override only the post-parse wrapping:
def _wrap_parse_output(self, parsed: Dict[str, Any]) -> TerminalInferencerResponse:
    return TerminalInferencerResponse.from_dict(parsed)

# Then have BOTH bases' _infer call self._wrap_parse_output(self.parse_output(...))
# default impl on TerminalInferencerBase returns parsed unchanged.
```

This is a 4-line addition to `TerminalInferencerBase._infer` and a 3-line `_wrap_parse_output` override on Session. Net deletion: ~25 lines from Session.

**Preserve Session's async streaming.** Session's `_ainfer_streaming`, `_read_stdout_with_exit_detection`, `_poll_process_exit`, `_force_close_pipes`, `_kill_process_group`, `_safe_process_cleanup`, `_build_full_command`, `_ainfer` accumulation, and `_infer_streaming`/`_ainfer_streaming` overloads **all stay** — these are the value-add Session brings on top of Terminal.

**Tests landing in Phase 2b:**
- `test_session_inherits_timeout_from_terminal`
- `test_session_inherits_env_vars_from_terminal`
- `test_session_inherits_post_exec_scripts_from_terminal`
- `test_session_pipe_hang_detection_still_works` — regression for the MCP hang fix.
- `test_attrs_field_order_under_diamond` — instantiate `TerminalSessionInferencerBase` and assert all expected fields are present; verify MRO field collection.
- `test_session_parse_output_still_wraps_in_response_object`.
- `test_mro_is_what_we_think` — assert `TerminalSessionInferencerBase.__mro__` equals the documented order.
- `test_session_no_duplicate_working_dir_attrib` — `attrs.fields(TerminalSessionInferencerBase)` should contain exactly one `working_dir` entry.
- **`test_diamond_attrs_slots_consistency` (audit C8)** — permanent regression test:
   ```python
   def test_diamond_attrs_slots_consistency():
       """All five classes in the TerminalSessionInferencerBase MRO use
       compatible @attrs slot settings (all slots=False under the legacy
       `from attr import attrs` API). If a future contributor migrates
       any of them to slots=True (e.g., by switching to the modern
       @attrs.define decorator), this test fires immediately so they
       catch the diamond-incompatibility before runtime.
       """
       classes_in_diamond = [
           InferencerBase,
           TemplatedInferencerBase,
           TemplatedStreamingInferencerBase,
           TerminalInferencerBase,
           TerminalSessionInferencerBase,
       ]
       for cls in classes_in_diamond:
           # Legacy attr.attrs defaults to slots=False; modern attrs.define
           # defaults to slots=True. Mixing them in a diamond is broken.
           # Asserting __slots__ is absent or empty catches the mismatch.
           slots = getattr(cls, "__slots__", None)
           assert slots is None or slots == (), (
               f"{cls.__name__} has non-empty __slots__={slots!r} — "
               "diamond inheritance with mixed slots will break MRO-based "
               "field resolution. Stay on the legacy `from attr import attrs` "
               "API for all five classes in this MRO."
           )
   ```

### 7.3 Phase 2 risk register

| Risk | Mitigation |
|---|---|
| `attrs` diamond produces unexpected field order. | Phase 0 / Phase 2b explicit `test_attrs_field_order_under_diamond`. Run with `attrs.fields(cls)` introspection. |
| `__attrs_post_init__` MRO chain calls `TerminalInferencerBase`'s but skips `TemplatedStreamingInferencerBase`'s side effects (or vice versa). | `TemplatedStreamingInferencerBase` DOES define `__attrs_post_init__` (line 222 of streaming_inferencer_base.py) — it calls `super().__attrs_post_init__()` first, then registers the recovery template root. With the diamond, MRO runs: `TerminalInferencerBase.__attrs_post_init__` → `super()` → `TemplatedStreamingInferencerBase.__attrs_post_init__` → `super()` → `TemplatedInferencerBase` (no override) → `InferencerBase.__attrs_post_init__`. Each layer runs exactly once. Add explicit MRO assertion test AND a test that asserts the recovery template root is registered when `template_manager` is set. |
| Session leaves override `_infer` or `_ainfer` and the new Terminal `_infer` breaks them. | Audit: per Phase 0 investigation, ClaudeCodeCli and KiroCli both override `_infer`/`ainfer`. Confirm their overrides do not call `super()._infer` in a way that would now hit Terminal's version unexpectedly. |
| The `_wrap_parse_output` hook introduces a behavioral change for synchronous Terminal-only subclasses. | Default impl returns parsed unchanged. All 5 test stubs in `test_terminal_inferencer_base.py` continue to pass. |
| `pre_exec_scripts` semantics differ between Terminal (`_execute_scripts` separate run, `fail_on_pre_script_error` flag) and Session (`_build_full_command` chains via `&&`). | **Real divergence — must reconcile.** Decision: Session keeps `_build_full_command` for streaming paths (where pre-scripts must be in the same shell as the main command for env var propagation). Terminal's `_execute_scripts` continues to be used for the sync `_infer` path. Document the asymmetry explicitly in `_build_full_command` docstring. |
| `large_input_mode` (Session-specific) interacts with Terminal's `_execute_command` signature. | `large_input_mode` is only consumed by Session's `_ainfer_streaming`. Terminal's `_execute_command` is untouched. No interaction. |

---

## 8. Phase 3 — Leaf migrations

### 8.1 `DevmateCliInferencer` — `repo_path` mirrored into `target_path` (v2)

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/devmate/devmate_cli_inferencer.py`

> **⚠️ CRITICAL — v2 redesign (audit correction C2):** The v1 plan proposed replacing `repo_path: Optional[str] = attrib(default=None)` with a `@property`. **That would not work.** `attrs` generates `__init__` from the `attrib()` declarations; once you remove the `attrib()`, the generated `__init__` no longer accepts `repo_path=` as a kwarg. **All 45+ existing test sites passing `repo_path=` would fail with `TypeError: __init__() got an unexpected keyword argument 'repo_path'` *before* Python even sees the property.**
>
> **The v2 fix:** keep `repo_path` as the canonical `attrib` (it remains the source of truth at construction time). Mirror its value into `target_path` in `__attrs_post_init__` *before* calling `super()` so the base's defaulting logic sees the user-intended path. Both `repo_path` and `target_path` are present on Devmate instances; either may be used by callers. `repo_path` is the historical Devmate name; `target_path` is the framework-canonical name.

**Keep** the `repo_path` attrib (line 163) — UNCHANGED:

```python
repo_path: Optional[str] = attrib(default=None)  # KEEP (canonical for Devmate)
```

**Replace `__attrs_post_init__` (lines 195–215):**

```python
def __attrs_post_init__(self):
    """Devmate-specific defaults: ~/fbsource as the operating target;
    inject cd-into-repo as pre-exec script (Devmate CLI requires it).

    repo_path is Devmate's historical name for what the framework calls
    target_path. We mirror repo_path → target_path BEFORE super() so the
    base's working_dir defaulting (working_dir → target_path) sees the
    correct value.
    """
    # ─── DO NOT DELETE: Devmate-specific default (~/fbsource) ──────
    if self.repo_path is None:
        self.repo_path = os.path.expanduser("~/fbsource")
    # ─── End must-preserve ─────────────────────────────────────────

    # Mirror repo_path into target_path before base post-init runs.
    # The base will then default working_dir = target_path = repo_path.
    if self.target_path is None:
        self.target_path = self.repo_path

    # ─── DO NOT DELETE: cd-into-repo pre-exec script ───────────────
    # Devmate CLI requires being invoked from inside the repo root
    # AND the env activation must happen in the same shell as the
    # main command (so cwd= alone is not sufficient — pre-script is).
    cd_script = f'cd "{self.repo_path}" || exit 1'
    if self.pre_exec_scripts is None:
        self.pre_exec_scripts = [cd_script]
    elif cd_script not in self.pre_exec_scripts:
        self.pre_exec_scripts.insert(0, cd_script)
    # ─── End must-preserve ─────────────────────────────────────────

    super().__attrs_post_init__()
```

**Why this approach is correct:**
- `attrs` generates `__init__(self, repo_path=None, ...)` — every existing test site continues to pass `repo_path=` as a kwarg without any change. **Zero test breakage on the 45+ sites.**
- `target_path` is *additionally* available as a framework-canonical name (inherited from `TerminalInferencerBase`). New code can use either name.
- The two fields stay in sync at construction time. They could drift if someone reassigns `self.repo_path` post-construction — document this as a "use either name consistently, do not mix" guideline.

**Test impact (v2 — corrected):**
- All **45+ test sites** (verified via grep — the v1 plan's "15+" estimate was 3× low) using `repo_path=` continue to work unchanged.
- No new tests required for back-compat (the existing tests *are* the back-compat tests).
- One additional test: `test_devmate_repo_path_mirrors_to_target_path` — `inst = DevmateCliInferencer(repo_path="/x"); assert inst.target_path == "/x"`.

### 8.2 `KiroCliInferencer` — drop local `target_path` declaration

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/kiro/kiro_cli_inferencer.py`

**Remove** the local `target_path` attrib (line 73):
```python
# DELETED — now inherited from TerminalInferencerBase:
# target_path: Optional[str] = attrib(default=None)
```

**Simplify `__attrs_post_init__` (lines 83–95):**

```python
def __attrs_post_init__(self) -> None:
    """Resolve model_name (BUSINESS LOGIC — must not delete);
    target_path/working_dir handled by base."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.common import (
        resolve_model_tag,
    )
    # ─── DO NOT DELETE: model normalization, not path defaulting ───
    # This converts user-supplied model names like "sonnet" to the
    # canonical Kiro model tag. It is independent of the path/workspace
    # refactor and must survive the simplification of post-init.
    if self.model_name and self.model_name != "auto":
        self.model_name = resolve_model_tag(self.model_name)
    # ─── End must-preserve ─────────────────────────────────────────
    super().__attrs_post_init__()
    # NOTE: target_path default (os.getcwd()) and
    # working_dir = target_path are now handled in
    # TerminalInferencerBase.__attrs_post_init__.
```

### 8.3 `ClaudeCodeCliInferencer` — drop local `target_path` declaration

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`

> **Audit F11 verdict (✅ SAFE):** Confirmed that `_resolve_claude_command()` (lines 125–186) has **zero** dependencies on `self.target_path` or `self.working_dir` — it only tests CLI availability and env vars. All subprocess calls go through `self._resolve_subprocess_cwd()` which reads `self.working_dir` (set by base post-init). No MCP server references. No subtle ordering issues.

**Remove** the local `target_path` attrib (line 92):
```python
# DELETED — now inherited from TerminalInferencerBase:
# target_path: Optional[str] = attrib(default=None)
```

**Simplify `__attrs_post_init__` (lines 111–123):**

```python
def __attrs_post_init__(self) -> None:
    """Default target_path to ~/fbsource (Claude's convention);
    resolve claude command path; base handles working_dir."""
    # ─── DO NOT DELETE: Claude-specific default target ─────────────
    # Claude Code's convention is to operate on the fbsource monorepo
    # when no explicit target is supplied. This overrides the base's
    # generic os.getcwd() default.
    if self.target_path is None:
        self.target_path = os.path.expanduser("~/fbsource")
    # ─── End must-preserve ─────────────────────────────────────────

    super().__attrs_post_init__()
    # NOTE: working_dir = target_path now handled in
    # TerminalInferencerBase.__attrs_post_init__.

    # ─── DO NOT DELETE: claude command path resolution ─────────────
    # Resolves the `claude` binary location (Node.js wrapper paths).
    # Independent of target_path/working_dir; safe to run after super().
    self._resolve_claude_command()
    # ─── End must-preserve ─────────────────────────────────────────
```

Note: `_resolve_claude_command()` is moved to *after* `super()` because it has no path-defaulting dependencies (F11 verified). Keeping it before super was a v1 plan oversight — it's cleaner after.

### 8.4 `RovoDevCliInferencer` — critical session-resumption preservation

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/rovodev/rovodev_cli_inferencer.py`

> **Audit F10 verdict (✅ SAFE):** Exhaustive trace of all 6 `self.working_dir` reads confirms they all expect "the directory the CLI agent operates on" (= target_path). Specifically: line 336 (`subprocess.run(cwd=...)`), lines 623/628/677/683 (`find_latest_session_id`/`ensure_session_metadata` for both read AND write — verified no read/write path mismatch). No log-file paths, no inferencer-bookkeeping paths use `working_dir`. The leaf is fully safe under the new `working_dir = target_path` default.

This leaf needs special care because the session-finding logic (lines 623, 628, 677, 683) passes `workspace_path=self.working_dir` to `find_latest_session_id()` and `ensure_session_metadata()`. After Phase 1+2, `working_dir` equals `target_path` (not `workspace.root`) — which is **the correct value** for session lookup because RovoDev's session metadata is keyed by the directory the CLI was launched in.

**Change required:** Replace `workspace_path=self.working_dir` with `workspace_path=self.target_path` in all four call sites for semantic clarity. Behavior is identical (working_dir == target_path by default), but the intent is explicit.

```python
# Lines 623, 628, 677, 683 — pattern:
# OLD:
# session_id_found = find_latest_session_id(workspace_path=self.working_dir)
# NEW:
session_id_found = find_latest_session_id(workspace_path=self.target_path)
```

**Drop local working_dir initialization (lines 144–146):**
```python
# DELETED — handled by TerminalInferencerBase.__attrs_post_init__:
# if self.working_dir is None:
#     self.working_dir = os.getcwd()
```

### 8.5 `MetamateCliInferencer` — minimal change

Audit for any local `target_path` / `working_dir` initialization; remove if present, call `super().__attrs_post_init__()`. Apply the same `workspace_path` clarification if it has session-finding logic. (Pre-merge sub-audit required to confirm Metamate has no hidden dependencies analogous to F10/F11; expected: clean.)

### 8.6 Test stub updates

Five Terminal-only stubs in `test_terminal_inferencer_base.py` (lines 18, 43, 71, 98, 122): no changes — they continue to inherit `TerminalInferencerBase` and now get `target_path` as a free new field they don't need to use.

Three Session stubs in `test_large_arg_offload.py:36`, `test_large_input_mode.py:19`, `test_terminal_session_inferencer_pipe_hang.py:50`: no API changes; verify they construct cleanly under the new MRO.

### 8.7 Phase 3 risk register

| Risk | Mitigation |
|---|---|
| `repo_path` property alias does not work with `attrs` cmp/hash. | The alias is read-through to `target_path` which IS an attrib, so `attrs.cmp`/`attrs.hash` work on the underlying field. Add `test_devmate_repo_path_alias_round_trip`. |
| `repo_path` setter called before `__attrs_post_init__` finishes (during attrs construction) raises AttributeError. | Define the property OUTSIDE the attrs field block (after the class body's attribs). Confirmed safe by attrs docs. |
| RovoDev session-resumption tests fail because they assumed `working_dir == workspace.root`. | Phase 4 adds explicit session-resumption integration tests using a temp workspace + target_path triangle. |
| Devmate's `cd "$repo_path"` pre-exec script was relying on `working_dir == repo_path` for some downstream behavior. | The pre-exec script is now defensive: it `cd`s explicitly. The subprocess `cwd=target_path` already lands the shell in the right place; the `cd` is now redundant but harmless. Document this in the migration commit message. |

---

## 9. Phase 4 — Session-resumption regression suite + final validation

### 9.1 New test file: `test/agent_foundation/common/inferencers/external/test_session_resumption_after_axes_refactor.py`

**Test cases:**

1. `test_rovodev_session_resume_with_workspace_and_target_path_distinct`
   - Construct `RovoDevCliInferencer(target_path=tmp_repo, workspace=tmp_ws)`.
   - Mock the actual CLI to write a fake session metadata file under target_path.
   - Call `inf.ainfer(...)`; assert `find_latest_session_id` is called with `workspace_path=tmp_repo` (not `tmp_ws`).
   - Verify `active_session_id` is set from the metadata.

2. `test_claude_code_session_resume_uses_target_path` — analogous for Claude Code.

3. `test_kiro_session_resume_uses_target_path` — analogous for Kiro.

4. `test_orchestrator_propagates_workspace_but_not_target_path`
   - Use `BreakdownThenAggregateInferencer(workspace=root_ws)` with a child terminal inferencer that has `target_path="/repo"`.
   - After child workspace propagation, child's `_workspace` == root_ws.child("..."), child's `target_path` == "/repo" (unchanged), child's `working_dir` == "/repo".
   - Subprocess launches with `cwd="/repo"`; artifacts written to `child._workspace.root`.

5. `test_orchestrator_child_without_explicit_target_path_defaults_to_cwd`
   - Same setup but child has no `target_path` set.
   - Expectation: child.target_path == os.getcwd() (NOT the propagated workspace root). This is a behavior change from the pre-refactor world — document loudly.

6. `test_full_triangle_independence_under_load`
   - Parametrize over (workspace_set, target_path_set, working_dir_set) ∈ {T,F}³.
   - For each combination, instantiate each of the 5 leaves and assert the resolved values match the default rule (§3.3).

### 9.2 Documentation updates

Update:
- `docs/` README for inferencers (if exists) to describe the three-field contract.
- Docstring at top of `terminal_inferencer_base.py` explaining the contract.
- Add a `MIGRATION_NOTES.md` entry under `_docs/_plans/_archive/` once Phase 4 ships.

### 9.3 Final acceptance criteria (all must pass before merge to main)

- [ ] All 6 Phase 0 tests now GREEN.
- [ ] All Phase 1 cache_folder / logger / propagation tests GREEN.
- [ ] All 3 Phase 1.5 rename / alias / deprecation-warning tests GREEN.
- [ ] All Phase 2 MRO / field-order / `_wrap_parse_output` tests GREEN.
- [ ] All 15+ existing DevMate tests pass unchanged via `repo_path` alias.
- [ ] All Phase 3 leaf-migration tests GREEN.
- [ ] All Phase 4 session-resumption tests GREEN.
- [ ] No regression in existing `test_terminal_inferencer_base.py`, `test_terminal_session_inferencer_pipe_hang.py`, `test_large_arg_offload.py`, `test_large_input_mode.py`.
- [ ] No regression in existing streaming/recovery tests (`test_streaming_inferencer_dual_timer.py`, `test_streaming_recovery.py`, `test_inferencer_resumable*.py`, `test_get_final_output.py`, `examples/.../recovery/*`).
- [ ] `test_supports_prompt_rendering`, `test_propagate_to_children`, `test_templated_inferencer_modes`, `test_mfdual_workspace_anomalies_integration` all pass (templating layer untouched but diamond MRO must not break their isinstance checks).
- [ ] `pytest -k "real"` smoke tests for at least one real CLI leaf (Kiro or Claude) execute against a temp_repo + temp_workspace pair and produce correct artifacts in workspace.root.
- [ ] Grep `^StreamingInferencerBase$` in production source returns zero hits (eager-migration complete); only the alias-emitting `__getattr__` in `streaming_inferencer_base.py` remains.
- [ ] CI log contains at most one `DeprecationWarning: StreamingInferencerBase has been renamed…` per process (validates the `once` filter).

---

## 10. Migration & rollback strategy

### 10.1 Branch strategy
- Single feature branch: `refactor/terminal-inferencer-axes-and-streaming-rename`.
- **Six** sequential commits (one per phase: 0, 1, 1.5, 2, 3, 4). Each commit is independently revertable.
- Phase 0 commit lands first via its own PR, merged immediately.
- Phase 1.5 (the rename) is also small and self-contained enough to merit its own PR — recommend landing it second, before the rest of the feature branch, so the new class name is available to the remaining phases without requiring rebase churn.
- Phases 1, 2, 3, 4 land on the feature branch and are reviewed together as one PR.

### 10.2 Back-compat surface
- **DevMate `repo_path`:** property alias survives indefinitely. Deprecation warning may be added in a future release; not in this PR.
- **Old `Leaf(workspace=ws)` (no `target_path`) behavior:** previously caused `working_dir=ws.root`. Now causes `working_dir=os.getcwd()`. **This is a behavior change.** Mitigation: add a one-shot `DeprecationWarning` in `_configure_for_workspace` if `hasattr(self, "target_path")` AND `target_path == os.getcwd()` AND `workspace.root != os.getcwd()` — flag possible accidental reliance on the old behavior.
- **Implicit `target_path = workspace.root` migrations:** any caller that genuinely wanted the CLI to run in workspace.root must now write that explicitly: `Leaf(workspace=ws, target_path=ws.root)`.

### 10.3 Rollback plan
- Phase 1 alone is rollback-safe (revert one commit, single-file change).
- Phase 1.5 alone is rollback-safe (revert restores the old class name; alias machinery vanishes). The DeprecationWarning artifacts in CI logs disappear on revert.
- Phases 2–3 are coupled (Session re-parenting + leaf cleanups must roll back together). If Phase 1.5 already shipped, the rollback of Phase 2 still keeps `TemplatedStreamingInferencerBase` as the name — Phases 2/3 do not depend on the old name being restored.
- If Phase 4 reveals a session-resumption regression, revert Phase 3 only; Phases 1, 1.5, and 2 remain in effect (they don't touch the session-finding code).
- If a critical external consumer is broken by Phase 1.5 despite the alias, revert *only* Phase 1.5 (independent commit). Phases 1, 2, 3, 4 still work because they're written to be name-agnostic for the streaming base (they reference whichever name is current at the time of authoring).

### 10.4 Sequencing across teams
- Notify owners of `BreakdownThenAggregateInferencer`, `MultiFlowDualInferencer`, and any orchestrator that propagates workspaces to terminal children: their composition tests must be re-run on the feature branch before merge.
- No changes required to `Dual`, `LinearWorkflow`, `PTI`, `Conversational*` because they delegate to children rather than executing themselves.

---

## 11. Open questions for review

1. **Should `target_path` accept `InferencerWorkspace.root` as a "source of truth" default when both `workspace` and `target_path` are unset, OR strictly default to `os.getcwd()`?** Plan currently says `os.getcwd()` (strict separation). Alternative would be: "if `workspace` is set and `target_path` is not, default `target_path=workspace.root` with a debug log". Trade-off: convenience vs. semantic purity. **Recommendation:** keep strict — explicit is better than implicit; debug log doesn't help when behavior is wrong silently in production.

2. **Should `_wrap_parse_output` be promoted to `InferencerBase` for symmetry?** Currently it lives on `TerminalInferencerBase`. Argument for promotion: other base classes might want output-wrapping hooks. Argument against: YAGNI; only Terminal/Session need it today. **Recommendation:** defer until a second use case appears.

3. **Should DevMate's `cd "$repo_path"` pre-exec script be deleted entirely now that `cwd=target_path` is enforced by `subprocess.run`?** It's redundant but harmless. **Recommendation:** delete in a follow-up PR after Phase 4 ships — keep this PR focused on the structural refactor.

4. **Do we want to add `target_path` to `RemoteInferencerBase` or `ApiInferencerBase`?** Plan says no (the field is meaningless for cloud APIs). Confirm with reviewers.

5. **Multi-inheritance on `TerminalSessionInferencerBase` — diamond risk.** Alternative: extract `_TerminalExecMixin` (no inheritance). Plan rejects this because it complicates `isinstance(x, TerminalInferencerBase)` checks. Reviewers should challenge this. **Recommendation:** prototype on Kiro first (Phase 2b prototype task), confirm `attrs` field order is stable, then proceed.

6. **Should `StreamingInferencerBase` be *split* (extract recovery into a separate mixin) instead of merely renamed?** Considered and rejected. The recovery subsystem (`_render_recovery_prompt` + the four code paths that call it: `_try_resume_from_cache`, `_atry_resume_from_cache`, `_infer_recovery`, `_ainfer_recovery`) runs on *every* `infer()`/`ainfer()` call through `_*_single`, not just on the error path. Extracting it would leave a hollowed-out base that no production subclass would want to inherit, and would force all 10 existing subclasses to re-parent to the new "with recovery" class — adding cost without adding value. The **rename** is sufficient: it makes the templating coupling visible without inventing a class no one will use. See §1 P4 for the full rationale. **Recommendation:** confirm acceptance; do not split.

---

## 12. Estimated effort

| Phase | Implementation | Testing | Review | Total (person-hours) |
|---|---|---|---|---|
| 0 | 0 | 4 | 1 | 5 |
| 1 | 1 | 3 | 2 | 6 |
| **1.5** | **3** | **2** | **2** | **7** |
| 2a | 2 | 3 | 2 | 7 |
| 2b | 4 | 6 | 4 | 14 |
| 3 | 3 | 4 | 3 | 10 |
| 4 | 1 | 8 | 3 | 12 |
| **Total** | **14** | **30** | **17** | **61** |

Roughly **1.7 engineer-weeks** assuming one engineer, with ~50% in test authoring and ~28% in review cycles. The rename (Phase 1.5) adds ~7 hours but reduces ongoing onboarding cost indefinitely (every new contributor that would have asked "why does StreamingInferencerBase inherit from TemplatedInferencerBase?" now gets the answer from the class name itself).

---

## 13. Appendix — verified line numbers (for plan-implementation cross-check)

### `inferencer_base.py` (CoreProjects)
- `has_local_access` attrib: line **114**
- `workspace` attrib: line **161**
- `_workspace` property getter/setter: lines **224–236**
- `_propagate_workspace_to_children`: lines **301–360**
- `_configure_for_workspace`: lines **366–398** (clobber to remove: lines **374–384**)
- `__attrs_post_init__`: lines **424–518** (workspace handling: **436–443**)
- `resolve_output_path`: lines **1366–1391**

### `terminal_inferencer_base.py` (CoreProjects)
- Class declaration: line **17** (`TerminalInferencerBase(TemplatedInferencerBase)`)
- Attribs: lines **42–52**
- `__attrs_post_init__`: lines **54–58**
- `_resolve_subprocess_cwd`: lines **60–80**
- `_execute_command`: lines **229–306**
- `_infer`: lines **474–555**

### `streaming_inferencer_base.py` (CoreProjects) — touched by Phase 1.5
- Class declaration: line **103** (`StreamingInferencerBase(TemplatedInferencerBase)` → rename to `TemplatedStreamingInferencerBase`)
- `__attrs_post_init__`: lines **222–231** (calls `super()` then conditionally registers recovery template root)
- Templating touch-points: lines **162, 165, 225, 230, 246, 250** (all stay; class is renamed around them)
- Recovery prompt call sites: lines **347, 372, 974, 1023** (within `_try_resume_from_cache`, `_atry_resume_from_cache`, `_ainfer_recovery`, `_infer_recovery` respectively)

### `terminal_session_inferencer_base.py` (CoreProjects)
- Class declaration: line **56** (`TerminalSessionInferencerBase(StreamingInferencerBase)` → re-parent to `(TerminalInferencerBase, TemplatedStreamingInferencerBase)`)
- Attribs (to deduplicate): lines **71–82**
- `_resolve_subprocess_cwd` (duplicate to delete): lines **136–156**
- `_ainfer_streaming` (preserve): lines **377–423** and **559–593**
- `_infer` (replace with inherited): lines **474–498**
- `_ainfer` (preserve): lines **444–472**

### `inferencer_workspace.py` (CoreProjects)
- Class declaration: line **34**
- `DEFAULT_OUTPUT_FILENAME` constant: line **29**
- Public properties: `outputs_dir`, `artifacts_dir`, `checkpoints_dir`, `logs_dir`, `children_dir`, `deliverables_dir`, `has_deliverables`, `deliverable_paths`
- Public methods: `deliverable_path`, `surface_outputs_from`, `ensure_dirs`, `output_path`, `artifact_path`, `checkpoint_path`, `log_path`, `analysis_path`, `results_path`, `subdir`, `child`, `child_output`, `glob_outputs`, `glob_artifacts`, `write_marker`, `has_marker`, `clear_marker`

### `templated_inferencer_base.py` (CoreProjects)
- Class declaration: line **72** (`@attrs(slots=False)` — important: NOT slots-based)
- Attribs: lines **90–114**
- **No `__attrs_post_init__` override** — inherits InferencerBase's. (Verified by grep.)

### `breakdown_then_aggregate_inferencer.py` (CoreProjects) — touched by Phase 1
- `_configure_for_workspace` override: lines **1135–1139**. Calls `super()._configure_for_workspace(workspace)` then propagates to `breakdown_inferencer`. **Plan does NOT modify this** — the v2 conditional clobber in `inferencer_base.py` preserves BTA's expected behavior (breakdown_inferencer is constructed with target_path=None, so the conditional fires and working_dir tracks the propagated workspace as before).

### Verified slot compatibility across the diamond (audit C7)
All five classes in the `TerminalSessionInferencerBase` MRO use the **legacy** `from attr import attrs, attrib` API which defaults to `slots=False`:
- `inferencer_base.py:29` — `@attrs` (bare → slots=False under legacy API)
- `templated_inferencer_base.py:72` — `@attrs(slots=False)` (explicit)
- `streaming_inferencer_base.py:103` — `@attrs` (bare → slots=False)
- `terminal_inferencer_base.py:17` — `@attrs` (bare → slots=False)
- `terminal_session_inferencer_base.py:56` — `@attrs` (bare → slots=False)

**No slot mismatch risk.** A permanent regression test (§7.2 Phase 2b — `test_diamond_attrs_slots_consistency`) guards against future contributors accidentally switching one of these to `@attrs.define` (modern API, defaults to slots=True), which would break the diamond.

### DevMate `repo_path` test-site count correction (audit C7)
- v1 plan estimated: 15+ test sites passing `repo_path=` as kwarg.
- v2 verified via grep: **45+ test sites** (3× higher than estimated). Includes `test_dual_inferencer/__main__.py:131`, `test_devmate_cli_inferencer_real.py:54, 108, 156, 207`, and ~40 others.
- v2 design (§8.1) keeps `repo_path` as an `attrib` precisely so none of these 45+ sites need to change.

### Correction to a claim in an earlier plan revision
An earlier revision of this plan asserted "`StreamingInferencerBase` does NOT define its own `__attrs_post_init__`". **This was incorrect.** Verified by inspection: `streaming_inferencer_base.py:222–231` defines `__attrs_post_init__` which calls `super().__attrs_post_init__()` then conditionally registers a recovery template root via `self.template_manager.add_template_root(...)`. The diamond MRO analysis in §7.3 has been updated to reflect this. The chain is **safe** (both Terminal's and the streaming base's post-init bodies call `super()` exactly once, so each layer runs exactly once), but the MRO assertion test added in Phase 2b must explicitly verify *both* post-init layers ran — not just Terminal's.

### Audit follow-ups F10 / F11 — RESOLVED
The v1 plan flagged two TODO sub-audits before Phase 3:
- **F10** (RovoDev `working_dir` exhaustive trace): ✅ SAFE. All 6 reads (line 336 subprocess cwd; lines 623/628/677/683 session metadata) correctly expect "the directory the CLI agent operates on" = `target_path`. No log-file or bookkeeping paths use `working_dir`. No read/write path mismatch.
- **F11** (ClaudeCodeCli `_resolve_claude_command`): ✅ SAFE. The method has zero references to `target_path` / `working_dir` / `_workspace`. It only tests CLI availability and Node.js paths. Safe to call before or after `super().__attrs_post_init__()`.

Both sub-audits are now incorporated as §8.3 / §8.4 inline notes.

### Production leaves (CoreProjects)
- `ClaudeCodeCliInferencer`: `claude_code_cli_inferencer.py:31`
- `KiroCliInferencer`: `kiro_cli_inferencer.py:25`
- `DevmateCliInferencer`: `devmate_cli_inferencer.py:63` (`repo_path` at line **163**, `__attrs_post_init__` at **195–215**)
- `MetamateCliInferencer`: `metamate_cli_inferencer.py:33`
- `RovoDevCliInferencer`: `rovodev_cli_inferencer.py:75` (session-finding at lines **624, 629, 678, 684**)

### Test stubs (CoreProjects)
- `test_terminal_inferencer_base.py`: 5 Terminal stubs at lines **18, 43, 71, 98, 122**
- `test_large_arg_offload.py:36`
- `test_large_input_mode.py:19`
- `test_terminal_session_inferencer_pipe_hang.py:50`

---

## 14. Why this plan is "no ad-hoc, no hacky" — design principles applied

1. **Single Responsibility per field.** Each of `workspace`, `target_path`, `working_dir` has exactly one writer and one semantic role.
2. **Locality of behavior.** Default-resolution logic lives in exactly one place (`TerminalInferencerBase.__attrs_post_init__`), not scattered across leaves.
3. **Explicit > implicit.** `workspace.root → target_path` auto-binding is rejected in favor of explicit user opt-in.
4. **Backwards compatibility via property/module aliases**, not via "if/elif compatibility shims" inside `__attrs_post_init__`. DevMate's `repo_path` is a property alias on `target_path`; `StreamingInferencerBase` is a module-level `__getattr__` alias on `TemplatedStreamingInferencerBase`. Both preserve existing imports/instantiations exactly.
5. **Diamond inheritance accepted only when both parents trace to the same root** (`TemplatedInferencerBase`), making MRO deterministic by C3 linearization. The Phase 1.5 rename strengthens this — both diamond parents are now named honestly so readers immediately see *why* the MRO converges where it does.
6. **Test-first via Phase 0** — failing tests precede source changes.
7. **Phased rollout with independent revertability** — Phase 1 alone is shippable; Phase 1.5 alone is shippable; subsequent phases are gated on their success.
8. **No magic.** Every behavior is discoverable from the field's docstring, the class docstring, and the `__attrs_post_init__` body. No metaclass tricks, no `setattr` indirection, no decorator magic. The only "clever" construct is module-level `__getattr__` for the deprecation alias — a documented Python 3.7+ idiom (PEP 562).
9. **Cascade-injection semantics preserved** — by NOT putting `target_path` on `InferencerBase`, we don't accidentally make every cloud/orchestrator inferencer a recipient of `target_path` propagation.
10. **Symmetric sync/async treatment** — both `_infer` and `_ainfer` paths are considered in the contract; neither is left weaker than the other.
11. **Names tell the truth.** `TemplatedStreamingInferencerBase` is named for what it actually does (renders templated recovery prompts as part of streaming) rather than for the abstract concept it superficially resembles ("just streaming"). The rename addresses §1 P4 not by changing behavior but by changing the *advertisement* of behavior to match reality. `TerminalInferencerBase` is *not* renamed because its current name is already honest (it doesn't call `self.template_manager(...)` from its own body).
12. **One bad idea explicitly rejected per axis.** §11 documents (with reasoning) what we considered and chose not to do: don't put `target_path` on `InferencerBase` (Q4); don't extract `_TerminalExecMixin` (Q5); don't split `StreamingInferencerBase` into pure-stream + recovery-mixin (Q6). Reviewers can revisit any of these without having to reconstruct the analysis.

---

*End of plan. Reviewers: please challenge §3.3 (default rule), §6.4 (Phase 1 risk register), §6A.5 (rename risk register), §7.3 (diamond MRO under the renamed parent), and §11 (open questions Q1–Q6) most carefully.*

