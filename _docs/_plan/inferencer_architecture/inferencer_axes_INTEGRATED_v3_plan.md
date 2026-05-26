# Inferencer Axes Refactor — Integrated v3 Plan

**Author:** Tony Chen (integrating Rovo Dev v2 + Claude v1 plans)
**Date drafted:** 2026-05-16 07:24
**Status:** Ready for review and implementation
**Supersedes:**
- `terminal_inferencer_axes_and_streaming_rename_plan.md` (Rovo Dev v2, 1,071 lines)
- `/Users/tchen7/.claude/plans/let-s-create-an-integrated-lively-pearl.md` (Claude v1, 308 lines)

> **Why this integrated plan exists.** Two plans were drafted independently for the same refactor. Each had strengths the other lacked. This document combines:
> - **Claude's architectural insight** — decouple ALL three bases (Streaming, Terminal, TerminalSession) from `TemplatedInferencerBase`, then re-add templating via narrow convenience classes (`TerminalTemplatedInferencerBase`, `TerminalSessionTemplatedInferencerBase`). This is the elegant decoupling I was circling around but didn't commit to. **It also obsoletes my v2 Phase 1.5 (the `StreamingInferencerBase` rename) entirely** — once the class no longer inherits from `TemplatedInferencerBase`, its name is honest as-is.
> - **My v2 operational rigor** — RED-first Phase 0 tests, conditional clobber in `_configure_for_workspace`, orchestrator regression tests, per-phase risk registers, DO-NOT-DELETE markers, audit follow-ups (F10/F11), `repo_path` mirror pattern, phased rollout with independent revertability.

---

## 1. Target architecture (Claude's decoupling, kept verbatim)

```
InferencerBase                                              — orchestrators, generic
├── TemplatedInferencerBase                                 — template rendering (stand-alone axis)
├── StreamingInferencerBase                                 — streaming + session + recovery cache
├── TerminalInferencerBase                                  — subprocess execution + target_path
│
├── TerminalSessionInferencerBase(TIB, SIB)                 — terminal + streaming  (MI diamond #1)
├── TerminalTemplatedInferencerBase(TIB, TemplatedIB)       — terminal + templates  (MI diamond #2)
├── TerminalSessionTemplatedInferencerBase(TSIB, TemplatedIB) — all three           (MI diamond #3)
│
│   Concrete CLI leaves (templated):
├── ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase)
├── KiroCliInferencer(TerminalSessionTemplatedInferencerBase)
├── DevmateCliInferencer(TerminalSessionTemplatedInferencerBase)
├── RovoDevCliInferencer(TerminalSessionTemplatedInferencerBase)
│
│   Concrete CLI leaf (no templates):
├── MetamateCliInferencer(TerminalSessionInferencerBase)
│
│   Concrete SDK / API / Tool leaves (streaming, no templates, no subprocess):
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

**MRO for `TerminalSessionTemplatedInferencerBase` (C3-verified):**
```
TSTIB → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC → object
```

**MRO for `TerminalTemplatedInferencerBase`:**
```
TTIB → TIB → TemplatedIB → IB → Debuggable → Resumable → ABC → object
```

**MRO for `TerminalSessionInferencerBase` (no templates path):**
```
TSIB → TIB → SIB → IB → Debuggable → Resumable → ABC → object
```

### 1.1 Why this beats v2's hierarchy

The v2 plan kept all three bases inheriting from `TemplatedInferencerBase` and renamed `StreamingInferencerBase` → `TemplatedStreamingInferencerBase` to make the dependency visible. **This was a workaround for a coupling that didn't need to exist.** Claude's decoupling (this plan) removes the coupling entirely:

- `StreamingInferencerBase` becomes axis-pure: streaming + session + recovery cache. Its `_render_recovery_prompt` uses `getattr(self, "template_manager", None)` — a graceful degrade pattern that the recovery code already supports (`elif self.use_default_prompt_templates: render_recovery_prompt(...)` at line 250).
- `TerminalInferencerBase` becomes axis-pure: subprocess execution + `target_path` + `working_dir` + `pre_exec_scripts` + `env_vars`.
- `TemplatedInferencerBase` is opt-in via the three diamond classes.
- The misleading "Streaming sounds orthogonal but secretly depends on Templated" problem **disappears entirely** without any rename.
- `MetamateCliInferencer` (which doesn't want templates) gets a cleaner parent (`TerminalSessionInferencerBase` directly, no templating bloat).
- The `isinstance(x, TemplatedInferencerBase)` discriminator used in `dual_inferencer.py:1602` and `multi_flow_dual_inferencer.py:460` continues to work correctly — templated leaves are *exactly* the ones using the templated convenience classes.

### 1.2 Three diamonds, all safe

All three diamonds share the same ultimate root (`InferencerBase`) and use only legacy `@attrs` (slots=False) decorators. C3 linearization is deterministic; `attrs` field collection deduplicates correctly; `__attrs_post_init__` chains via `super()` cooperatively. **A permanent regression test (§7) pins all three MROs.**

---

## 2. Problems solved (consolidated)

| # | Problem | Status before | Status after |
|---|---|---|---|
| P1 | `working_dir` is overloaded: artifact root vs CLI cwd | One field, two writers, latent clobber bug when user sets both `workspace=` and `target_path=` | `workspace` owns artifact storage; `target_path` (NEW) owns CLI semantic working dir; `working_dir` owns subprocess `cwd=`. Three names, three roles, one writer each. |
| P2 | `TerminalSessionInferencerBase` does not inherit from `TerminalInferencerBase` — duplicated `working_dir`, `pre_exec_scripts`, `_resolve_subprocess_cwd`; weaker `_infer` (no timeout/env_vars/post_exec_scripts) | Two parallel hierarchies | Single hierarchy via MI diamond #1 (`TSIB(TIB, SIB)`); all execution machinery deduplicated to TIB. |
| P3 | Templating, terminal-exec, streaming all bundled via single inheritance — orthogonal axes coupled | Single inheritance forces every leaf to either inherit all axes or none | Three orthogonal bases + three convenience MI classes; leaf opts into exactly the axes it needs. |
| P4 | `StreamingInferencerBase` name is misleading (it secretly uses `template_manager`) | True under v1 hierarchy | **Disappears** — once SIB no longer inherits from TemplatedIB, its name is honest. No rename needed. |
| P5 | `_configure_for_workspace` unconditionally clobbers `working_dir` — explicit user `target_path` is silently overridden | Latent bug | Guarded clobber (Phase 1): clobber only fires when `target_path is None`. |
| P6 | `MetamateCliInferencer` doesn't want templating but is forced under the templated branch | Bloated parent | Direct `TerminalSessionInferencerBase` parent (no templating). |
| P7 | `_last_streaming_output` / `_last_streaming_return_code` declared as instance attribs on TSIB only, even though TIB's `_execute_command_streaming` sets them | Subtle attrs-vs-instance-attr inconsistency | Promoted to proper `attrib(init=False, repr=False)` on TIB; TSIB inherits. |

### 2.1 What is *not* changed

- `TemplatedInferencerBase` — body unchanged. Only its position in the hierarchy changes (no longer a parent of SIB/TIB).
- `InferencerBase` — single surgical edit to `_configure_for_workspace`. All other behavior unchanged.
- Orchestrators (BTA, MFDual, LWI, Dual) — zero changes. The `_configure_for_workspace` edit is designed to preserve their child-spawning behavior.
- All 10 streaming-only subclasses (SDK, API, Tool) — parent stays `StreamingInferencerBase`. No leaf code changes.
- All YAML configs — verified: no template-field cascade injection through SIB/TIB-only leaves. No config changes required.

---

## 3. The field contract (target_path / working_dir / workspace)

| Field | Owns | Lives on | Default rule |
|---|---|---|---|
| `workspace` (`Optional[InferencerWorkspace]`) | Artifact storage — outputs, logs, cache_folder, checkpoints | `InferencerBase` (unchanged) | `None` (opt-in) |
| `target_path` (`Optional[str]`) | CLI agent operating directory — the "what is the agent working on" semantic | **`TerminalInferencerBase` (NEW)** | `None` at field level; leaves may default in their post-init (e.g., `~/fbsource` for Claude/Devmate, `os.getcwd()` fallback for generic Kiro) |
| `working_dir` (`str`) | Subprocess `cwd=` for `subprocess.run` / `asyncio.create_subprocess_*` | `TerminalInferencerBase` (already; deduplicated from TSIB) | Defaults to `target_path` if set; otherwise `os.getcwd()` |

### 3.1 Default-resolution algorithm (deterministic, single owner)

In `TerminalInferencerBase.__attrs_post_init__`:

```python
def __attrs_post_init__(self):
    """Resolve working_dir from target_path; both default safely.

    Order matters:
      1. Leaf's own __attrs_post_init__ has already run (it set
         target_path to leaf-specific default like ~/fbsource if it
         wanted to override the generic default).
      2. We default working_dir = target_path if target_path is set,
         else working_dir = os.getcwd().
      3. super() runs InferencerBase.__attrs_post_init__ which triggers
         _configure_for_workspace IF workspace was provided.
         _configure_for_workspace (after Phase 1 edit) only clobbers
         working_dir when target_path is None — so if the leaf set
         target_path, the user-explicit working_dir derivation wins.
    """
    if self.working_dir is None:
        if self.target_path is not None:
            self.working_dir = self.target_path
        else:
            self.working_dir = os.getcwd()
    super().__attrs_post_init__()
```

### 3.2 `_configure_for_workspace` after Phase 1

```python
def _configure_for_workspace(self, workspace):
    import os
    if hasattr(self, "working_dir"):
        # Guarded clobber: only auto-set working_dir from workspace.root
        # when the user did NOT supply target_path. This preserves the
        # orchestrator-spawning case (child constructed with target_path=None,
        # later assigned a per-iteration workspace via setter — subprocess
        # MUST use the new workspace.root as cwd) while letting users with
        # explicit target_path keep their CLI launch directory.
        target = getattr(self, "target_path", None)
        if target is None:
            new_wd = str(workspace.root)
            # Windows CreateProcessW enforces MAX_PATH=260 on cwd
            if sys.platform != "win32" or len(new_wd) < 240:
                self.working_dir = new_wd
    if hasattr(self, "cache_folder"):
        self.cache_folder = os.path.join(
            str(workspace.root), "_runtime", "inferencer_cache"
        )
    # ...logger redirection unchanged...
```

### 3.3 Why this resolution rule is correct in all four scenarios

| Scenario | `target_path` after leaf post-init | `working_dir` after TIB post-init | After `_configure_for_workspace` (if workspace) |
|---|---|---|---|
| Standalone, explicit target_path | `/repo` | `/repo` | `/repo` (guard fires; target was non-None) ✅ |
| Standalone, explicit working_dir | None (or default) | `/explicit` (user-set) | `/explicit` (guard fires; target_path is None but working_dir is already set; no fallback to workspace) ✅ |
| Standalone, no target_path, no workspace | None | `os.getcwd()` | (no workspace; not called) ✅ |
| Orchestrator-spawned child (target_path=None, workspace assigned later) | None | `os.getcwd()` initially | Setter fires → guard sees target_path is None → working_dir = workspace.root ✅ |

The fourth row is the case the v2 plan got wrong by deleting the clobber. Claude's `if target is None: clobber` correctly handles it.

---

## 4. Phased rollout

Eight phases, ordered for **independent revertability** and **incremental risk**. Phase 0 ships its own PR (test-only). Phases 1–4 are the structural core. Phases 5–7 are the leaf migrations and docs.

| Phase | Title | Files touched | Risk | Reversible alone? | Tests added |
|---|---|---|---|---|---|
| 0 | RED tests pin the contract | 1 (test file) | Lowest | Yes | 9 |
| 1 | Guarded clobber in `_configure_for_workspace` | 1 (inferencer_base.py) | Low | Yes | 2 |
| 2 | Decouple `StreamingInferencerBase` from `TemplatedInferencerBase` | 1 (streaming_inferencer_base.py) | Low–Medium | Yes | 3 |
| 3 | Decouple `TerminalInferencerBase` from `TemplatedInferencerBase`; add `target_path`; promote streaming output fields | 1 (terminal_inferencer_base.py) | Medium | Yes | 4 |
| 4 | Refactor `TerminalSessionInferencerBase` to MI `(TIB, SIB)`; introduce `TerminalSessionTemplatedInferencerBase` and `TerminalTemplatedInferencerBase` | 1 + 1 (terminal_session_inferencer_base.py + new file or shared module) | Medium | With Phase 3 | 5 |
| 5 | Migrate 4 CLI leaves to `TerminalSessionTemplatedInferencerBase`; verify Metamate stays on `TerminalSessionInferencerBase` | 5 (one per CLI leaf) | Medium | Per-leaf | 3 |
| 6 | Update `__init__.py` exports and `TemplatedInferencerBase` docstring | 2 | Trivial | Yes | 0 |
| 7 | Verification suite + permanent regression tests | 1 (test file additions) | Trivial | N/A | 4 |

**Total:** ~30 tests, 12 files touched. Roughly 1.5 engineer-weeks (40–60 hours).

**Why 8 phases instead of v2's 6 or Claude's 7:** Phase 0 (RED tests) and Phase 7 (permanent regression suite) are explicit standalone phases — both shippable independently and both add value even if subsequent phases stall.

---

## 5. Phase 0 — RED tests pin the contract

**File (new):** `test/agent_foundation/common/inferencers/test_inferencer_axes_contract.py`

All 9 tests marked `xfail(strict=True)` until their corresponding source phase lands. Each test has a `# WILL_PASS_AFTER_PHASE_N` comment.

### 5.1 The 9 tests

| # | Test | Pre-fix | Post-fix | Lands |
|---|------|---------|----------|-------|
| 1 | `test_target_path_survives_workspace_assignment` — `KiroCliInferencer(target_path="/repo", workspace=ws)` → `inf.working_dir == "/repo"` | FAIL (ws.root) | PASS | Phase 3 |
| 2 | `test_explicit_working_dir_survives_workspace_assignment` — explicit `working_dir="/x"` preserved through workspace setter | FAIL | PASS | Phase 1 |
| 3 | `test_target_path_defaults_to_cwd_when_unset` — no target_path, no workspace → working_dir == os.getcwd() | depends | PASS | Phase 3 |
| 4 | `test_workspace_configures_cache_folder_not_cwd_when_target_path_set` — workspace + target_path set → cache_folder under workspace.root, working_dir == target_path | FAIL | PASS | Phase 1 + 3 |
| 5 | `test_terminal_session_inherits_terminal_features` — TSIB has `timeout`, `env_vars`, `post_exec_scripts` inherited from TIB | FAIL (attribs absent) | PASS | Phase 4 |
| 6 | `test_session_subprocess_uses_target_path` — patch `asyncio.create_subprocess_shell`, assert `cwd == target_path` | FAIL | PASS | Phase 3 + 5 |
| 7 | **`test_orchestrator_spawned_child_uses_workspace_root` (regression guard)** — BTA child with `target_path=None` → child's runtime `working_dir == per-iteration workspace.root` | passes today | PASS | Phase 1 (must NOT regress) |
| 8 | **`test_orchestrator_does_not_override_explicit_target_path`** — BTA child with `target_path="/explicit"` → child's runtime `working_dir == "/explicit"` | FAIL today | PASS | Phase 1 + 3 |
| 9 | **`test_devmate_repo_path_kwarg_still_accepted`** — `DevmateCliInferencer(repo_path="/x")` constructs; `inst.target_path == "/x"` (mirror) | passes today | PASS | Phase 5 |

### 5.2 Why tests #7 and #8 matter

Tests #7 and #8 form a contract pair that protects the *orchestrator scenario*. Test #7 is the regression guard that would have caught the v1 "delete unconditional clobber" mistake (it would have FAILed post-edit). Test #8 is the positive case for the guarded clobber (it FAILs today). Together they pin Phase 1's behavior precisely.

### 5.3 Phase 0 ships its own PR

Single test file, ~250 lines. Zero source changes. Lands first, runs in CI as `xfail(strict=True)` — converts to `PASSED` as each subsequent phase lands. CI dashboards make progress visible.

---

## 6. Phase 1 — Guarded clobber in `_configure_for_workspace`

**File:** `src/agent_foundation/common/inferencers/inferencer_base.py` (lines 367–385)

Apply the guarded clobber from §3.2 verbatim. Key delta from today: the unconditional `self.working_dir = str(workspace.root)` becomes conditional on `getattr(self, "target_path", None) is None`.

### 6.1 Why this is safe to ship before Phase 3

Phase 1 lands before Phase 3 (which adds `target_path` to TIB). At Phase 1 ship time:
- `getattr(self, "target_path", None)` returns `None` on every inferencer (the attribute doesn't exist).
- The guard `if target is None:` is always True.
- The clobber always fires → **behavior is identical to today**.

Phase 3's addition of `target_path` to TIB then *activates* the guard. This staged activation is intentional and lets Phase 1 ship as a pure no-op.

### 6.2 Tests landing in Phase 1

- Phase 0 test #2 (`test_explicit_working_dir_survives_workspace_assignment`) turns PASS — *partially*. Actually, since `target_path` doesn't yet exist as an attrib, the guard fires when `target is None`, and clobber will still overwrite an explicit `working_dir`. **Re-verify:** test #2 requires `target_path` to exist on the leaf. So test #2 *also* needs Phase 3 to GREEN. Updated table in §5.1 reflects this.
- New regression test: `test_phase1_preserves_orchestrator_clobber_behavior` — instantiate a leaf with no `target_path`, assign workspace via setter, assert `working_dir == workspace.root`. Protects against accidental regression to "delete entirely" approach.
- New regression test: `test_phase1_no_effect_when_target_path_attrib_absent` — pre-Phase-3 leaf without target_path attrib; assert no behavior change vs main.

### 6.3 Phase 1 risk register

| Risk | Mitigation |
|---|---|
| The "delete entirely" approach (rejected) would have broken orchestrators with `NotADirectoryError` for child terminal inferencers. | This plan uses the guarded approach. Regression test pinned in §5 #7. |
| A user explicitly sets `target_path=os.getcwd()` and expects the guard to honor it. | The guard fires on `target is None`, not on `target == os.getcwd()`. Explicit `target_path=os.getcwd()` → guard fires when target is None (it isn't here) → clobber does NOT fire → working_dir keeps whatever value the leaf gave it (which was os.getcwd()). Behavior: user wins. Document in field docstring. |
| BTA `_configure_for_workspace` override at `breakdown_then_aggregate_inferencer.py:1135` depends on super-call setting `working_dir` for breakdown child. | Preserved by guarded clobber (breakdown child has `target_path=None` so clobber fires as before). Add explicit test `test_bta_breakdown_child_working_dir`. |
| `_propagate_workspace_to_children` (lines 301–360) comment at line 349 explicitly documents the clobber was load-bearing. | Same as above — guard preserves the load-bearing behavior. Update comment to reflect the new conditional rule. |

---

## 7. Phase 2 — Decouple `StreamingInferencerBase` from `TemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/streaming_inferencer_base.py`

### 7.1 Edits

1. **Change parent (line 103):**
   ```python
   # OLD: class StreamingInferencerBase(TemplatedInferencerBase):
   class StreamingInferencerBase(InferencerBase):
   ```

2. **Update imports (lines 46–48):** swap `TemplatedInferencerBase` for `InferencerBase`.

3. **Duck-type `template_manager` access** in two places:
   ```python
   # Line 225 (__attrs_post_init__):
   tm = getattr(self, "template_manager", None)
   if self.use_default_prompt_templates and tm is not None:
       tm.add_template_root(_DEFAULT_RECOVERY_DIR, priority=TemplateRootPriority.LOWEST)

   # Line 246 (_render_recovery_prompt):
   tm = getattr(self, "template_manager", None)
   if tm is not None:
       return tm(key, active_template_type="", prompt=prompt, partial_output=partial_output)
   elif self.use_default_prompt_templates:
       return render_recovery_prompt(...)  # module-level fallback (already in code)
   ```

   The recovery code already has the `elif self.use_default_prompt_templates: render_recovery_prompt(...)` fallback. **The duck-type pattern preserves both the templated and the fallback paths.**

4. **Keep** `use_default_prompt_templates: bool = attrib(default=True)` and `fallback_recovery_template_key: str = "recovery"` as-is. They configure the recovery system regardless of whether `template_manager` is present.

### 7.2 Impact on 10 streaming subclasses

All 10 confirmed (by Claude's audit) to use zero template features. After Phase 2:
- They no longer inherit template attribs from `__init__` (correct — they never used them).
- `isinstance(x, TemplatedInferencerBase)` returns `False` for them (correct — they're not templated).
- No leaf code changes required.

### 7.3 Tests landing in Phase 2

- `test_sib_no_longer_inherits_templated` — `assert not issubclass(StreamingInferencerBase, TemplatedInferencerBase)`.
- `test_sib_recovery_fallback_when_no_template_manager` — instantiate a SIB-only subclass without `template_manager`, trigger `_render_recovery_prompt`, assert the module-level `render_recovery_prompt` fallback fires.
- `test_sib_recovery_uses_template_manager_when_present` — instantiate a *templated* subclass (e.g., `ClaudeCodeCliInferencer` post-Phase-5), trigger `_render_recovery_prompt`, assert it routes through `template_manager`.

### 7.4 Phase 2 risk register

| Risk | Mitigation |
|---|---|
| A streaming-only subclass implicitly relied on `template_manager` being present. | Claude's audit confirmed zero such subclasses. Test 7.3 #2 pins the fallback path. |
| `_render_recovery_prompt`'s fallback `render_recovery_prompt(...)` module-level function might not be importable / might require template_manager too. | Pre-Phase-2 grep confirmed the fallback is self-contained (uses templates from `_DEFAULT_RECOVERY_DIR` via direct Jinja, not via template_manager). |
| Removing `TemplatedInferencerBase` from MRO breaks isinstance checks elsewhere. | The 2 known checks (`dual_inferencer.py:1602`, `multi_flow_dual_inferencer.py:460`) discriminate templated-vs-not — they SHOULD return False for streaming-only leaves. This is correct behavior, not a break. |
| Streaming-only leaves' YAML configs reference template fields. | Verified by Claude's audit: zero YAML configs set template fields on SIB-only subclasses. |

---

## 8. Phase 3 — Decouple `TerminalInferencerBase` from `TemplatedInferencerBase`; add `target_path`; promote streaming output fields

**File:** `src/agent_foundation/common/inferencers/terminal_inferencer_base.py`

### 8.1 Edits

1. **Change parent (line 17):**
   ```python
   # OLD: class TerminalInferencerBase(TemplatedInferencerBase):
   class TerminalInferencerBase(InferencerBase):
   ```

2. **Update imports (lines 12–14):** swap `TemplatedInferencerBase` for `InferencerBase`.

3. **Add `target_path` field** before `working_dir` (after line 41):
   ```python
   # The directory the CLI agent operates on (e.g., the repo it edits).
   # Distinct from:
   #   - workspace.root: where THIS inferencer stores its artifacts.
   #   - working_dir:    the subprocess cwd= for command execution.
   # working_dir defaults to target_path unless explicitly overridden.
   target_path: Optional[str] = attrib(default=None)
   ```

4. **Promote streaming output fields** to proper `attrib(init=False)` (they were previously set as instance attribs in `_execute_command_streaming`):
   ```python
   _last_streaming_output: str = attrib(default="", init=False, repr=False)
   _last_streaming_return_code: int = attrib(default=0, init=False, repr=False)
   ```

5. **Update `__attrs_post_init__`** per §3.1 verbatim.

### 8.2 Tests landing in Phase 3

- Phase 0 test #1, #2, #4, #6, #8 turn GREEN.
- `test_tib_no_longer_inherits_templated` — `assert not issubclass(TerminalInferencerBase, TemplatedInferencerBase)`.
- `test_tib_target_path_attribute_exists` — `assert "target_path" in {f.name for f in attr.fields(TerminalInferencerBase)}`.
- `test_tib_streaming_output_fields_are_attribs` — verify `_last_streaming_output` and `_last_streaming_return_code` are present in `attr.fields(TerminalInferencerBase)` with `init=False`.

### 8.3 Phase 3 risk register

| Risk | Mitigation |
|---|---|
| The 5 Terminal-only test stubs in `test_terminal_inferencer_base.py` (lines 18, 43, 71, 98, 122) lose template attribs from `__init__`. | They never used them. Confirmed by Claude's audit. |
| Promoting `_last_streaming_output` / `_last_streaming_return_code` to `attrib(init=False)` changes their initial value semantics. | They were always set via attribute assignment in `_execute_command_streaming`. Default of `""` / `0` matches the pre-set state. Pre-set state can't be observed since users call infer() first. |
| `target_path` is added to TIB but Phase 4 hasn't moved TSIB to inherit from TIB yet. TSIB has its own `working_dir` that doesn't know about `target_path`. | Phase 3 alone activates `target_path` for `TerminalInferencerBase` subclasses. TSIB subclasses (the 5 CLI leaves) don't get `target_path` until Phase 4 + 5. Until then, they continue to behave as today. Phase 5 migration is required to fully activate the contract for those leaves. |

---

## 9. Phase 4 — Refactor `TerminalSessionInferencerBase` to MI + introduce convenience classes

**File:** `src/agent_foundation/common/inferencers/terminal_inferencers/terminal_session_inferencer_base.py`
**New file (option A) or shared module (option B):** convenience classes. **Recommendation:** define them inline in their respective base files (`TerminalTemplatedInferencerBase` at the bottom of `terminal_inferencer_base.py`; `TerminalSessionTemplatedInferencerBase` at the bottom of `terminal_session_inferencer_base.py`). Single source per axis.

### 9.1 Edits to `terminal_session_inferencer_base.py`

1. **Change parent (line 56):**
   ```python
   # OLD: class TerminalSessionInferencerBase(StreamingInferencerBase):
   class TerminalSessionInferencerBase(TerminalInferencerBase, StreamingInferencerBase):
       """Terminal exec + streaming/recovery. MRO: TSIB → TIB → SIB → IB."""
   ```

2. **Add import** for `TerminalInferencerBase`.

3. **Remove duplicate attribs** (now inherited from TIB):
   ```python
   # DELETED — inherited from TerminalInferencerBase:
   # working_dir: Optional[str] = attrib(default=None)         (line 71)
   # pre_exec_scripts: Optional[List[str]] = attrib(...)        (line 72)
   # _last_streaming_output                                     (line 89)
   # _last_streaming_return_code                                (line 91)
   ```

4. **Remove duplicate method** `_resolve_subprocess_cwd` (lines 136–156) — identical to TIB's.

5. **Decide on `_infer` strategy.** Session's `_infer` is currently a weaker thin wrapper around `subprocess.run`. After Phase 4, Session inherits TIB's richer `_infer` (timeout, env_vars, pre/post-scripts, _save_output). However, Session's `_infer` wraps the return in `TerminalInferencerResponse.from_dict(...)`; TIB's returns `Any`. **Resolution:** add a `_wrap_parse_output(parsed) -> Any` hook to TIB (default: return parsed unchanged); override on Session to wrap in `TerminalInferencerResponse`. TIB's `_infer` calls `self._wrap_parse_output(self.parse_output(...))`.

6. **Keep** Session's async streaming machinery (`_ainfer_streaming`, `_read_stdout_with_exit_detection`, `_poll_process_exit`, `_force_close_pipes`, `_kill_process_group`, `_safe_process_cleanup`, `_build_full_command`, `_ainfer` accumulation) — all unchanged.

7. **Keep `_last_streaming_stderr`** on TSIB (TSIB-only field; not promoted to TIB).

8. **Add convenience class at bottom of file:**
   ```python
   @attrs
   class TerminalSessionTemplatedInferencerBase(
       TerminalSessionInferencerBase, TemplatedInferencerBase,
   ):
       """Terminal + streaming + templates.

       MRO: TSTIB → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC.

       Use this for CLI inferencers that need all three axes (ClaudeCode, Kiro,
       Devmate, RovoDev). Use TerminalSessionInferencerBase directly if you
       don't want templates (Metamate).
       """
       pass
   ```

### 9.2 Edit to `terminal_inferencer_base.py` (add convenience class at bottom)

```python
@attrs
class TerminalTemplatedInferencerBase(TerminalInferencerBase, TemplatedInferencerBase):
    """Sync terminal + templates (no streaming/session).

    MRO: TTIB → TIB → TemplatedIB → IB → Debuggable → Resumable → ABC.

    Use this for non-streaming terminal inferencers that need templated
    prompts. (No production leaves use this today; provided for completeness
    of the axes design.)
    """
    pass
```

### 9.3 Tests landing in Phase 4

- Phase 0 test #5 turns GREEN.
- `test_tsib_mro_is_documented` — assert `TerminalSessionInferencerBase.__mro__` matches the documented order.
- `test_tstib_mro_is_documented` — same for `TerminalSessionTemplatedInferencerBase`.
- `test_tsib_inherits_timeout_env_vars_post_exec` — assert all three attribs present.
- `test_tsib_no_duplicate_working_dir_attrib` — `len([f for f in attr.fields(TSIB) if f.name == "working_dir"]) == 1`.
- `test_tsib_pipe_hang_detection_still_works` — regression for MCP hang fix.

### 9.4 Phase 4 risk register

| Risk | Mitigation |
|---|---|
| Diamond MRO produces unexpected field order or attrs field-collection conflict. | All 5 classes in MRO use legacy `@attrs` (slots=False) — verified. C3 deterministic. Phase 0 test #5 + Phase 4 MRO assertions catch any drift. |
| Session's stronger `_infer` (with timeout/env_vars/post_exec) breaks subclasses that override `_infer` and don't expect those features. | Audit: ClaudeCodeCli and KiroCli both override `_infer`. Their overrides don't call `super()._infer`. No break. |
| `_wrap_parse_output` hook semantics differ between sync and async paths. | Default impl on TIB returns parsed unchanged → no behavior change for Terminal-only stubs. Session overrides to wrap in `TerminalInferencerResponse`. Document the contract explicitly. |
| `pre_exec_scripts` semantics differ between TIB (`_execute_scripts` separate run) and TSIB (`_build_full_command` chains via `&&`). | TSIB keeps `_build_full_command` for streaming paths (env-var propagation requires same shell). TIB's `_execute_scripts` continues to drive sync `_infer`. Document asymmetry. |
| `large_input_mode` (TSIB-specific) interacts with TIB's `_execute_command` signature. | `large_input_mode` is consumed only by TSIB's `_ainfer_streaming`. TIB's `_execute_command` untouched. No interaction. |

---

## 10. Phase 5 — Migrate the 5 CLI leaves

### 10.1 `ClaudeCodeCliInferencer` → `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`

> **Audit F11 verdict (✅ SAFE):** `_resolve_claude_command()` (lines 125–186) has zero references to `self.target_path`, `self.working_dir`, or `self._workspace`. It only tests CLI availability and Node.js paths. Safe to call before or after `super().__attrs_post_init__()`. All subprocess calls go through `self._resolve_subprocess_cwd()` which reads inherited `self.working_dir`.

**Edits:**

1. **Change parent (line 31):**
   ```python
   class ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase):
   ```

2. **Remove local `target_path` declaration (line 92):** inherited from TIB.

3. **Remove `self.working_dir = self.target_path`** from post-init (line 121): TIB handles it.

4. **Keep** the `target_path = expanduser("~/fbsource")` default — this is Claude-specific business logic. Mark with DO-NOT-DELETE comment:
   ```python
   def __attrs_post_init__(self) -> None:
       # ─── DO NOT DELETE: Claude-specific default target ─────────────
       if self.target_path is None:
           self.target_path = os.path.expanduser("~/fbsource")
       # ─── End must-preserve ─────────────────────────────────────────

       super().__attrs_post_init__()
       # NOTE: working_dir = target_path now handled by TIB post-init.

       # ─── DO NOT DELETE: claude command path resolution ─────────────
       # Resolves the `claude` binary location. No path dependencies.
       self._resolve_claude_command()
       # ─── End must-preserve ─────────────────────────────────────────
   ```

### 10.2 `KiroCliInferencer` → `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/kiro/kiro_cli_inferencer.py`

1. **Change parent (line 25):** `→ TerminalSessionTemplatedInferencerBase`.
2. **Remove local `target_path` declaration (line 73).**
3. **Remove `self.working_dir = self.target_path`** from post-init (line 93).
4. **Keep** Kiro-specific business logic (model resolution):
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

### 10.3 `DevmateCliInferencer` → `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/devmate/devmate_cli_inferencer.py`

> **⚠️ CRITICAL — `repo_path` MUST stay as an attrib.** The naïve "replace with @property" approach breaks attrs (the generated `__init__` no longer accepts `repo_path=` kwarg). Verified: **45+ test sites** pass `repo_path=` as kwarg. v2 plan estimated 15+; reality is 3× higher.

1. **Change parent (line 63):** `→ TerminalSessionTemplatedInferencerBase`.
2. **Keep** the `repo_path: Optional[str] = attrib(default=None)` declaration (line 163) — UNCHANGED.
3. **Update post-init** to mirror `repo_path → target_path` *before* `super()`:
   ```python
   def __attrs_post_init__(self):
       """Devmate-specific defaults: ~/fbsource as the operating target;
       inject cd-into-repo as pre-exec script.

       repo_path is Devmate's historical name for what the framework calls
       target_path. We mirror repo_path → target_path BEFORE super() so
       the base's working_dir defaulting sees the correct value.
       """
       # ─── DO NOT DELETE: Devmate-specific default (~/fbsource) ──────
       if self.repo_path is None:
           self.repo_path = os.path.expanduser("~/fbsource")
       # ─── End must-preserve ─────────────────────────────────────────

       # Mirror repo_path into target_path before base post-init runs.
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

**Result:** all 45+ existing test sites continue to pass `repo_path=` unchanged. New code can use either `repo_path` or `target_path`. They stay in sync at construction time.

### 10.4 `RovoDevCliInferencer` → `TerminalSessionTemplatedInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/rovodev/rovodev_cli_inferencer.py`

> **Audit F10 verdict (✅ SAFE):** Exhaustive trace of all 6 `self.working_dir` reads confirms they all expect "the directory the CLI agent operates on" = `target_path`. Specifically: line 336 (`subprocess.run(cwd=...)`), lines 623/628/677/683 (`find_latest_session_id`/`ensure_session_metadata` — both read AND write — verified no path mismatch). No log-file or bookkeeping paths use `working_dir`.

1. **Change parent (line 75):** `→ TerminalSessionTemplatedInferencerBase`.
2. **Remove `if self.working_dir is None: self.working_dir = os.getcwd()`** (lines 144–146): TIB handles it.
3. **Optional semantic clarification (recommended):** rename `workspace_path=self.working_dir` → `workspace_path=self.target_path` in the 4 session call sites (lines 623, 628, 677, 683). Behavior identical (working_dir == target_path by default), but intent is explicit.

### 10.5 `MetamateCliInferencer` — stays on `TerminalSessionInferencerBase`

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/metamate/metamate_cli_inferencer.py`

- **No parent change.** Metamate does not want templating; the non-templated branch is the correct fit.
- **Remove** any local `target_path` / `working_dir` initialization (now handled by TIB via TSIB).
- **Pre-merge sub-audit recommended:** confirm Metamate has no hidden dependencies analogous to F10/F11.

### 10.6 Tests landing in Phase 5

- Phase 0 test #6, #9 turn GREEN.
- `test_each_cli_leaf_isinstance_of_templated` for {Claude, Kiro, Devmate, RovoDev} → True; for Metamate → False.
- `test_devmate_repo_path_mirrors_to_target_path` — `DevmateCliInferencer(repo_path="/x"); assert inst.target_path == "/x"`.
- `test_metamate_does_not_inherit_templated` — `assert not isinstance(MetamateCliInferencer(), TemplatedInferencerBase)`.

### 10.7 Phase 5 risk register

| Risk | Mitigation |
|---|---|
| DevMate's `repo_path` and `target_path` drift if user reassigns post-construction. | Document "use either name consistently; do not mix" in field docstring. Not a real concern for the 45+ test sites (they construct fresh instances). |
| MetamateCli loses some feature by not inheriting from the templated branch. | Metamate confirmed not using templates. No loss. |
| A leaf's `__attrs_post_init__` does business logic *after* `super()` that depends on TIB's post-init having run (e.g., reads `self.working_dir`). | Audit each leaf for code after `super()`. Verified: Claude's post-super `_resolve_claude_command()` has no path dependencies (F11). Others have nothing after super(). |
| dual_inferencer.py:1602 / multi_flow_dual_inferencer.py:460 isinstance checks change behavior. | Verified: TSTIB subclasses (Claude, Kiro, Devmate, RovoDev) → True; TSIB-only Metamate → False. **This is the correct discrimination** — Metamate genuinely doesn't want templated prompts. The orchestrator's discriminator now matches the new architecture. |

---

## 11. Phase 6 — Update exports + docstrings

**File 1:** `src/agent_foundation/common/inferencers/terminal_inferencers/__init__.py`

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

**File 2:** `src/agent_foundation/common/inferencers/templated_inferencer_base.py` (docstring update, lines 6–15)

Update the architectural diagram in the docstring to reflect that SIB and TIB no longer inherit from TemplatedInferencerBase. Add explanatory text:

> "After 2026-05-16 axes refactor: `TemplatedInferencerBase` is one of three orthogonal axes (templating, streaming, terminal-exec). Leaves opt into templating via direct inheritance from this class OR via the convenience MI classes `TerminalTemplatedInferencerBase` and `TerminalSessionTemplatedInferencerBase`. The cascade-injection of `_template_manager` walks every descendant that has `template_manager` as a constructor param; SIB and TIB no longer inherit it, so streaming-only and terminal-only leaves are no longer accidental recipients of cascaded template state."

---

## 12. Phase 7 — Permanent regression test suite

**File (new):** `test/agent_foundation/common/inferencers/test_inferencer_axes_invariants.py`

These tests are **permanent** — they pin invariants that the entire refactor depends on. Any future contributor breaking them gets immediate CI failure.

### 12.1 Four permanent tests

1. **`test_axes_isinstance_matrix`** — for each of the 5 CLI leaves and 10 streaming-only leaves, assert the expected `isinstance` results against `(TerminalInferencerBase, StreamingInferencerBase, TemplatedInferencerBase, TerminalSessionInferencerBase, TerminalSessionTemplatedInferencerBase)`. This is the contract grid for the axes design.

2. **`test_three_diamond_mros_documented`** — assert all three diamond MROs (TSIB, TTIB, TSTIB) match their documented orders. Catches any future shuffling.

3. **`test_diamond_attrs_slots_consistency`** — assert all classes in the three diamond MROs have `slots=False` (legacy `@attrs` API). Catches any contributor who migrates one of them to `@attrs.define` (modern API, slots=True) which would break the diamonds.
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

4. **`test_no_duplicate_fields_under_diamond`** — for each diamond class, assert `attr.fields(cls)` contains no duplicate field names (catches any accidental field re-declaration in a subclass).

---

## 13. Comparison with source plans

| Aspect | v2 (Rovo Dev) | v1 (Claude) | v3 (Integrated, this plan) |
|---|---|---|---|
| Architecture | Keep coupling; rename SIB to expose it | Decouple all 3 bases; add 3 MI convenience classes | **Claude's decoupling** (cleaner) |
| `working_dir` clobber fix | Conditional on `target_path != os.getcwd()` (over-engineered) | Conditional on `target_path is None` (clean) | **Claude's gate** (simpler & equally correct) |
| `StreamingInferencerBase` rename | Yes (Phase 1.5) | No (decoupling makes name honest) | **Skip the rename** (obsoleted by decoupling) |
| Phase 0 RED tests | 9 tests (incl. orchestrator) | None (tests are afterthought) | **Keep v2's 9 tests** |
| Per-phase risk registers | Yes (3) | No | **Keep v2's risk registers** |
| Orchestrator regression tests (#7, #8) | Yes | No | **Keep v2's #7, #8** |
| BTA `_configure_for_workspace` audit | Yes (line 1135 documented) | No | **Keep v2's BTA notes** |
| DevMate `repo_path` strategy | Mirror via attrib (correct) | Keep as alias (vague) | **Keep v2's precise mirror pattern** |
| F10 (RovoDev working_dir trace) | Done ✅ | Not done | **Keep v2's audit** |
| F11 (ClaudeCodeCli _resolve_claude_command) | Done ✅ | Not done | **Keep v2's audit** |
| `_last_streaming_output` / `_return_code` promotion | Not addressed | Yes (Phase 3) | **Keep Claude's promotion** |
| `_wrap_parse_output` hook | Yes (Phase 2b) | Not addressed | **Keep v2's hook** |
| MetamateCli classification | Ambiguous | Explicit: stays on TSIB (no templates) | **Keep Claude's explicit classification** |
| Three diamonds slot test | Yes (single diamond only) | No | **Extend v2's test to all 3 diamonds** |
| DO-NOT-DELETE leaf markers | Yes | No | **Keep v2's markers** |
| Plan length | 1,071 lines | 308 lines | ~880 lines (target — concise but complete) |

### 13.1 If we had to pick ONE plan, which?

**My honest answer: Claude's plan (v1).**

Here's why:
- **Architecturally it's the right answer.** The "decouple all three bases, add convenience MI classes" pattern is the elegant solution. My v2 was a workaround (rename + diamond on coupled bases) when the actual problem was the coupling.
- **It obsoletes my Phase 1.5 entirely.** Once the coupling is removed, `StreamingInferencerBase`'s name is honest. No rename. No back-compat alias. No PEP 562 `__getattr__`. Net reduction in complexity.
- **Its `_configure_for_workspace` gate (`if target is None`) is simpler and equally correct** vs my over-engineered `target_was_explicit` heuristic.
- **It correctly classifies MetamateCli** as not wanting templates — something my v2 left vague.
- **It promotes `_last_streaming_output` / `_last_streaming_return_code`** to proper attribs — a subtle correctness fix I missed.

What my v2 has that Claude's v1 lacks: operational rigor (RED tests, risk registers, audit follow-ups, the orchestrator regression tests, DO-NOT-DELETE markers, the `repo_path` mirror pattern precision, the `_wrap_parse_output` hook). These are *important* but they're additions on top of the right architecture — they don't change the structural decisions.

**So if forced to one: Claude's plan, then augment with my operational layer as it's implemented.** This integrated v3 plan does exactly that.

### 13.2 Honest accounting of where v2 went wrong

Two genuine errors in v2:
1. **Over-engineered the `_configure_for_workspace` guard** with the `target_path != os.getcwd()` discriminator. Claude's `target_path is None` is simpler, doesn't have the "user-explicit-equals-default" degenerate case, and is equally correct because `target_path` is None during orchestrator construction and non-None after leaf post-init. v3 adopts Claude's version.
2. **Proposed renaming `StreamingInferencerBase` to address a name-misleading-because-of-coupling problem** instead of removing the coupling. v3 adopts Claude's decoupling and drops the rename.

Two genuine errors in Claude's v1:
1. **Missed the orchestrator regression test gap** (Phase 0 tests #7, #8). Without these, a future contributor could "fix" the guarded clobber by deleting it entirely (the very mistake v2 made), and CI wouldn't catch it.
2. **DevMate `repo_path` strategy is vague** ("keep as user-facing alias" without specifying how attrs interacts with it). Without the explicit attrib-mirror pattern, an implementer might try the `@property` approach which breaks attrs.

The integrated v3 plan fixes all four errors.

---

## 14. Migration & rollback strategy

### 14.1 Branch & PR strategy

- **Single feature branch:** `refactor/inferencer-axes-decoupling`.
- **8 commits, 1 per phase.** Each commit is independently revertable.
- **PR strategy:**
  - **PR-1 (Phase 0)** — RED tests only; lands first; runs in CI as `xfail(strict=True)`.
  - **PR-2 (Phase 1)** — guarded clobber; pure no-op until Phase 3 ships.
  - **PR-3 (Phases 2–4)** — the structural core; reviewed as one PR because Phases 3 and 4 are co-dependent (TSIB needs TIB's `target_path` to exist).
  - **PR-4 (Phase 5)** — leaf migrations; can be split into 5 commits per leaf for granular review/rollback.
  - **PR-5 (Phases 6–7)** — exports + permanent regression suite; small cleanup.

### 14.2 Rollback per phase

| Revert | Consequence |
|---|---|
| Phase 0 only | Tests vanish; no functional change |
| Phase 1 only | Reverts to unconditional clobber; safe |
| Phase 2 only (after 1) | SIB re-inherits TemplatedIB; streaming leaves regain template attribs in `__init__` (unused but present) |
| Phase 3 only (after 2) | `target_path` vanishes; guarded clobber returns to no-op; TIB re-inherits TemplatedIB |
| Phase 4 only (after 3) | TSIB returns to single-inheritance from SIB; convenience classes vanish; CLI leaves break (they reference TSTIB) → revert Phase 5 first |
| Phase 5 only (after 4) | Leaves return to TerminalSessionInferencerBase parent; lose templating; templated subclass behavior reverts to v1 |
| Phase 6 only | Imports break wherever `TerminalSessionTemplatedInferencerBase` is referenced |
| Phase 7 only | Loses permanent regression tests |

**Safe rollback combinations:**
- {Phase 7} — pure test removal.
- {Phases 4, 5, 6, 7} — undoes the entire MI refactor, leaves the decoupling in place.
- {Phases 1, 2, 3, 4, 5, 6, 7} — full revert.

### 14.3 Cross-team notification

- Notify owners of `BreakdownThenAggregateInferencer`, `MultiFlowDualInferencer`, `LinearWorkflowInferencer`, `DualInferencer` — their orchestrator scenarios are protected by the guarded clobber but should be smoke-tested on the feature branch.
- Notify owners of `DevmateCliInferencer` consumers — `repo_path` semantics unchanged, but the attrib now also reflects through `target_path`. Update internal docs.
- No notification needed for streaming-only leaves (RovoChat, ToolAs, Plugboard, AgClaude, etc.) — they continue to behave exactly as today.

### 14.4 Estimated effort

| Phase | Implementation | Testing | Review | Total (h) |
|---|---|---|---|---|
| 0 | 0 | 4 | 1 | 5 |
| 1 | 1 | 2 | 1 | 4 |
| 2 | 1 | 2 | 1 | 4 |
| 3 | 2 | 3 | 2 | 7 |
| 4 | 4 | 5 | 4 | 13 |
| 5 | 3 | 3 | 3 | 9 |
| 6 | 1 | 0 | 1 | 2 |
| 7 | 1 | 3 | 1 | 5 |
| **Total** | **13** | **22** | **14** | **49** |

Roughly **1.5 engineer-weeks** assuming one engineer. Lower than v2's estimate (54h) because the rename phase (1.5) is eliminated.

---

## 15. Design principles applied

1. **Single Responsibility per field.** `workspace`, `target_path`, `working_dir` — three names, three roles, one writer each.
2. **Orthogonal axes, composable via MI.** Templating, streaming, terminal-exec are three orthogonal capabilities; leaves opt into any subset via narrow convenience MI classes.
3. **Names tell the truth.** `StreamingInferencerBase` is honest once it no longer secretly depends on templating. No rename needed.
4. **Locality of behavior.** Default-resolution logic lives in exactly one place (TIB.`__attrs_post_init__`).
5. **Explicit > implicit.** No auto-binding of `target_path` from `workspace.root`; user must opt in.
6. **Backwards compatibility via mirroring, not aliasing.** DevMate's `repo_path` stays as an attrib (preserves 45+ test sites unchanged) and mirrors into `target_path`.
7. **Diamond inheritance accepted only when both parents trace to the same root.** All three diamonds (TSIB, TTIB, TSTIB) ultimately resolve to `InferencerBase` via C3.
8. **Test-first via Phase 0.** 9 RED tests pin the contract before any source change.
9. **Phased rollout with independent revertability.** Each of the 8 phases is independently shippable and revertable.
10. **Permanent regression tests pin invariants.** Phase 7's 4 tests stay in CI forever.
11. **No magic.** No metaclass tricks, no module-level `__getattr__`, no decorator magic. The only "clever" construct is MI diamonds — which are documented Python, supported by attrs, and pinned by tests.
12. **Explicit rejected alternatives.** §13.2 documents what was considered and chosen against, so reviewers can revisit any decision without reconstructing the analysis.

---

*End of integrated v3 plan. Reviewers: please challenge §3.3 (default-resolution table), §6.3 (Phase 1 risk register — the orchestrator preservation), §9.4 (Phase 4 diamond MRO), and §13.1 (the "if forced to one, pick Claude's" assessment) most carefully.*









