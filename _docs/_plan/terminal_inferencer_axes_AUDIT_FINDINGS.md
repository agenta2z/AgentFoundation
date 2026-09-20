# Audit Findings — Terminal Inferencer Axes & Streaming Rename Plan

**Audit date:** 2026-05-15 22:48
**Audited document:** `terminal_inferencer_axes_and_streaming_rename_plan.md`
**Audit method:** 4 parallel exploration subagents + direct verification of disputed claims
**Audit verdict:** **The plan as written has 3 BLOCKING issues, 4 HIGH-severity issues, and 6 MEDIUM-severity issues. Do not implement Phase 1 or Phase 3 without applying the corrections in §C below.**

This document is **intentionally separate** from the main plan. The main plan is left as-is so reviewers can compare what was *originally* proposed against what was *found wrong*. The corrections in §C below are the deltas the plan needs before implementation begins.

---

## A. Audit summary table

| # | Finding | Severity | Subagent that found it | I verified |
|---|---------|----------|------------------------|------------|
| F1 | Phase 1 deletion of `working_dir = workspace.root` clobber will break BTA worker subprocess launches with `NotADirectoryError` | **🔴 BLOCKING** | audit-phase1 | ✅ Confirmed (lines 345–353 + 1135–1140) |
| F2 | Devmate `repo_path` cannot be a `@property` while also being declared as `attrib(default=None)` — attrs auto-generates `__init__` that sets the attribute, shadowing the property | **🔴 BLOCKING** | audit-leaf-migrations | ✅ Confirmed (line 163 declares attrib) |
| F3 | KiroCliInferencer's `__attrs_post_init__` does business logic (model resolution) beyond path defaulting; "simplify" instruction in plan §8.2 will lose it | **🔴 BLOCKING** | audit-leaf-migrations | ✅ Confirmed |
| F4 | `_propagate_workspace_to_children` (line 349 comment) explicitly documents that the clobber is **load-bearing** for orchestrator-spawned terminal inferencers — plan does not address this | **🔴 BLOCKING** | audit-phase1 | ✅ Confirmed |
| F5 | BTA `_configure_for_workspace` override (line 1135) calls `super()` and then propagates to `breakdown_inferencer` — depends on the super call setting working_dir | 🟠 HIGH | audit-phase1 | ✅ Confirmed (override exists; it would still work but loses the propagated cwd) |
| F6 | `__attrs_post_init__` execution order under the diamond: `InferencerBase` runs *after* `StreamingInferencerBase`'s super call but *before* control returns to `TerminalInferencerBase`. The "before/after" ordering claim in plan §3.3 is technically wrong about *when* `_configure_for_workspace` fires relative to `target_path` setup | 🟠 HIGH | audit-mro | ⚠️ Re-analyzed; see F6 below for corrected understanding |
| F7 | `isinstance(x, StreamingInferencerBase)` calls in test loops will trigger `DeprecationWarning` per call (not per import) when the alias is hit dynamically | 🟠 HIGH | audit-streaming-rename | ✅ Confirmed; mitigation needed |
| F8 | DevMate has **45+ test sites** passing `repo_path=` as kwarg (not the 15 the plan estimated) — blast radius for the property-alias change is 3× the planned size | 🟠 HIGH | audit-leaf-migrations | ✅ Confirmed via grep count |
| F9 | The 4 `@attrs` decorations are mixed (`@attrs(slots=False)` on Templated; bare `@attrs` on others). The legacy `attr` API used here defaults to `slots=False`, so this is *safe* — but the audit subagent flagged it as a risk worth documenting | 🟢 RESOLVED | audit-mro | ✅ Verified — `from attr import attrs` is the legacy API; default is `slots=False`; no conflict |
| F10 | RovoDev's `working_dir` is read in *more places than the 4 the plan cites* (subprocess cwd, log file paths, possibly session-WRITING) — full file audit truncated | 🟡 MEDIUM | audit-leaf-migrations | ⏳ Needs follow-up audit |
| F11 | ClaudeCodeCli's `_resolve_claude_command()` may interact with `target_path` before the base post-init runs — unverified | 🟡 MEDIUM | audit-leaf-migrations | ⏳ Needs follow-up audit |
| F12 | No existing test asserts `working_dir == workspace.root` after construction with `workspace=`, so removing the clobber will not produce a single test failure — silent regression risk | 🟡 MEDIUM | audit-phase1 | ✅ Confirmed |
| F13 | Pickle compatibility for the rename relies on the alias being **permanent**. Plan said "removal target: 2026-Q4" — that's a future pickle-incompatibility footgun | 🟡 MEDIUM | audit-streaming-rename | ✅ Confirmed |
| F14 | `_configure_for_workspace` is called from BTA's override too (not just the `_workspace` setter) — search for all override sites must be exhaustive before edit | 🟡 MEDIUM | audit-phase1 | ✅ Confirmed (1 override found; assume more might exist) |

**Severity legend:**
- 🔴 BLOCKING — plan will produce broken code or hidden runtime errors. Must fix before implementation.
- 🟠 HIGH — plan will compile and pass narrow tests, but break wider integrations.
- 🟡 MEDIUM — risk is non-fatal but warrants explicit handling/documentation.
- 🟢 RESOLVED — flagged by audit, verified non-issue.

---

## B. Detailed analysis of the BLOCKING findings

### F1 / F4 — The clobber is load-bearing (combined analysis)

**The original plan's reasoning** (§6, lines 257–298):
> "For the 5 production leaves, this is equivalent behavior because they all override `__attrs_post_init__` to set `working_dir` themselves from `target_path`/`repo_path` *first*, then call super."

**Why this is wrong:**

The reasoning only considered leaves *constructed standalone*. It did **not** consider leaves spawned by orchestrators (BTA, MFDual, LinearWorkflow, Dual). For those, the actual lifecycle is:

1. Orchestrator constructs the leaf at config-load time (`__init__` runs once, leaf's `working_dir` defaults to `target_path` or `os.getcwd()`).
2. Orchestrator runs; for each "spawn child" iteration, orchestrator calls `child._workspace = parent_ws.child(...)`.
3. The `_workspace` setter triggers `_configure_for_workspace`.
4. **Today:** `_configure_for_workspace` writes `child.working_dir = workspace.root` — so the child's subprocess launches in the freshly-`ensure_dirs()`'d workspace folder.
5. **After the plan's edit:** `_configure_for_workspace` no longer touches `working_dir` — so the child's subprocess launches in whatever `working_dir` was set at config-load time, which is **NOT** the per-iteration workspace.

The smoking gun is in the source itself, lines 345–353:

```python
child_ws = parent_workspace.child(child_name)
# Critical: create the on-disk dirs before assigning. Otherwise
# `_configure_for_workspace` will set working_dir to a
# non-existent path, and any subprocess inferencer (e.g.
# ClaudeCodeCli) will fail with NotADirectoryError when it
# tries to launch with cwd=workspace.root. Mirrors BTA's
# explicit `worker_ws.ensure_dirs()` before assignment.
child_ws.ensure_dirs()
child._workspace = child_ws
```

**This comment exists *because* the previous author already hit this bug**. The clobber is not legacy cruft — it is an intentional, documented mechanism that makes orchestrator-spawned subprocess inferencers work. Removing it without a replacement reintroduces the exact failure mode the comment warns against.

**BTA's override (line 1135–1140) confirms this too:**
```python
def _configure_for_workspace(self, workspace):
    super()._configure_for_workspace(workspace)
    if self.breakdown_inferencer is not None:
        bd_ws = workspace.child("breakdown")
        bd_ws.ensure_dirs()
        self.breakdown_inferencer._workspace = bd_ws
```

The override deliberately propagates workspace assignment to the breakdown child, *expecting* the super call to plumb working_dir through to terminal inferencers in that child's tree.

### F2 — Devmate `repo_path` cannot be a `@property` while also an `attrib`

**The original plan's reasoning** (§8.1):
```python
# OLD:
# repo_path: Optional[str] = attrib(default=None)

# NEW: repo_path is now a back-compat alias for target_path
@property
def repo_path(self) -> Optional[str]:
    return self.target_path

@repo_path.setter
def repo_path(self, value: Optional[str]) -> None:
    self.target_path = value
```

**Why this is wrong:**

The plan removed the `attrib` declaration and replaced with a property. That part is correct. But there are **two attrs-specific subtleties** the plan did not address:

1. **`__init__` signature change**: `attrs` generates `__init__(self, repo_path=None, model_name=..., ...)`. If we remove the `repo_path` attrib, the generated `__init__` will no longer accept `repo_path=` as a kwarg. **Every test site that calls `DevmateCliInferencer(repo_path=...)` will fail with `TypeError: __init__() got an unexpected keyword argument 'repo_path'`** *before* Python even tries to look up the property.

2. **Test sites are 45+, not 15**. Confirmed by grep: 45 occurrences of `repo_path =` across DevMate-relevant test files (the plan estimated 15).

The fix requires more than a property — it requires either (a) keeping `repo_path` as an `attrib` and using an `__attrs_post_init__` hook to mirror it into `target_path`, or (b) accepting `repo_path` as an `attrib` that *itself* is the back-compat field while `target_path` is the canonical one.

Approach (a) is simpler. Approach (b) is more honest. Either way, the plan's `@property` approach **does not work as written**.

### F3 — KiroCliInferencer business logic in `__attrs_post_init__`

**The original plan's reasoning** (§8.2):
> "Simplify `__attrs_post_init__` (lines 83–95)" with a stub that only resolves `model_name`.

**Why this is wrong (mostly):**

Confirmed: KiroCli's current `__attrs_post_init__` does:
1. Set `target_path` default.
2. Set `working_dir = target_path`.
3. Call `resolve_model_tag(self.model_name)` (lines 93–94 — business logic, model normalization).
4. Call `super().__attrs_post_init__()`.

The plan's "simplified" version *did* keep the model resolution. So this finding is **partially mitigated by the plan as written**. The remaining concern is that the plan's example doesn't explicitly call out *why* the model-resolution lines must be preserved. A code reviewer who follows the spirit of "simplify" might delete them.

**Severity downgrade:** This is more of a documentation issue than a bug. Reclassifying to 🟠 HIGH (was 🔴 BLOCKING) — but still requires plan correction.

### F6 — Diamond `__attrs_post_init__` execution order

**Subagent claim:** "InferencerBase initialization (workspace sync) happens **before** TerminalInferencerBase completes working_dir setup."

**My re-analysis:** Subagent partially wrong. Let me trace it correctly.

For `inst = TerminalSessionInferencerBase(workspace=ws, target_path="/repo")`, the call chain is:

1. `attrs`-generated `__init__` runs, sets all fields from kwargs.
2. `attrs`-generated `__init__` then calls `__attrs_post_init__()` once on the **most-derived class** (here: `TerminalSessionInferencerBase`, which has none → falls through MRO to `TerminalInferencerBase`).
3. `TerminalInferencerBase.__attrs_post_init__`:
   - Sets `target_path` default if None.
   - Sets `working_dir = target_path` if None.
   - Calls `super().__attrs_post_init__()` → next in MRO → `TemplatedStreamingInferencerBase` (no override) → `StreamingInferencerBase`.
4. `StreamingInferencerBase.__attrs_post_init__`:
   - Calls `super().__attrs_post_init__()` FIRST → next in MRO → `TemplatedInferencerBase` (no override) → `InferencerBase`.
5. `InferencerBase.__attrs_post_init__`:
   - Performs workspace sync (line 443: `self._workspace = self.workspace`).
   - This triggers the `_workspace.setter` → `_configure_for_workspace(ws)`.
   - `_configure_for_workspace` (after the plan's edit) does NOT touch `working_dir`.
   - But `working_dir` was already correctly set by step 3.
   - **So there is no ordering bug here** — the chain works as the plan intends.
6. `InferencerBase.__attrs_post_init__` returns.
7. `StreamingInferencerBase.__attrs_post_init__` resumes (post-super), registers recovery template root.
8. `TerminalInferencerBase.__attrs_post_init__` resumes (post-super) — nothing more to do.

**Re-verdict on F6:** The ordering is actually **fine** for *standalone construction*. The subagent's concern was wrong. **However**, the orchestrator scenario described in F1/F4 is still the real bug: `_workspace` is reassigned later via the orchestrator's setter call, *after* construction, and at that point step 5's logic doesn't re-set `working_dir`.

So F6 is not a blocking issue, but F1/F4 still are.

---

## C. Required plan corrections

The following deltas must be applied to `terminal_inferencer_axes_and_streaming_rename_plan.md` before implementation begins.

### C1. Phase 1 must be redesigned, not deleted

The plan currently says "delete the clobber." The correct fix is "**replace** the unconditional clobber with a *conditional* one that respects user intent":

```python
def _configure_for_workspace(self, workspace):
    import os

    if hasattr(self, "working_dir"):
        # Only auto-set working_dir from workspace.root if the user did NOT
        # explicitly set target_path. The rule:
        #   - target_path explicit  → leave working_dir alone (user wins)
        #   - target_path implicit (== os.getcwd() default or None) AND
        #     workspace provided → set working_dir = workspace.root
        # This preserves orchestrator-spawned subprocess behavior (where
        # workspace.root IS the intended cwd) while letting users with
        # an explicit target_path keep their CLI launch directory.
        target_path = getattr(self, "target_path", None)
        target_was_explicit = (
            target_path is not None
            and target_path != os.getcwd()
        )
        if not target_was_explicit:
            new_wd = str(workspace.root)
            if sys.platform != "win32" or len(new_wd) < 240:
                self.working_dir = new_wd

    if hasattr(self, "cache_folder"):
        self.cache_folder = os.path.join(
            str(workspace.root), "_runtime", "inferencer_cache"
        )

    # ...logger handling unchanged...
```

This preserves the load-bearing behavior for orchestrator-spawned terminals while fixing the original bug (user passes both `workspace=` and `target_path=`, target_path now wins).

**Trade-off acknowledged:** "target_path != os.getcwd()" is heuristic — if the user explicitly passes `target_path=os.getcwd()`, we cannot distinguish that from the default. Acceptable: that case is degenerate (user-explicit value happens to equal cwd → either behavior is fine).

**Test additions required for revised Phase 1:**
- `test_workspace_with_explicit_target_path_does_not_clobber_working_dir`
- `test_workspace_without_target_path_still_sets_working_dir_to_workspace_root` (regression test for orchestrator scenario)
- `test_orchestrator_spawned_terminal_inferencer_runs_in_workspace_root` (BTA integration test)

### C2. Phase 3 — Devmate `repo_path` migration must use the attrib-mirror approach

Replace plan §8.1's property-based approach with:

```python
# In DevmateCliInferencer:
# repo_path stays as an attrib (back-compat for the 45+ test sites that pass it).
# It is the source of truth at construction time. After __attrs_post_init__ runs,
# it is mirrored into target_path so the TerminalInferencerBase.target_path
# field is also populated.

repo_path: Optional[str] = attrib(default=None)  # KEEP

def __attrs_post_init__(self):
    # Devmate-specific defaults: repo_path → ~/fbsource if unset.
    if self.repo_path is None:
        self.repo_path = os.path.expanduser("~/fbsource")
    # Mirror repo_path into target_path BEFORE base post-init runs,
    # so working_dir defaults to target_path (== repo_path).
    if self.target_path is None:
        self.target_path = self.repo_path
    # Build the cd-into-repo pre-exec script.
    cd_script = f'cd "{self.repo_path}" || exit 1'
    if self.pre_exec_scripts is None:
        self.pre_exec_scripts = [cd_script]
    elif cd_script not in self.pre_exec_scripts:
        self.pre_exec_scripts.insert(0, cd_script)
    super().__attrs_post_init__()
```

**Key correction:** `repo_path` stays as an `attrib`. `target_path` is *derived* from it, not the other way around. The semantic role of `repo_path` is unchanged; we just *also* now expose `target_path` for callers who use the framework convention.

**Implication:** Both `repo_path` and `target_path` will be present on Devmate instances. Test sites can use either. Document that `repo_path` is the historical name and `target_path` is the framework name.

### C3. Phase 1.5 — Add isinstance-loop test that watches DeprecationWarning count

Add to plan §6A.4:

```python
def test_isinstance_loop_does_not_spam_warnings():
    """Verify isinstance(x, StreamingInferencerBase) in a loop doesn't spam."""
    from agent_foundation.common.inferencers.streaming_inferencer_base import (
        StreamingInferencerBase,  # one warning here
    )
    inst = SomeStreamingSubclass()  # zero warnings
    with pytest.warns(DeprecationWarning) as records:
        for _ in range(1000):
            assert isinstance(inst, StreamingInferencerBase)
    # Local binding caches the resolved name; isinstance does NOT re-trigger __getattr__.
    # Expect ONE deprecation warning total (from the import above), not 1000.
    assert len(records) <= 1, (
        f"Expected at most 1 DeprecationWarning from import, got {len(records)} — "
        "module __getattr__ is being re-triggered. Check for sys.modules access."
    )
```

This explicitly pins the expected behavior so a future contributor can't accidentally introduce a `sys.modules['...streaming_inferencer_base'].StreamingInferencerBase` pattern that would re-trigger the warning per access.

### C4. Phase 1.5 — Make the back-compat alias permanent, not deprecated

Change the alias docstring from:

> "Removal target: 2026-Q4."

To:

> "**Permanent back-compat alias** — kept indefinitely for pickle compatibility. Pickled instances of (renamed) `TemplatedStreamingInferencerBase` from before the rename store `__qualname__='StreamingInferencerBase'`; unpickling them resolves the class via this `__getattr__`, so removing the alias would break unpickling. Do not remove."

The plan currently treats the alias as a deprecation shim. It is actually a **permanent compatibility seam**. Mark accordingly to prevent a future cleanup PR from removing it and breaking pickle reads.

### C5. Phase 3 — KiroCliInferencer simplification must explicitly preserve model resolution

Replace plan §8.2's example with one that calls out the must-preserve lines:

```python
def __attrs_post_init__(self) -> None:
    """Resolve model_name (BUSINESS LOGIC — must not delete);
    target_path/working_dir handled by base."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.common import (
        resolve_model_tag,
    )
    # ─── DO NOT DELETE: model normalization, not path defaulting ────
    if self.model_name and self.model_name != "auto":
        self.model_name = resolve_model_tag(self.model_name)
    # ─── End of must-preserve section ───────────────────────────────
    super().__attrs_post_init__()
    # NOTE: target_path default (os.getcwd()) and
    # working_dir = target_path are now handled in
    # TerminalInferencerBase.__attrs_post_init__.
```

Same comment block must be added to the analogous Phase 3 examples for ClaudeCodeCli (`_resolve_claude_command`), DevmateCli (`cd` script injection), RovoDev, MetamateCli.

### C6. Phase 0 — Add three tests covering the orchestrator scenario

The original 6 tests in §5 only cover standalone construction. Add:

7. `test_orchestrator_propagated_workspace_sets_child_working_dir` — RED today; GREEN after plan correction C1. Verifies BTA-spawned terminal child has `working_dir == per-iteration workspace.root`.

8. `test_orchestrator_propagated_workspace_does_not_override_explicit_target_path` — Verifies the new conditional logic: child constructed with `target_path="/explicit"` keeps working_dir=/explicit even when orchestrator propagates workspace.

9. `test_devmate_repo_path_kwarg_still_accepted_as_init_arg` — Verifies that after Phase 3, all 45+ test sites passing `repo_path=` still construct successfully (no `TypeError` on unexpected kwarg).

### C7. Documentation — Note in §13 Appendix

Add to the Appendix:
- `breakdown_then_aggregate_inferencer.py:1135` — BTA's `_configure_for_workspace` override. Plan must NOT modify this; revised Phase 1 logic in `inferencer_base.py` is sufficient.
- Devmate test count: 45+ sites passing `repo_path=` (corrected from earlier "15+" estimate).
- Confirm: all 4 diamond classes use the *legacy* `attr.attrs` API which defaults to `slots=False`. No slot-mismatch risk.

### C8. Phase 4 — Add explicit slot-compatibility test

Even though F9 was resolved (no actual mismatch), add a permanent regression test:

```python
def test_diamond_attrs_slots_consistency():
    """All four classes in the TerminalSessionInferencerBase MRO use slots=False.

    If a future contributor switches one to slots=True, this test fires
    so they catch the diamond-incompatibility before runtime.
    """
    classes_in_diamond = [
        InferencerBase,
        TemplatedInferencerBase,
        TemplatedStreamingInferencerBase,
        TerminalInferencerBase,
        TerminalSessionInferencerBase,
    ]
    for cls in classes_in_diamond:
        assert not hasattr(cls, "__slots__") or cls.__slots__ == (), (
            f"{cls.__name__} has non-empty __slots__ — diamond inheritance "
            "with mixed slots will break MRO-based field resolution."
        )
```

---

## D. Honest re-assessment of plan readiness

| Phase | Pre-audit verdict | Post-audit verdict | Blocker count |
|-------|-------------------|---------------------|---------------|
| 0 | Ready to write | Needs +3 tests (C6) | 0 |
| 1 | Ready to ship | **Redesign required** (C1) | 1 |
| 1.5 | Ready to ship | Minor corrections (C3, C4) | 0 |
| 2a | Ready to design | Ready to design | 0 |
| 2b | Ready to design | Add slot test (C8) | 0 |
| 3 | Ready to design | **Devmate redesign required** (C2); doc-correct Kiro/Claude/RovoDev/Meta (C5) | 1 |
| 4 | Ready to design | Ready to design | 0 |

**Net:** the original plan was ~85% correct. The 15% that's wrong is concentrated in two places that would cause real production bugs (Phase 1 deletion and Phase 3 Devmate property). Both are now spec'd to be fixed in this audit document.

**Implementation may proceed** once corrections C1–C8 are applied to the main plan or carried alongside it as binding amendments.

---

## E. Honest meta-assessment

Three things this audit got right:
1. Catching the load-bearing nature of the clobber (would have caused `NotADirectoryError` in production for every BTA-style orchestrator).
2. Catching the attrs-property collision on Devmate (would have caused `TypeError` on every existing test site).
3. Catching the 3× test-site count error on Devmate (the plan estimated 15; reality is 45+).

Two things this audit *initially* got wrong:
1. Subagent F6 ("post-init ordering bug") — overcautious; on re-analysis, standalone construction is fine.
2. Subagent F9 ("@attrs slots mismatch") — false alarm; legacy `attr.attrs` API defaults to `slots=False`.

One thing the audit could not complete in the available iterations:
- F10/F11 (full audit of RovoDev `working_dir` uses + ClaudeCodeCli `_resolve_claude_command` interactions). These need a follow-up audit before Phase 3 lands. Tracking these as TODOs in the corrections.

---

## F. Recommended next action

1. **Apply corrections C1–C8 to the main plan** (or accept this audit as a binding addendum and reference it from the main plan).
2. **Run the F10/F11 follow-up audit** before Phase 3 implementation.
3. **Then** proceed with Phase 0 → Phase 1.5 → Phase 1 (revised) → Phase 2 → Phase 3 → Phase 4.

---

*End of audit. Reviewers: please challenge findings F1, F2, F4 (the BLOCKING issues) most carefully — if you can refute any of them, the corresponding plan correction can be reverted.*