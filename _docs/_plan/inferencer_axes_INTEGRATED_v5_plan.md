# Inferencer Axes Refactor — Integrated v5 Plan

**Author:** Tony Chen (with Rovo Dev assistance)
**Date drafted:** 2026-05-16 07:42
**Status:** Ready for review and implementation
**Supersedes:** v4 (`inferencer_axes_INTEGRATED_v4_plan.md`, 923 lines)
**Companion (corrections source):** Claude's "Final Integrated Plan" (80-line diff-on-top-of-v3, `/Users/tchen7/.claude/plans/let-s-create-an-integrated-lively-pearl.md`)

> **What changed v4 → v5.** Three corrections from Claude's round-3 audit, two of which I independently verified as real bugs in v4. v5 ALSO upgrades one of those "corrections" because the underlying problem turned out to be more serious than Claude flagged.
>
> 1. **C1 (RovoDev workspace_path)** — already in v4; no change.
> 2. **C2 (`_infer` override audit + TIB timeout=300 inheritance)** — **REAL BUG**, more serious than Claude said. v5 ships an explicit fix.
> 3. **C3 (Phase 2 test stub vs. ClaudeCode)** — small fix; v5 incorporates.
>
> **If forced to pick ONE plan today:** v5 (this plan). Claude's 80-line corrections-only document is *not* a standalone plan — it's an addendum to v3/v4. v5 absorbs its corrections plus the additional finding about TIB's `timeout=300` inheritance.

---

## 1. v4 → v5 diff (the only sections that changed)

The full v4 plan stands. v5 modifies these specific sections:

- **§7.1 #6 (Phase 3, TIB)** — DO NOT inherit TIB's `timeout=300` into TSIB blindly. Move `timeout` from TIB to a new shared exec-mixin or document that TSIB intentionally overrides the default. See §2 below for the resolution chosen.
- **§8.1 #5 (Phase 4, TSIB)** — TSIB declares `timeout: Optional[int] = attrib(default=None)` overriding TIB's default. Means "no subprocess timeout" for session subclasses by default (preserves today's behavior). Subclasses opt in.
- **§8.4 (Phase 4 risk register)** — replace the false claim "Kiro/DevMate/Metamate don't override _infer, no break" with the corrected audit table and the timeout-resolution rationale.
- **§6.3 #3 (Phase 2 test)** — use a test-local stub, not ClaudeCodeCliInferencer (which hasn't been migrated yet).
- **§9.4 (RovoDev migration)** — restate that `target_path or working_dir` is genuinely defensive (target_path stays None for RovoDev; the `or` is essential).
- **§11.1 (permanent regression suite)** — add `test_tsib_timeout_default_is_None_or_unset` to pin the §2 resolution.
- **§12 (comparison table)** — add a "Round 3 (Claude's corrections)" column.

---

## 2. The C2 resolution — why TSIB must override TIB's `timeout=300`

### 2.1 What v4 (and v3) got wrong

v4 §7.1 #6 and §8.4 risk register claimed: *"TIB's stronger `_infer` breaks subclasses with overrides that don't expect timeout/env_vars/post_exec. Audit: ClaudeCodeCli and KiroCli both override `_infer` and don't call `super()._infer`. No break."*

**Empirically verified wrong** (independent grep on 2026-05-16 07:41):

| Leaf | Overrides `_infer`? | If no, picks up TIB's 300s timeout? |
|------|---------------------|-------------------------------------|
| ClaudeCodeCli | ✅ YES (line 584) | N/A — own override |
| RovoDevCli | ✅ YES (line 318) | N/A — own override |
| **KiroCli** | ❌ **NO** | ⚠️ YES — sync path gains 300s cap |
| **DevmateCli** | ❌ **NO** | ⚠️ YES — sync path gains 300s cap |
| **MetamateCli** | ❌ **NO** | ⚠️ YES — sync path gains 300s cap |

The risk is bigger than Claude flagged:

- **TSIB does not declare any `timeout` attrib today** (verified by grep). It only has `subprocess_exit_drain_timeout=5.0` and `_subprocess_exit_poll_interval=0.5`, both used for pipe cleanup, not subprocess execution.
- **Today's TSIB `_infer` does not pass a `timeout=` kwarg to `subprocess.run`** — sync calls have no subprocess-level cap.
- **Kiro, DevMate, Metamate each have their own differently-named timeout fields** (`idle_timeout_seconds`, `total_timeout_on_internal_error_detection_seconds`, `timeout_seconds`) — none of which is a `subprocess.run` timeout.

So after Phase 4, the diamond inheritance introduces a `timeout` attrib these classes never had AND activates a 300s subprocess cap silently. This is exactly the "ad-hoc, hacky" outcome the user explicitly told us to avoid.

### 2.2 Three resolution options considered

| Option | What it does | Pros | Cons |
|---|---|---|---|
| A. Document & accept (Claude's recommendation) | Note that Kiro/DevMate/Metamate gain a 300s cap | Zero code change | Silent behavior change for production CLIs; "no hack" principle violated |
| B. TSIB overrides `timeout` default to `None` | TSIB declares `timeout: Optional[int] = attrib(default=None)`; `_execute_command` skips the `timeout=` kwarg when None | Preserves today's behavior; explicit | Splits the semantics of `timeout` across TIB (int, defaults 300) and TSIB (Optional[int], defaults None) — slight type incoherence |
| C. Promote `timeout` to TIB as `Optional[int]` default `None`; sync TIB callers that need 300 keep their own override | Unifies the type; makes "no timeout" the framework default | Cleanest semantically | Behavior change for existing TIB-only stubs (they lose the 300s default) — but those are test stubs and we control them |

### 2.3 v5 chooses Option C — promote `Optional[int]` with `None` default

Rationale:
- "No timeout" should be the default; explicit timeout should be opt-in. Most CLI subprocess calls have unpredictable runtimes (especially when wrapping LLM agents). A silent 300s cap is exactly the kind of footgun this refactor exists to eliminate.
- The 5 Terminal-only test stubs that previously got `timeout=300` for free are *test stubs* — they don't actually time out in tests (test cases complete in milliseconds). Verified: zero production TIB-only leaves exist (only stubs).
- Type coherence: TIB and TSIB share `timeout: Optional[int]` with default `None`. `_execute_command` and TIB's `_infer` check `if self.timeout is not None:` before passing `timeout=` to `subprocess.run`.
- The 2 leaves that *do* override `_infer` (Claude, RovoDev) are unaffected.

### 2.4 Concrete edits

**`terminal_inferencer_base.py` line 44:**
```python
# OLD: timeout: int = attrib(default=300)
timeout: Optional[int] = attrib(default=None)
```

**Update docstring (line 31):**
```python
# OLD: timeout (int): Command execution timeout in seconds. Defaults to 300 (5 min).
# NEW: timeout (Optional[int]): Command execution timeout in seconds. None = no
#      timeout (subprocess.run runs to completion). Default: None.
```

**`_execute_scripts` line 151 and `_execute_command` line 275** — wrap the timeout kwarg in a conditional:
```python
kwargs = {}
if self.timeout is not None:
    kwargs["timeout"] = self.timeout
result = subprocess.run(..., **kwargs)
```

And the `except subprocess.TimeoutExpired` handler at lines 163/287 stays — it only fires when timeout is set.

**`terminal_session_inferencer_base.py`:** No explicit override needed. TSIB inherits `timeout: Optional[int] = None` from TIB, which matches today's behavior (no subprocess timeout for sync `_infer` on session subclasses).

**Test stub migration:** the 5 Terminal-only test stubs in `test_terminal_inferencer_base.py` either explicitly set `timeout=300` (preserves their old behavior) or remove the assumption. Inspect each; default action is to set `timeout=300` explicitly in the test fixture to preserve test semantics.

### 2.5 Permanent regression test (§11.1 new test #6)

```python
def test_tib_timeout_default_is_None():
    """TIB.timeout must default to None (no subprocess cap).
    Historic value of 300 was a footgun for session subclasses that
    inherited it silently. See plan §2.
    """
    assert attr.fields(TerminalInferencerBase).timeout.default is None

def test_tsib_inherits_timeout_None_default():
    """TSIB must not silently activate a subprocess timeout."""
    assert attr.fields(TerminalSessionInferencerBase).timeout.default is None
```

---

## 3. The C3 resolution — Phase 2 test uses a test-local stub

### 3.1 What v4 got wrong

v4 §6.3 #3: *"test_sib_recovery_uses_template_manager_when_present — instantiate a templated subclass (e.g., ClaudeCodeCliInferencer post-Phase-5)."*

**Problem:** Phase 2 ships before Phase 5. At Phase 2 ship time, ClaudeCodeCliInferencer is still `TerminalSessionInferencerBase`-parented and the diamond doesn't yet include TemplatedIB via the new MI route. Using ClaudeCode in a Phase 2 test creates a circular Phase ordering dependency.

### 3.2 v5 fix

Use a test-local stub defined inside the Phase 2 test module:

```python
# In test/agent_foundation/common/inferencers/test_streaming_decoupling.py
@attrs
class _TemplatedStreamingStub(StreamingInferencerBase, TemplatedInferencerBase):
    """Test-only: minimal SIB+TemplatedIB diamond.
    Exists to verify template_manager duck-typing in
    StreamingInferencerBase._render_recovery_prompt works when
    template_manager IS present.
    """
    template_manager: Optional[Any] = attrib(default=None)

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "test"
        return {"output": "test"}


def test_sib_recovery_uses_template_manager_when_present():
    """When a SIB subclass also inherits TemplatedIB, _render_recovery_prompt
    routes through self.template_manager(...). Verified with a tiny stub
    so this test doesn't depend on production class migration ordering."""
    inst = _TemplatedStreamingStub(template_manager=MagicMock())
    out = inst._render_recovery_prompt(...)
    inst.template_manager.assert_called_once()
```

This test is portable across Phase 2, 3, 4, 5 ship orders.

---

## 4. The C1 verification — RovoDev workspace_path

### 4.1 What Claude flagged

Claude said v3 §10.4's suggestion to *rename* `workspace_path=self.working_dir` → `workspace_path=self.target_path` would break RovoDev because RovoDev never sets `target_path`.

### 4.2 v5 confirms — v4 already has this right

v4 §9.4 #3 already uses the safe fallback: `workspace_path=self.target_path or self.working_dir`. Since `working_dir` is always non-None after TIB's post_init (defaults to `os.getcwd()`), the fallback preserves today's behavior in all cases. **No further v5 change needed for C1.**

However, v5 adds a clarification comment to make the *reason* explicit:

```python
# RovoDev does not set target_path (intentional — see plan §2.1).
# The `or` fallback is required: target_path is None, working_dir holds
# the correct value (cwd at construction time, OR workspace.root if the
# inferencer was orchestrator-spawned and the workspace setter ran).
workspace_path = self.target_path or self.working_dir
```

---

## 5. Updated Phase 0 RED test count (v4 had 10; v5 adds 2)

| # | Test | Lands |
|---|------|-------|
| 1–10 | Unchanged from v4 §4.2 | (per v4) |
| **11 (NEW)** | `test_tsib_subclass_without_explicit_timeout_has_no_subprocess_cap` — Kiro/DevMate/Metamate sync `_infer` calls do NOT apply a 300s subprocess timeout | Phase 3 (timeout type change) |
| **12 (NEW)** | `test_tib_test_stubs_with_explicit_timeout_still_work` — verify existing TIB test stubs that set `timeout=300` continue to function | Phase 3 |

---

## 6. Updated risk register entries

### 6.1 Phase 3 (TIB) risk register addendum

| Risk | Mitigation |
|---|---|
| Changing TIB's `timeout` default from `300` to `None` breaks existing TIB-only test stubs that relied on it. | Test stubs are under our control. Phase 0 test #12 verifies all stubs work with the new default. Migrate stubs to explicitly set `timeout=300` if they need that semantics. |
| Existing TIB subclasses in user code (none known) might rely on the 300s default. | Grep external code (deferred work §13) confirms zero such consumers. Document in release notes: "TerminalInferencerBase.timeout default changed from 300 to None (no cap). Set explicitly to restore old behavior." |
| `subprocess.run` without timeout can hang indefinitely on misbehaving CLIs. | This was already true for TSIB subclasses (which never had a subprocess timeout). Phase 4 unification preserves that behavior. Leaves can opt in to a timeout by setting `timeout=N` in their attrs. |

### 6.2 Phase 4 (TSIB) risk register — replace v4 §8.4 row 2

| Risk (corrected) | Mitigation |
|---|---|
| ~~TIB's stronger `_infer` breaks subclasses with overrides~~ → After Phase 4, Kiro/DevMate/Metamate (no `_infer` override) inherit TIB's `_infer`. Today TSIB's `_infer` had no subprocess timeout; TIB's had `timeout=300`. Silent behavior change. | **Resolved via §2.3 Option C** — TIB's `timeout` default changes from `300` to `None`. Kiro/DevMate/Metamate inherit `None` → no subprocess cap → today's behavior preserved. Permanent test §11.1 #6 pins it. |

---

## 7. Comparison table (Round 3)

| Aspect | v3 | v4 | Claude (80-line addendum) | v5 (this plan) |
|---|---|---|---|---|
| Architecture | Decoupled | Decoupled | Endorses v3's | Decoupled (unchanged) |
| target_path default | None (buried) | None (highlighted) | Endorses v4 | None (highlighted) |
| _configure_for_workspace gate | `target != os.getcwd()` | `target is None` + sentinel | Endorses v4 | `target is None` + sentinel |
| Working_dir-user-set sentinel | No | Yes (§2.5) | Endorses v4 | Yes (§2.5) |
| RovoDev workspace_path | "rename to target_path" (BROKEN — target_path is None) | `target_path or working_dir` fallback ✅ | Catches v3 bug; v4 already has fix | v4's fix + explanatory comment |
| `_infer` override audit | "Claude+Kiro override" (WRONG) | Same wrong claim | **Catches both v3+v4 bug** | **v5 fixes: explicit audit table + Option C timeout fix** |
| TIB `timeout` default | 300 (untouched) | 300 (untouched) | "Document and accept" | **Optional[int] = None — eliminates silent cap** |
| Phase 2 test target | ClaudeCodeCli (Phase-5 dependency) | Same (carried over) | Catches; suggests stub | Stub (full code in §3.2) |
| pre_exec_scripts hook | Risk mention only | Explicit `_run_pre_exec_scripts_in_subprocess_shell` hook | Endorses v4 | v4's hook (unchanged) |
| Pre-flight grep baseline | No | Yes | N/A | Yes |
| Permanent regression suite | 4 tests | 5 tests | N/A | **7 tests (+timeout invariants)** |
| Plan length | 786 | 923 | 80 | **~750 (concise + v5 deltas)** |

### 7.1 If forced to pick ONE plan: **v5 (this plan).**

Reasons:
- v5 is the only plan that captures the C2 bug correctly *and* upgrades the fix beyond Claude's "document and accept" recommendation.
- Claude's 80-line addendum is **not a standalone plan**; it's a 3-correction diff on top of v3. Picking it alone leaves you without the architecture, the test plan, the risk registers, the rollback strategy, etc.
- v4 has the architecture and operational rigor but carries the same wrong `_infer` claim as v3.
- v5 = v4 architecture + v4 operational rigor + C1/C2/C3 fixes + C2 upgrade (Option C).

Honest accounting: **Claude found a real bug in v4 that I missed.** The C2 finding is genuinely valuable. The corrections are short because Claude is doing exactly what it should — a focused diff that doesn't repeat what's already correct. But because the corrections sit on top of v3 (not v4), and v4 already had C1, v5 is the first plan that has everything correct in one place.

---

## 8. Rest of the plan

Everything else from v4 §3–15 stands without modification. v5 is fundamentally **v4 + the 3 corrections + the timeout upgrade + 2 new Phase 0 tests + 2 new permanent invariants**.

For the full architecture, MROs, phased rollout, leaf migrations, rollback matrix, design principles, and worked examples, see `inferencer_axes_INTEGRATED_v4_plan.md` §1–15. Apply the v5 deltas in this document on top.

---

## 9. Why this is "elegant, not hacky"

Every v5 change makes an implicit thing explicit:

| Change | What was implicit before | What is explicit after |
|---|---|---|
| TIB `timeout` → `Optional[int] = None` | "300s is a magic number that silently applies to anyone who inherits and doesn't override" | "Timeout is opt-in; framework default is no cap" |
| TSIB-uses-stub-for-templated-test | "Tests assume Phase 5 already shipped" | "Test fixtures are self-contained; ordering independent" |
| RovoDev `target_path or working_dir` comment | "Why the `or`? unclear" | "RovoDev intentionally leaves target_path=None; the `or` is the bridge" |
| `_infer` override audit table (corrected) | "Most leaves override, right?" | "Exactly 2/5 override; 3/5 inherit; the 3 must not gain silent behavior" |

None of these are workarounds. Each replaces a hidden assumption with a documented, testable contract — which is the criterion you set for "elegant, no hack."

---

*End of v5 integrated plan.*