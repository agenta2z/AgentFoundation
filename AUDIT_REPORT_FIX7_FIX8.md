# Audit Report: Fix #7 (Symlink Hardening) & Fix #8 (Snapshot-at-Phase-Time)

**File Audited:** `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

**Audit Date:** 2026-05-10

---

## Fix #7: Audit Symlink Hardening

### Property Checklist

#### 1. Cross-MFDual Leakage Detection ❌
**Requirement:** Detect when target lives outside `self._workspace.root` and emit `<phase>.LEAKAGE.diagnostic.txt` sibling file.

**Finding:**
- Lines 764-771: Leakage detection logic EXISTS
- **BUT:** Only logs an ERROR message via `logger.error()` — does NOT emit diagnostic file
- Current code:
  ```python
  my_root = str(self._workspace.root).rstrip("/") + "/"
  if not target.startswith(my_root):
      logger.error(
          "Audit: cross-worker leakage at round_%02d/%s: "
          "target %s outside %s",
          round_idx, phase, target, self._workspace.root,
      )
  ```
- **Gap:** No `<phase>.LEAKAGE.diagnostic.txt` file is created
- **Impact:** Silent logging only; diagnostics not persisted to disk for easy audit trail

---

#### 2. Within-Worker Role Aliasing Detection ❌
**Requirement:** Detect when two phases in same `round_NN/` point to same target with different phase names, emit `<phase>.ALIASING.diagnostic.txt`.

**Finding:**
- Lines 758-781: No aliasing detection logic present
- Method does not track or compare symlink targets across phases within a round
- **Gap:** Complete absence of aliasing detection
- **Impact:** Cannot identify when a role switch maps multiple phase names to identical workspace roots

---

#### 3. Duplicate-Phase-Overwrite Detection ❌
**Requirement:** Detect when same phase symlinks twice in same round, emit `<phase>.OVERWRITE.diagnostic.txt`.

**Finding:**
- Lines 773-774: Only unlinks and replaces existing symlink
  ```python
  if os.path.islink(link_path):
      os.unlink(link_path)
  ```
- No detection of whether a symlink already existed before this call
- No diagnostic file emitted on overwrite
- **Gap:** Silently overwrites; no persistence of overwrite history
- **Impact:** Audit trail loses information about which phase was overwritten

---

#### 4. Symlink Preservation (Backward Compat) ✅
**Requirement:** Symlink itself remains (diagnostics are SIBLING files, not replacements).

**Finding:**
- Lines 775-776: Symlink creation proceeds normally
  ```python
  try:
      os.symlink(target, link_path, target_is_directory=True)
  ```
- Diagnostics (if they existed) would be sibling files (`<phase>.LEAKAGE.diagnostic.txt`, etc.)
- Symlink is NOT replaced or removed by diagnostic logic
- **Status:** ✅ COMPLIANT — symlink remains intact

---

#### 5. Diagnostics Written Even on os.symlink Failure ❌
**Requirement:** Diagnostics persist even when `os.symlink` raises (Windows pointer fallback scenario).

**Finding:**
- Lines 775-780: Fallback to pointer file exists
  ```python
  try:
      os.symlink(target, link_path, target_is_directory=True)
  except (OSError, NotImplementedError):
      pointer = os.path.join(nav_dir, f"{phase}.pointer.txt")
      with open(pointer, "w") as f:
          f.write(f"# Workspace pointer\n# Target: {target}\n")
  ```
- However, leakage/aliasing/overwrite detection (lines 764-771) happens BEFORE the symlink attempt
- If symlink fails and pointer is written instead, the diagnostics from lines 764-771 are still only logs
- **Gap:** Diagnostic files are never written anywhere in the code (not in success path, not in exception path)
- **Impact:** Diagnostics exist only as logger output, not as persistent files

---

## Fix #8: Snapshot-at-Phase-Time Semantics

### Property Checklist

#### 1. Accept `workspace_root_at_phase` Optional Kwarg ✅
**Requirement:** `_record_round_audit` accepts `workspace_root_at_phase: Optional[str] = None` kwarg.

**Finding:**
- Line 712-713: Signature includes the kwarg
  ```python
  def _record_round_audit(self, round_idx, phase, inferencer, extra=None,
                          workspace_root_at_phase=None):
  ```
- **Status:** ✅ COMPLIANT

---

#### 2. Use Snapshot Instead of Live Read ✅
**Requirement:** When provided, use `workspace_root_at_phase` INSTEAD OF `inferencer._workspace.root` for symlink target.

**Finding:**
- Line 740-741: Snapshot-or-fallback logic implemented
  ```python
  # Fix #8: use snapshot if provided, else live read
  target = workspace_root_at_phase or str(inferencer._workspace.root)
  ```
- When `workspace_root_at_phase` is provided, it is used as `target`
- Fallback to live read only when `workspace_root_at_phase` is None
- **Status:** ✅ COMPLIANT — snapshot takes precedence over live value

---

#### 3. All 4 Call Sites Updated to Capture Workspace Eagerly ✅
**Requirement:** Call sites (_step_propose_impl, _step_review_impl, _step_fix_impl) capture workspace BEFORE ainfer() and pass via workspace_root_at_phase=.

**Finding:**

**3a. _step_propose_impl (Lines 1082-1086, 1119-1122):**
```python
_propose_ws_snapshot = (
    str(self.base_inferencer._workspace.root)
    if getattr(self.base_inferencer, "_workspace", None) is not None
    else None
)
```
- Captured BEFORE `await self.base_inferencer.ainfer()` at line 1087
- Passed to audit at lines 1119-1122:
  ```python
  self._record_round_audit(
      state["total_iterations"] + 1, "propose", self.base_inferencer,
      workspace_root_at_phase=_propose_ws_snapshot,
  )
  ```
- **Status:** ✅ COMPLIANT

**3b. _step_review_impl (Lines 1238-1242, 1309-1312):**
```python
_review_ws_snapshot = (
    str(self.review_inferencer._workspace.root)
    if getattr(self.review_inferencer, "_workspace", None) is not None
    else None
)
```
- Captured BEFORE `await self.review_inferencer.ainfer()` at lines 1247-1258
- Passed to audit at lines 1309-1312:
  ```python
  self._record_round_audit(
      total_iters, "review", self.review_inferencer,
      workspace_root_at_phase=_review_ws_snapshot,
  )
  ```
- **Status:** ✅ COMPLIANT

**3c. _step_fix_impl (Lines 1402-1406, 1460-1463):**
```python
_fix_ws_snapshot = (
    str(self.fixer_inferencer._workspace.root)
    if getattr(self.fixer_inferencer, "_workspace", None) is not None
    else None
)
```
- Captured BEFORE `await self.fixer_inferencer.ainfer()` at lines 1407-1422
- Passed to audit at lines 1460-1463:
  ```python
  self._record_round_audit(
      total_iters, "fix", self.fixer_inferencer,
      workspace_root_at_phase=_fix_ws_snapshot,
  )
  ```
- **Status:** ✅ COMPLIANT

**3d. _select_reviewer_and_fixer:**
- **Finding:** Method not found in the file
- **Grep search:** Only 3 call sites exist (propose, review, fix) — no call to audit in _select_reviewer_and_fixer
- **Status:** ⚠️ N/A — method does not exist or does not call _record_round_audit

---

#### 4. Tests for Snapshot Semantics ❌
**Requirement:** Tests exist for snapshot semantics, e.g., `test_audit_uses_snapshot_not_live_value` where role mutation between phase and audit still preserves snapshot.

**Finding:**
- Searched test files: `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_dual_inferencer/test_mfdual_resume.py`
- **Existing tests in Tier4_StateRestorationTest:**
  - `test_round_audit_symlinks_recreated_on_resume()` (line 275): Only checks method exists, no semantic verification
  - `test_round_log_jsonl_consistent_after_resume()` (line 285): Verifies "round_log" in source, not snapshot behavior
  - No test named `test_audit_uses_snapshot_not_live_value`
  - No test that mutates workspace between phase execution and audit recording
  - No test that verifies symlink points to snapshot, not live post-mutation value

- **Gap:** Snapshot semantics NOT tested
- **Impact:** Regression risk if workspace mutation logic changes

---

## Summary

### Fix #7 Status: ❌ INCOMPLETE
| Property | Status | Notes |
|----------|--------|-------|
| Cross-MFDual Leakage Detection | ❌ | Logs only, no `.LEAKAGE.diagnostic.txt` file created |
| Within-Worker Aliasing Detection | ❌ | Zero implementation; no tracking across phases |
| Duplicate-Phase Overwrite Detection | ❌ | Silent overwrite, no `.OVERWRITE.diagnostic.txt` file |
| Symlink Preservation | ✅ | Symlink remains; diagnostics would be sibling files |
| Diagnostics on Exception | ❌ | No diagnostic files created anywhere in code |

**Critical Gaps:**
1. **No diagnostic files are persisted to disk** — all detection logic results in logger output only
2. **Aliasing detection completely missing**
3. **Overwrite detection completely missing**
4. **Leakage logging is incomplete** — error log only, no file artifact

---

### Fix #8 Status: ✅ MOSTLY COMPLETE
| Property | Status | Notes |
|----------|--------|-------|
| Accept `workspace_root_at_phase` kwarg | ✅ | Signature correct |
| Use snapshot over live | ✅ | Line 741: `workspace_root_at_phase or str(inferencer._workspace.root)` |
| Propose call site updated | ✅ | Lines 1082-1122 |
| Review call site updated | ✅ | Lines 1238-1312 |
| Fix call site updated | ✅ | Lines 1402-1463 |
| _select_reviewer_and_fixer call site | N/A | Method not found in codebase |
| Snapshot semantics tests | ❌ | No test for snapshot-vs-live-value verification |

**Critical Gaps:**
1. **No regression test** for snapshot semantics — vulnerable to future mutations

---

## Recommendations

### For Fix #7:
1. **Implement diagnostic file creation** for LEAKAGE, ALIASING, and OVERWRITE conditions
   - File format: `<phase>.<CONDITION>.diagnostic.txt`
   - Content: Detailed explanation + metadata
   - Location: Same `nav_dir` as symlink

2. **Implement aliasing detection** by tracking symlink targets within each round
   - Before creating symlink, check if any other phase in `round_NN/` targets same destination
   - If yes, emit `<phase>.ALIASING.diagnostic.txt`

3. **Implement overwrite detection** by checking pre-existence of symlink
   - If `os.path.islink(link_path)` and it points to a DIFFERENT target, emit diagnostic

4. **Move leakage/aliasing/overwrite detection INTO the exception handler**
   - Ensure diagnostics are written even if symlink() fails

### For Fix #8:
1. **Add integration test** `test_audit_uses_snapshot_not_live_value`:
   - Capture workspace snapshot before phase
   - Mutate inferencer._workspace.root during phase
   - Verify audit symlink points to snapshot, not mutated value
   - Validates post-mutation rebinding does not drift audit

2. **Consider adding a `_select_reviewer_and_fixer` call** if this method exists elsewhere or should exist

---

## Code Quality Notes

✅ **Good:** Fail-safe design (lines 781-782) prevents audit from crashing inference
✅ **Good:** Pointer file fallback (lines 778-780) handles Windows/OSError scenarios
✅ **Good:** Backward compatibility preserved (None default for workspace_root_at_phase)

⚠️ **Risk:** Diagnostic gaps mean production deployments won't have full audit trail for MFDual leakage/aliasing/overwrite scenarios

