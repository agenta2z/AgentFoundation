# RovoDevCliInferencer — Non-Legacy Mode Support Plan

## 1. Background & Motivation

The current `RovoDevCliInferencer` hardcodes `"legacy"` in `construct_command()`:
```python
parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]
```

This means it always invokes `acli rovodev legacy <message>`, which is the non-TUI
programmatic mode. The Rovo Dev CLI has since evolved, and the **default mode**
(`acli rovodev <message>`) is now the primary interface — "legacy" may eventually
be deprecated.

**Goal**: Add an `enable_legacy` boolean (default `True` for backward compatibility)
that also enables the non-legacy (`acli rovodev <message>`) path.

---

## 2. Investigation: CLI Differences Between Legacy & Non-Legacy

### 2.1 Command Structure

| Aspect | Legacy Mode | Non-Legacy (TUI/Default) Mode |
|--------|-------------|-------------------------------|
| **Invocation** | `acli rovodev legacy <message>` | `acli rovodev <message>` |
| **Description** | "Run the legacy (non-TUI) Rovo Dev CLI" | "Run the Rovo Dev TUI application" |
| **Behavior with message** | Runs headlessly, exits on completion | Runs headlessly when message provided (exits), interactive otherwise |

### 2.2 Flag Comparison

| Flag | Legacy | Non-Legacy | Notes |
|------|--------|------------|-------|
| `--config-file` | ✅ | ✅ | Same |
| `--verbose` | ✅ | ✅ `--verbose/--no-verbose` | Non-legacy uses explicit toggle |
| `--restore` | ✅ `TEXT` | ✅ `TEXT` | Same behavior |
| `--resume` | ❌ | ✅ (alias for `--restore`) | Non-legacy adds alias |
| `--yolo` | ✅ | ✅ (alias for `--disable-permission-checks`) | Different primary name |
| `--disable-permission-checks` | ❌ | ✅ (primary name for yolo) | Non-legacy primary |
| `--xid` | ✅ | ❌ | Legacy only |
| `--jira` | ✅ | ❌ | Legacy only |
| `--enable-deep-plan` | ✅ | ❌ | Legacy only |
| `--output-schema` | ✅ | ✅ | Same |
| `--output-file` | ✅ | ❌ | **Legacy only — critical for clean output capture** |
| `--config-override` | ❌ | ✅ | Non-legacy only, JSON string |
| `--worktree` | ❌ | ✅ | Non-legacy only, git worktree support |
| `--interactive/-i` | ❌ | ✅ | Force interactive mode |
| `--web/--no-web` | ❌ | ✅ | Web UI mode |
| `--port` | ❌ | ✅ | Web UI port |
| `--agent-mode` | ✅ (inferred) | ❌ | Legacy only |

### 2.3 Critical Differences & Impact

1. **`--output-file` is legacy-only**: This is the **most impactful difference**. The current
   implementation relies heavily on `--output-file` for clean output capture (bypassing Rich
   TUI formatting, ANSI codes, preserving XML tags). Non-legacy mode does NOT have this flag.
   - **Mitigation**: In non-legacy mode, we must fall back to ANSI-stripped stdout parsing,
     or use `--output-schema` with a passthrough schema to get clean output.

2. **`--xid`, `--jira`, `--enable-deep-plan`, `--agent-mode` are legacy-only**: These flags
   must be silently skipped (with a warning log) in non-legacy mode, OR we find the non-legacy
   equivalents (possibly via `--config-override`).

3. **Session restore**: Both modes support `--restore`. Non-legacy also accepts `--resume` as
   an alias. No change needed for session management — `_build_session_args()` already uses
   `--restore`.

4. **`--yolo` works in both**: Non-legacy adds `--disable-permission-checks` as primary name
   but `--yolo` is still accepted as an alias. No change needed.

5. **Non-legacy exclusive features**: `--worktree`, `--config-override`, `--interactive/-i`,
   `--web` could be exposed as new attributes but are NOT required for MVP.

---

## 3. Design

### 3.1 New Attribute

```python
enable_legacy: bool = attrib(default=True)
```

- `True` (default): Current behavior, invokes `acli rovodev legacy <message>`
- `False`: Invokes `acli rovodev <message>` (non-legacy/TUI mode)

### 3.2 New Optional Attributes (non-legacy exclusive)

```python
config_override: Optional[str] = attrib(default=None)  # JSON string for --config-override
worktree: Optional[str] = attrib(default=None)          # Git worktree name for --worktree
```

These are only used when `enable_legacy=False`. If set when `enable_legacy=True`,
they are ignored with a debug log.

### 3.3 Changes to `construct_command()`

The core change is in how the command parts are assembled:

```python
def construct_command(self, inference_input, **kwargs):
    acli = self.acli_path
    if not acli:
        raise RovoDevNotFoundError()

    prompt = ...  # (unchanged extraction logic)

    if self.enable_legacy:
        parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]
    else:
        parts = [acli, ACLI_SUBCOMMAND, shlex.quote(prompt)]

    # Flags common to both modes:
    if self.yolo:
        parts.append("--yolo")  # works in both (alias in non-legacy)
    if self.config_file:
        parts.extend(["--config-file", self.config_file])
    if self.output_schema:
        parts.extend(["--output-schema", shlex.quote(self.output_schema)])

    # Session restore — works in both modes (same --restore flag)
    session_id = kwargs.get("session_id")
    is_resume = kwargs.get("resume", False)
    if is_resume:
        session_args = self._build_session_args(session_id or "", is_resume)
        if session_args:
            parts.append(session_args)

    if self.enable_legacy:
        # Legacy-only flags:
        out_path = kwargs.get("output_file") or self.output_file
        if not out_path and self.raw_output_to_file:
            out_path = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["_auto_output_file"] = out_path
        if out_path:
            parts.extend(["--output-file", out_path])

        if self.xid:
            parts.extend(["--xid", self.xid])
        if self.jira:
            parts.extend(["--jira", self.jira])
        if self.enable_deep_plan:
            parts.append("--enable-deep-plan")
        if self.agent_mode:
            parts.extend(["--agent-mode", self.agent_mode])
    else:
        # Non-legacy mode:
        # --output-file not available — warn if raw_output_to_file was expected
        if self.raw_output_to_file:
            logger.debug(
                "[%s] raw_output_to_file ignored in non-legacy mode "
                "(--output-file not available)", self.__class__.__name__
            )

        if self.config_override:
            parts.extend(["--config-override", shlex.quote(self.config_override)])
        if self.worktree:
            parts.extend(["--worktree", shlex.quote(self.worktree)])

        # Warn about legacy-only flags being set
        for flag_name, flag_val in [
            ("xid", self.xid), ("jira", self.jira),
            ("enable_deep_plan", self.enable_deep_plan),
            ("agent_mode", self.agent_mode),
        ]:
            if flag_val:
                logger.warning(
                    "[%s] '%s' is legacy-only and ignored in non-legacy mode",
                    self.__class__.__name__, flag_name,
                )

    if self.extra_cli_args:
        parts.extend(self.extra_cli_args)

    return " ".join(parts)
```

### 3.4 Changes to `_infer()` (Sync Execution)

In non-legacy mode, `--output-file` is not available. The `_infer()` method currently
generates a temp output file and reads from it. For non-legacy:

- Skip temp output file generation
- Rely on ANSI-stripped stdout (via existing `parse_output()` fallback path)
- `parse_output()` already handles this — when no output file exists, it falls back
  to `strip_ansi_codes(stdout).strip()`

```python
def _infer(self, inference_input, inference_config=None, **kwargs):
    auto_output_file = None
    if self.enable_legacy and not self.output_file and self.raw_output_to_file:
        auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
        kwargs["output_file"] = auto_output_file

    # ... rest unchanged ...
```

### 3.5 Changes to `ainfer()` (Async Execution)

Same pattern — skip temp file in non-legacy mode:

```python
async def ainfer(self, inference_input, inference_config=None, **kwargs):
    auto_output_file = None
    if self.enable_legacy and not self.output_file and self.raw_output_to_file:
        auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
        kwargs["output_file"] = auto_output_file
        _current_output_file.set(auto_output_file)
    # ... rest unchanged ...
```

### 3.6 Changes to `_yield_filter()` (Streaming)

No changes needed. The ANSI stripping already applies to all streaming output
regardless of mode.

### 3.7 Changes to `_build_session_args()`

No changes needed. Both modes accept `--restore [session_id]` with the same semantics.

### 3.8 Docstring & Class-Level Updates

- Update class docstring to document both modes
- Update `construct_command()` docstring
- Update the module docstring usage examples

### 3.9 Changes to `rovodev_anthropic_proxy.py`

The proxy creates `RovoDevCliInferencer`. It should accept an optional `--non-legacy`
flag (or `--enable-legacy/--disable-legacy`) to control the mode. The `_parse_args()`
function and `create_app()` need a new parameter.

---

## 4. Impact Analysis

### 4.1 Backward Compatibility

- **Default `enable_legacy=True`**: Zero behavior change for existing code
- **All existing tests**: Unaffected (they use legacy mode by default)
- **Anthropic proxy**: Defaults to legacy, opt-in non-legacy via CLI flag

### 4.2 Output Quality in Non-Legacy Mode

**Risk**: Without `--output-file`, non-legacy stdout may contain Rich TUI formatting
artifacts even with ANSI stripping. The quality of `parse_output()` results may degrade.

**Mitigations**:
1. The existing `strip_ansi_codes()` + `.strip()` should handle most cases
2. Non-legacy mode with a `<message>` argument runs non-interactively, so TUI
   formatting should be minimal
3. If needed, `--output-schema '{"type":"object","properties":{"response":{"type":"string"}}}'`
   could force structured JSON output

### 4.3 Feature Parity Gaps

These legacy-only features have no non-legacy equivalent:
- `--xid` (experience ID) — may be available via `--config-override`
- `--jira` (Jira ticket) — may be available via `--config-override`
- `--enable-deep-plan` — may be available via `--config-override`
- `--agent-mode` — may be available via `--config-override`
- `--output-file` — no equivalent, must use stdout

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `rovodev_cli_inferencer.py` | Add `enable_legacy`, `config_override`, `worktree` attrs; refactor `construct_command()`; guard temp file in `_infer()`/`ainfer()`; update docstrings |
| `rovodev_anthropic_proxy.py` | Add `--enable-legacy`/`--disable-legacy` CLI flag; pass to `create_app()` and `RovoDevCliInferencer` |
| `common.py` | No changes needed |
| `rovodev_serve_inferencer.py` | No changes needed (serve mode is independent) |

---

## 6. Implementation Steps

| Step | Task | Estimate |
|------|------|----------|
| 1 | Add `enable_legacy`, `config_override`, `worktree` attributes to `RovoDevCliInferencer` | 5 min |
| 2 | Refactor `construct_command()` with legacy/non-legacy branching | 15 min |
| 3 | Guard temp output file generation in `_infer()` on `enable_legacy` | 5 min |
| 4 | Guard temp output file generation in `ainfer()` on `enable_legacy` | 5 min |
| 5 | Update `rovodev_anthropic_proxy.py` `_parse_args()` and `create_app()` | 10 min |
| 6 | Update docstrings (module, class, construct_command) | 10 min |
| 7 | Manual verification with `acli rovodev <message>` | 10 min |

---

## 7. Testing Strategy

### 7.1 Unit Tests (No acli required)

```python
def test_construct_command_legacy_mode():
    """Default (legacy) should produce 'acli rovodev legacy <prompt>'."""
    inf = RovoDevCliInferencer(acli_path="/usr/bin/acli", working_dir="/tmp")
    cmd = inf.construct_command("hello")
    assert "rovodev legacy" in cmd
    assert "'hello'" in cmd

def test_construct_command_non_legacy_mode():
    """Non-legacy should produce 'acli rovodev <prompt>' without 'legacy'."""
    inf = RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir="/tmp", enable_legacy=False
    )
    cmd = inf.construct_command("hello")
    assert "rovodev" in cmd
    assert "legacy" not in cmd
    assert "'hello'" in cmd

def test_non_legacy_skips_legacy_only_flags():
    """Legacy-only flags (xid, jira, etc.) should NOT appear in non-legacy commands."""
    inf = RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir="/tmp",
        enable_legacy=False, xid="test-xid", jira="http://jira/PROJ-1",
        enable_deep_plan=True, agent_mode="ask",
    )
    cmd = inf.construct_command("hello")
    assert "--xid" not in cmd
    assert "--jira" not in cmd
    assert "--enable-deep-plan" not in cmd
    assert "--agent-mode" not in cmd

def test_non_legacy_includes_config_override():
    """Non-legacy exclusive flags should appear."""
    inf = RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir="/tmp",
        enable_legacy=False, config_override='{"agent":{"modelId":"opus"}}',
    )
    cmd = inf.construct_command("hello")
    assert "--config-override" in cmd

def test_non_legacy_skips_output_file():
    """Non-legacy mode should not use --output-file."""
    inf = RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir="/tmp",
        enable_legacy=False, raw_output_to_file=True,
    )
    cmd = inf.construct_command("hello")
    assert "--output-file" not in cmd

def test_legacy_includes_output_file():
    """Legacy mode with raw_output_to_file should use --output-file."""
    inf = RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir="/tmp",
        enable_legacy=True, raw_output_to_file=True,
    )
    cmd = inf.construct_command("hello")
    assert "--output-file" in cmd

def test_session_restore_works_both_modes():
    """--restore should appear in both legacy and non-legacy modes."""
    for legacy in [True, False]:
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir="/tmp", enable_legacy=legacy
        )
        cmd = inf.construct_command("hello", session_id="abc-123", resume=True)
        assert "--restore" in cmd
```

### 7.2 Integration Tests (Require acli authenticated)

```python
@pytest.mark.integration
def test_non_legacy_basic_inference():
    inf = RovoDevCliInferencer(enable_legacy=False)
    result = inf("What is 2+2?")
    assert result.success
    assert result.output

@pytest.mark.integration
def test_non_legacy_session_resume():
    inf = RovoDevCliInferencer(enable_legacy=False)
    r1 = inf.new_session("Remember: the secret is 42")
    r2 = inf("What is the secret?")
    assert "42" in r2.output
```

---

## 8. Open Questions

1. **Non-legacy stdout cleanliness**: When `acli rovodev <message>` runs with a message
   argument (non-interactive), is stdout clean text or does it still contain Rich TUI
   formatting? Need to verify empirically.
   - If TUI artifacts are present, we may need enhanced `strip_ansi_codes()` or a
     `--config-override` to disable Rich formatting.

2. **`--config-override` equivalents**: Can legacy-only flags (`--xid`, `--jira`,
   `--enable-deep-plan`, `--agent-mode`) be passed via `--config-override` in non-legacy mode?
   - If yes, we could auto-translate these attrs to config override JSON.

3. **Future deprecation**: Will `legacy` mode be removed? If so, what's the timeline?
   This affects whether `enable_legacy=True` should eventually flip to `False` as default.
