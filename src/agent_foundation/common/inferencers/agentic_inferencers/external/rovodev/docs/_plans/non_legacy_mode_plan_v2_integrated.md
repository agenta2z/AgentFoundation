# RovoDevCliInferencer — Non-Legacy Mode Plan (v2 Integrated)

> This plan integrates the best of two independent plans after critical review.
> It corrects factual errors found in both plans and resolves all disagreements.

## 1. Background

The `RovoDevCliInferencer` hardcodes `"legacy"` in `construct_command()`:
```python
parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]
```

**Goal**: Add `enable_legacy: bool = True` to support both legacy (`acli rovodev legacy`)
and non-legacy/TUI (`acli rovodev <message>`) modes.

---

## 2. Critical Review of Prior Plans

### 2.1 Errors Found in Other Agent's Plan

| Issue | Other Plan Says | Reality (Verified via `--help`) |
|-------|----------------|--------------------------------|
| **`--xid` in TUI** | "YES (hidden)" — kept in non-legacy | **WRONG** — `--xid` does NOT appear in `acli rovodev run --help`. It is legacy-only. Must be skipped + warned. |
| **`--output-schema` auto-injection** | Auto-inject `_TUI_OUTPUT_SCHEMA` to capture output as JSON | **Clever but risky** — adds implicit behavior. Users may be surprised by JSON-wrapped output. Also couples inferencer to a specific schema contract. See §3.4 for refined approach. |
| **`extract_trailing_json()` in `common.py`** | Parse trailing JSON from TUI stdout | **Fragile** — assumes JSON is always at the end of stdout, that no other JSON appears in output, and that `{`/`}` matching is reliable with nested content. Needs hardening. |

### 2.2 Errors Found in My Original Plan

| Issue | My Plan Says | Reality |
|-------|-------------|---------|
| **`--output-schema` in non-legacy** | Listed as "Same" | **Correct** — verified present in both modes |
| **`--verbose` in non-legacy** | Listed as "✅ `--verbose/--no-verbose`" — suggesting toggle only | **Correct** — non-legacy uses `--verbose/--no-verbose` toggle, legacy uses `--verbose` alone |
| **Missing `--output-schema` strategy** | Mentioned as mitigation in §4.2 but didn't design it | The other plan's `--output-schema` auto-injection is a genuinely good idea — just needs refinement |

### 2.3 Agreement Between Plans (Validated)

Both plans correctly agree on:
- `enable_legacy: bool = True` default for backward compat
- `config_override` as new non-legacy-only attribute
- `--yolo` works in both modes (alias in non-legacy)
- `--restore` works in both modes
- `--jira`, `--enable-deep-plan`, `--agent-mode` are legacy-only → skip + warn
- `--output-file` is legacy-only → critical challenge for output capture
- Guard temp file in `_infer()` / `ainfer()` on `enable_legacy`

---

## 3. Verified Flag Comparison (Definitive)

Based on actual `acli rovodev run --help` and `acli rovodev legacy --help` output:

| Flag | Legacy | Non-Legacy (TUI) | Action |
|------|--------|-------------------|--------|
| `--config-file` | ✅ | ✅ | Common — no branching |
| `--verbose` | ✅ (flag) | ✅ `--verbose/--no-verbose` | Common — `--verbose` works in both |
| `--restore` | ✅ | ✅ (`--restore/--resume`) | Common — `--restore` works in both |
| `--yolo` | ✅ | ✅ (`--yolo/--disable-permission-checks`) | Common — `--yolo` works in both |
| `--output-schema` | ✅ | ✅ | Common — works in both |
| `--xid` | ✅ | ❌ | **Legacy-only** — skip + warn |
| `--jira` | ✅ | ❌ | **Legacy-only** — skip + warn |
| `--enable-deep-plan` | ✅ | ❌ | **Legacy-only** — skip + warn |
| `--agent-mode` | ✅ | ❌ | **Legacy-only** — skip + warn |
| `--output-file` | ✅ | ❌ | **Legacy-only** — critical for output capture |
| `--config-override` | ❌ | ✅ | **Non-legacy only** — new attribute |
| `--worktree` | ❌ | ✅ | **Non-legacy only** — new attribute |
| `--interactive/-i` | ❌ | ✅ | Non-legacy only — not needed for programmatic use |
| `--web/--no-web` | ❌ | ✅ | Non-legacy only — not needed for programmatic use |
| `--port` | ❌ | ✅ | Non-legacy only — not needed for programmatic use |

---

## 4. Design (Integrated Best of Both Plans)

### 4.1 New Attributes

```python
enable_legacy: bool = attrib(default=True)
config_override: Optional[str] = attrib(default=None)  # --config-override (non-legacy only)
worktree: Optional[str] = attrib(default=None)          # --worktree (non-legacy only)
```

### 4.2 Internal Constant for Auto-Injected Schema

Adopted from the other plan (good idea), but refined:

```python
_NON_LEGACY_OUTPUT_SCHEMA = '{"type":"object","properties":{"result":{"type":"string"}}}'
```

This schema is auto-injected in non-legacy mode ONLY when:
1. `self.output_schema` is not set by the user, AND
2. `self.raw_output_to_file` is True (i.e., user wants clean output)

This avoids surprising users who don't care about output cleanliness.

### 4.3 `construct_command()` Changes

```python
def construct_command(self, inference_input, **kwargs):
    acli = self.acli_path
    if not acli:
        raise RovoDevNotFoundError()

    prompt = ...  # unchanged extraction logic

    # --- Command stem ---
    if self.enable_legacy:
        parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]
    else:
        parts = [acli, ACLI_SUBCOMMAND, shlex.quote(prompt)]

    # --- Common flags (both modes) ---
    if self.yolo:
        parts.append("--yolo")
    if self.config_file:
        parts.extend(["--config-file", self.config_file])
    if self.output_schema:
        parts.extend(["--output-schema", shlex.quote(self.output_schema)])

    # Session restore (both modes support --restore)
    session_id = kwargs.get("session_id")
    is_resume = kwargs.get("resume", False)
    if is_resume:
        session_args = self._build_session_args(session_id or "", is_resume)
        if session_args:
            parts.append(session_args)

    # --- Mode-specific flags ---
    if self.enable_legacy:
        # Output file (legacy-only)
        out_path = kwargs.get("output_file") or self.output_file
        if not out_path and self.raw_output_to_file:
            out_path = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["_auto_output_file"] = out_path
        if out_path:
            parts.extend(["--output-file", out_path])
        # Legacy-only flags
        if self.xid:
            parts.extend(["--xid", self.xid])
        if self.jira:
            parts.extend(["--jira", self.jira])
        if self.enable_deep_plan:
            parts.append("--enable-deep-plan")
        if self.agent_mode:
            parts.extend(["--agent-mode", self.agent_mode])
    else:
        # Non-legacy: auto-inject output-schema if needed for clean output
        if not self.output_schema and self.raw_output_to_file:
            parts.extend(["--output-schema", shlex.quote(_NON_LEGACY_OUTPUT_SCHEMA)])
            kwargs["_auto_output_schema"] = True

        # Non-legacy exclusive flags
        if self.config_override:
            parts.extend(["--config-override", shlex.quote(self.config_override)])
        if self.worktree:
            parts.extend(["--worktree", shlex.quote(self.worktree)])

        # Warn about legacy-only flags (including --xid, contra other plan)
        for name, val in [
            ("xid", self.xid), ("jira", self.jira),
            ("enable_deep_plan", self.enable_deep_plan),
            ("agent_mode", self.agent_mode),
        ]:
            if val:
                logger.warning(
                    "[%s] '%s' is legacy-only and ignored in non-legacy mode",
                    self.__class__.__name__, name,
                )

    if self.extra_cli_args:
        parts.extend(self.extra_cli_args)

    return " ".join(parts)
```

### 4.4 `parse_output()` Changes

Add non-legacy JSON extraction path (adopted from other plan, hardened):

```python
def parse_output(self, stdout, stderr, return_code, output_file_path=None):
    # Non-legacy with auto-injected schema: extract from JSON in stdout
    if not self.enable_legacy and not self.output_schema:
        parsed = extract_json_from_output(stdout)
        if parsed and "result" in parsed:
            output = parsed["result"]
            return {
                "output": output, "raw_output": stdout,
                "stderr": stderr, "return_code": return_code,
                "success": return_code == 0,
            }
        # Fall through to ANSI stripping if JSON extraction fails
        logger.debug("[%s] JSON extraction failed, falling back to ANSI strip",
                     self.__class__.__name__)

    # ... existing logic unchanged ...
```

### 4.5 `extract_json_from_output()` Helper (in `common.py`)

Hardened version of the other plan's `extract_trailing_json()`:

```python
def extract_json_from_output(text: str) -> Optional[dict]:
    """Extract the last valid JSON object from text output.

    Searches backward from end of text to find the last complete JSON object.
    Handles nested braces correctly via json.loads validation.
    Returns None if no valid JSON found.
    """
    text = text.rstrip()
    if not text.endswith("}"):
        return None

    # Find matching opening brace by searching backwards
    depth = 0
    in_string = False
    escape_next = False
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '}':
            depth += 1
        elif ch == '{':
            depth -= 1
            if depth == 0:
                candidate = text[i:]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None
```

Note: The backward search + `json.loads` validation is more robust than simple
brace matching. If the text has multiple JSON objects, we get the last one (which
is where `--output-schema` output appears in the CLI).

### 4.6 `_infer()` and `ainfer()` Guards

Only create auto temp files in legacy mode:

```python
# In _infer():
auto_output_file = None
if self.enable_legacy and not self.output_file and self.raw_output_to_file:
    auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
    kwargs["output_file"] = auto_output_file

# Same pattern in ainfer()
```

### 4.7 No Changes Needed

- `_yield_filter()` — ANSI stripping applies to all modes
- `_build_session_args()` — `--restore` works in both modes
- `rovodev_serve_inferencer.py` — independent (uses `serve` subcommand)

### 4.8 `rovodev_anthropic_proxy.py` Update

Add `--disable-legacy` flag to `_parse_args()`:

```python
parser.add_argument("--disable-legacy", action="store_true", default=False,
                    help="Use non-legacy (TUI) mode instead of legacy CLI")
```

Pass to inferencer: `enable_legacy=not args.disable_legacy`

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `rovodev_cli_inferencer.py` | Add attrs, refactor `construct_command()`, guard temp file, update `parse_output()`, docstrings |
| `common.py` | Add `extract_json_from_output()` helper |
| `rovodev_anthropic_proxy.py` | Add `--disable-legacy` CLI flag |

---

## 6. Implementation Steps

| # | Task |
|---|------|
| 1 | Add `enable_legacy`, `config_override`, `worktree` attrs + `_NON_LEGACY_OUTPUT_SCHEMA` constant |
| 2 | Add `extract_json_from_output()` to `common.py` |
| 3 | Refactor `construct_command()` with legacy/non-legacy branching |
| 4 | Update `parse_output()` with JSON extraction path for non-legacy |
| 5 | Guard temp output file in `_infer()` and `ainfer()` |
| 6 | Update `rovodev_anthropic_proxy.py` with `--disable-legacy` flag |
| 7 | Update docstrings |
| 8 | Write unit tests |
| 9 | Verify existing behavior unchanged (legacy mode) |

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Test | Verifies |
|------|----------|
| `test_construct_command_legacy_default` | Default produces `acli rovodev legacy <prompt>` |
| `test_construct_command_non_legacy` | `enable_legacy=False` omits `legacy` subcommand |
| `test_non_legacy_skips_xid` | `--xid` absent in non-legacy (corrects other plan) |
| `test_non_legacy_skips_jira` | `--jira` absent + warning logged |
| `test_non_legacy_skips_deep_plan` | `--enable-deep-plan` absent |
| `test_non_legacy_skips_agent_mode` | `--agent-mode` absent |
| `test_non_legacy_skips_output_file` | `--output-file` never in non-legacy |
| `test_non_legacy_auto_output_schema` | `--output-schema` auto-injected when `raw_output_to_file=True` |
| `test_non_legacy_user_schema_preserved` | User's `output_schema` used, not auto-injected |
| `test_non_legacy_no_auto_schema_without_raw_output` | No auto-injection when `raw_output_to_file=False` |
| `test_non_legacy_yolo_present` | `--yolo` still works |
| `test_non_legacy_restore_present` | `--restore` still works |
| `test_non_legacy_config_override` | `--config-override` included |
| `test_non_legacy_worktree` | `--worktree` included |
| `test_extract_json_from_output_valid` | Parses clean JSON |
| `test_extract_json_from_output_with_tui_noise` | Parses JSON after TUI output |
| `test_extract_json_from_output_nested` | Handles nested `{}`  |
| `test_extract_json_from_output_no_json` | Returns None gracefully |
| `test_extract_json_from_output_invalid_json` | Returns None for malformed |
| `test_parse_output_non_legacy_json` | `parse_output()` extracts via JSON |
| `test_parse_output_non_legacy_fallback` | Falls back to ANSI strip if no JSON |
| `test_enable_legacy_default_true` | Backward compat default |

---

## 8. Resolved Open Questions

| Question | Resolution |
|----------|------------|
| Is `--xid` available in non-legacy? | **NO** — verified via `acli rovodev run --help`. Legacy-only. |
| How to get clean output without `--output-file`? | Use `--output-schema` with auto-injected schema + JSON extraction from stdout. |
| Is `--output-schema` available in non-legacy? | **YES** — verified in help output. |
| When to auto-inject schema? | Only when `raw_output_to_file=True` AND no user-set `output_schema`. Minimizes surprise. |

## 9. Remaining Open Questions

1. **TUI stdout noise level**: How much Rich formatting leaks into stdout when
   `--output-schema` is used? If the JSON is cleanly separated, `extract_json_from_output()`
   works well. If JSON is interleaved with TUI content, we may need `--no-verbose` auto-injection.

2. **`--config-override` for legacy-only features**: Can `--xid` etc. be passed via
   `--config-override` JSON? If so, we could auto-translate instead of warning.
