# Rovo Dev Inferencer — Implementation Plan (v4 Final)

## 1. Executive Summary

This plan creates Rovo Dev inferencers under `external/rovodev/` that wrap the `acli rovodev` CLI for the AgentFoundation framework. Due to a proof-of-origin mechanism (`X-RovoDev-Proof`), direct API calls get model-downgraded (opus → haiku), so going through the CLI binary is the only way to get full model quality.

**Two inferencers: run-mode first (simpler), then serve-mode (persistent sessions/streaming).**

---

## 2. Critical Analysis: Run Mode vs Serve Mode

### Run Mode (`acli rovodev run "prompt"`)

- Providing a `message` argument makes it non-interactive (`_is_interactive()` returns `False`). The process runs once and exits.
- `--output-file` captures the final text/JSON output cleanly (not TUI noise).
- `--output-schema` enables structured JSON responses.
- `--restore` (boolean flag, NOT a session ID arg) resumes the most recent session for the current working directory.
- **Limitation:** stdout contains Rich TUI formatting. `--output-file` only captures the final result, not streaming chunks.
- **Best for:** one-shot `_infer()` calls where you need the final result.

### Serve Mode (`acli rovodev serve <port>`)

- Starts a FastAPI server with clean REST API + SSE streaming.
- **Best for:** streaming (`_ainfer_streaming()`), persistent multi-turn, clean structured output.
- **Cost:** server lifecycle management.

### Decision: Build Both

| Inferencer | Base Class | Use Case |
|---|---|---|
| `RovoDevCliInferencer` | `TerminalSessionInferencerBase` | One-shot calls, matches ClaudeCodeCliInferencer pattern |
| `RovoDevServeInferencer` | `StreamingInferencerBase` | Streaming, persistent multi-turn, clean SSE |

Implementation order: Run-mode first, then serve-mode.

---

## 3. File Structure

```
rovodev/
├── __init__.py                    # Exports both inferencers
├── common.py                     # Shared constants & helpers
├── rovodev_cli_inferencer.py     # Phase 1: Run-mode inferencer
└── rovodev_serve_inferencer.py   # Phase 2: Serve-mode inferencer
```

---

## 4. Phase 1: Run-Mode Inferencer

### 4.1 `common.py`

```python
ACLI_BINARY = "acli"
ACLI_SUBCOMMAND = "rovodev"
DEFAULT_IDLE_TIMEOUT = 1800           # 30 min
DEFAULT_TOOL_USE_IDLE_TIMEOUT = 7200  # 2 hr for tool use
DEFAULT_PORT_RANGE = (19100, 19200)
HEALTHCHECK_POLL_INTERVAL = 0.5

class RovoDevNotFoundError(RuntimeError):
    """acli binary not found."""

class RovoDevAuthError(RuntimeError):
    """Authentication with Rovo Dev failed."""

def find_acli_binary(explicit_path: str | None = None) -> str:
    """Find acli binary, raise RovoDevNotFoundError if not found."""

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""

def find_available_port(start: int = 19100, end: int = 19200) -> int:
    """Find an available TCP port in the given range. Used by serve-mode."""
```

### 4.2 `RovoDevCliInferencer` (extends `TerminalSessionInferencerBase`)

**Reference:** `ClaudeCodeCliInferencer` at `external/claude_code/claude_code_cli_inferencer.py`

#### Attributes

```python
@attrs
class RovoDevCliInferencer(TerminalSessionInferencerBase):
    # --- Configuration ---
    acli_path: str | None = attrib(default=None)
    config_file: str | None = attrib(default=None)
    site_url: str | None = attrib(default=None)
    cloud_id: str | None = attrib(default=None)
    yolo: bool = attrib(default=True)
    enable_deep_plan: bool = attrib(default=False)
    xid: str | None = attrib(default=None)
    output_schema: str | None = attrib(default=None)
    output_file: str | None = attrib(default=None)  # User-facing: explicit output path
    agent_mode: str | None = attrib(default=None)
    jira: str | None = attrib(default=None)
    extra_cli_args: list[str] | None = attrib(default=None)

    # --- Timeouts ---
    idle_timeout_seconds: int = attrib(default=1800)
    tool_use_idle_timeout_seconds: int = attrib(default=7200)
```

**Inherited attributes (NOT redefined):**
- `auto_resume: bool = True` — from `StreamingInferencerBase`
- `session_arg_name` / `resume_arg_name` — from `TerminalSessionInferencerBase` (unused; `_build_session_args()` is fully custom)
- `working_dir` — from `TerminalSessionInferencerBase`
- `active_session_id` — from `StreamingInferencerBase`

#### `__attrs_post_init__()`

```python
def __attrs_post_init__(self) -> None:
    if self.acli_path is None:
        self.acli_path = shutil.which(ACLI_BINARY)
    if self.working_dir is None:
        self.working_dir = os.getcwd()
    super().__attrs_post_init__()
```

#### `construct_command(inference_input, **kwargs) -> str`

```python
def construct_command(self, inference_input, **kwargs):
    """Build the acli rovodev run command.

    Session management (session_id/resume) is injected into kwargs
    by the ainfer()/infer() overrides, following ClaudeCodeCliInferencer's pattern.
    """
    acli = self.acli_path
    if not acli:
        raise RovoDevNotFoundError()

    prompt = self._extract_prompt(inference_input)
    parts = [acli, "rovodev", "run", shlex.quote(prompt)]

    if self.yolo:
        parts.append("--yolo")

    # Output file (user-specified or auto-generated temp)
    out_path = kwargs.get("output_file") or self.output_file
    if out_path:
        parts.extend(["--output-file", out_path])

    if self.config_file:
        parts.extend(["--config-file", self.config_file])
    if self.site_url:
        parts.extend(["--site-url", self.site_url])
    if self.xid:
        parts.extend(["--xid", self.xid])
    if self.jira:
        parts.extend(["--jira", self.jira])
    if self.enable_deep_plan:
        parts.append("--enable-deep-plan")
    if self.output_schema:
        parts.extend(["--output-schema", shlex.quote(self.output_schema)])

    # Session resumption — delegated to _build_session_args()
    session_id = kwargs.get("session_id")
    is_resume = kwargs.get("resume", False)
    if is_resume and session_id:
        session_args = self._build_session_args(session_id, is_resume)
        if session_args:
            parts.append(session_args)

    if self.extra_cli_args:
        parts.extend(self.extra_cli_args)

    return " ".join(parts)
```

#### `_build_session_args(session_id, is_resume) -> str`

```python
def _build_session_args(self, session_id: str, is_resume: bool) -> str:
    """Build CLI session arguments.

    IMPORTANT: Rovo Dev's --restore is a boolean flag (no session ID argument).
    It always restores the most recent session for the current working directory.
    Unlike Claude Code which supports --resume <session_id>,
    Rovo Dev cannot target a specific session by ID.
    """
    if is_resume:
        if session_id and session_id != "active":
            logger.warning(
                "RovoDevCliInferencer: --restore always resumes the most recent session. "
                "Provided session_id=%s will be ignored.", session_id,
            )
        return "--restore"
    return ""
```

#### `ainfer()` / `infer()` Overrides — Session Management

These overrides follow the exact same pattern as `ClaudeCodeCliInferencer` (lines 395-503).
Without them, `session_id`/`resume` kwargs never get injected into `construct_command()`,
and `active_session_id` never gets updated after a successful call.

```python
async def ainfer(self, inference_input, inference_config=None, **kwargs):
    """Async inference with session management.

    1. Handle new_session flag
    2. Determine session context (auto-resume if applicable)
    3. Inject session_id/resume into kwargs
    4. Route through _ainfer_single for retry/timeout
    5. Update active_session_id from result
    """
    new_session = kwargs.pop("new_session", False)
    if new_session:
        self.active_session_id = None

    session_id = kwargs.get("session_id", self.active_session_id)
    is_resume = kwargs.get("resume", True)

    if session_id is None:
        if self.auto_resume and self.active_session_id:
            session_id = self.active_session_id
        else:
            is_resume = False

    kwargs["session_id"] = session_id
    kwargs["resume"] = is_resume and session_id is not None

    result = await self._ainfer_single(inference_input, inference_config, **kwargs)

    # Since --restore doesn't return a session ID, use a sentinel value
    # to indicate an active session exists for auto-resume.
    # result is TerminalInferencerResponse, not dict — use getattr()
    if getattr(result, "success", False):
        if self.active_session_id is None:
            self.active_session_id = "active"

    return result

def infer(self, inference_input, inference_config=None, **kwargs):
    """Sync inference with session management (mirrors ainfer)."""
    new_session = kwargs.pop("new_session", False)
    if new_session:
        self.active_session_id = None

    session_id = kwargs.get("session_id", self.active_session_id)
    is_resume = kwargs.get("resume", True)

    if session_id is None:
        if self.auto_resume and self.active_session_id:
            session_id = self.active_session_id
        else:
            is_resume = False

    kwargs["session_id"] = session_id
    kwargs["resume"] = is_resume and session_id is not None

    result = self._infer_single(inference_input, inference_config, **kwargs)

    # result is TerminalInferencerResponse, not dict — use getattr()
    if getattr(result, "success", False):
        if self.active_session_id is None:
            self.active_session_id = "active"

    return result
```

**Note on `active_session_id = "active"`:** Since Rovo Dev's `--restore` doesn't take or return a session ID, we use a sentinel value `"active"` to indicate that an active session exists. This makes `auto_resume` work: on the next call, `session_id` will be `"active"` (truthy), `is_resume=True`, and `_build_session_args()` will emit `--restore`.

#### `parse_output(stdout, stderr, return_code) -> dict`

```python
def parse_output(self, stdout, stderr, return_code):
    """Parse acli rovodev run output.

    Strategy:
    1. Read from --output-file if available (clean, no Rich formatting)
    2. Fall back to cleaned stdout
    """
    output = stdout

    # Check for output file (user-specified or from construct_command)
    out_path = self.output_file
    if out_path and Path(out_path).exists():
        output = Path(out_path).read_text(encoding="utf-8").strip()
    else:
        output = strip_ansi_codes(stdout).strip()

    return {
        "output": output,
        "raw_output": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "success": return_code == 0,
    }
```

#### `_yield_filter()` — Async Generator (Correct Signature)

```python
async def _yield_filter(
    self, chunks: AsyncIterator[str], **kwargs: Any
) -> AsyncIterator[str]:
    """Filter streaming output — strip ANSI codes from Rich TUI stdout.

    This is an async generator that transforms an entire async iterator of chunks,
    matching the StreamingInferencerBase._yield_filter signature.

    Note: stdout streaming from acli rovodev run is noisy due to Rich TUI.
    For clean streaming, use RovoDevServeInferencer (Phase 2).
    """
    async for line in super()._yield_filter(chunks, **kwargs):
        clean = strip_ansi_codes(line)
        if clean.strip():
            yield clean
```

---

## 5. Phase 2: Serve-Mode Inferencer (Outline)

**File:** `rovodev_serve_inferencer.py`

### `RovoDevServeInferencer` (extends `StreamingInferencerBase`)

**References:**
- `RovoChatInferencer` — streaming HTTP pattern
- `ClaudeCodeInferencer` — connection lifecycle pattern

#### Attributes

```python
port: int | None = attrib(default=None)
disable_session_token: bool = attrib(default=True)
startup_timeout: int = attrib(default=60)
non_interactive: bool = attrib(default=True)
respect_configured_permissions: bool = attrib(default=False)
agent_mode: str | None = attrib(default=None)

# Internal state
_server_process: asyncio.subprocess.Process | None = attrib(default=None, init=False)
_base_url: str = attrib(default="", init=False)
_http_client: httpx.AsyncClient | None = attrib(default=None, init=False)
```

#### Key Methods

```python
async def aconnect(self, **kwargs):
    """Start acli rovodev serve, poll /healthcheck until ready."""

async def adisconnect(self):
    """SIGTERM → wait 5s → SIGKILL. Close HTTP client."""

async def _ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
    """POST /v3/set_chat_message → GET /v3/stream_chat (SSE).
    Parse events: text_delta → yield text, tool_call → yield "", agent_run_end → stop.
    """

async def _ainfer(self, inference_input, inference_config=None, **kwargs):
    """Handle new_session (POST /v3/reset), accumulate streaming, return SDKInferencerResponse."""

async def __aenter__(self): await self.aconnect(); return self
async def __aexit__(self, *exc): await self.adisconnect()
```

**Dependency:** `httpx` (already in project)

**SSE event format needs validation** — run `acli rovodev serve` and inspect real events before implementing.

---

## 6. Registration

### `agentic_inferencers/__init__.py`

Add lazy imports (before `raise AttributeError`):

```python
elif name == "RovoDevCliInferencer":
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
        RovoDevCliInferencer,
    )
    return RovoDevCliInferencer
elif name == "RovoDevServeInferencer":
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
        RovoDevServeInferencer,
    )
    return RovoDevServeInferencer
```

Add both to `__all__`.

---

## 7. Session Resumption — Key Difference from Claude Code

| Feature | Claude Code CLI | Rovo Dev CLI |
|---------|----------------|--------------|
| Arbitrary session ID | `--resume <session_id>` | ❌ Not supported |
| Resume most recent | `--resume` (no session_id) | `--restore` (boolean flag) |
| Session persistence | Per working directory | Per working directory |
| Auto-resume mechanism | `active_session_id = <real_id>` from JSON output | `active_session_id = "active"` (sentinel) |

---

## 8. Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `external/rovodev/__init__.py` | Create — module exports | 1 |
| `external/rovodev/common.py` | Create — shared constants & helpers | 1 |
| `external/rovodev/rovodev_cli_inferencer.py` | Create — run-mode inferencer | 1 |
| `agentic_inferencers/__init__.py` | Edit — add lazy imports + `__all__` | 1 |
| `external/rovodev/rovodev_serve_inferencer.py` | Create — serve-mode inferencer | 2 |

---

## 9. Testing Plan

### 9.1 Unit Tests (no acli required)

```python
# === Command Construction ===
def test_construct_command_minimal():
    """acli rovodev run 'prompt' --yolo"""

def test_construct_command_with_config():
def test_construct_command_with_site_url():
def test_construct_command_with_output_file():
    """User-specified output_file attribute produces --output-file flag."""

def test_construct_command_with_output_schema():
def test_construct_command_with_jira():
def test_construct_command_with_restore():
    """--restore in session args when resume=True."""

def test_construct_command_with_deep_plan():
def test_construct_command_with_xid():
def test_construct_command_with_extra_args():
def test_construct_command_yolo_disabled():

# === Output Parsing ===
def test_parse_output_from_file():
def test_parse_output_fallback_to_stdout():
def test_parse_output_strips_ansi():
def test_parse_output_failure():

# === Session Management ===
def test_build_session_args_resume():
def test_build_session_args_no_resume():
def test_build_session_args_ignores_session_id_with_warning():

# === ainfer/infer Session Overrides ===
def test_ainfer_injects_session_kwargs():
    """session_id and resume are injected into kwargs."""

def test_ainfer_updates_active_session_id_on_success():
    """active_session_id set to 'active' after successful inference."""

def test_ainfer_new_session_clears_active_session():
    """new_session=True clears active_session_id."""

def test_ainfer_auto_resume_uses_active_session():
    """When auto_resume=True and active_session_id is set, resume=True."""

# === Yield Filter ===
def test_yield_filter_strips_ansi():
    """ANSI-only lines are filtered out."""

def test_yield_filter_passes_clean_text():
    """Clean text passes through."""

# === Error Handling ===
def test_acli_not_found_raises():

# === Utilities ===
def test_find_acli_binary_found():
def test_find_acli_binary_not_found():
def test_strip_ansi_codes():
def test_find_available_port():
```

### 9.2 Integration Tests (require acli authenticated)

```python
@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("acli"), reason="acli not installed")
class TestRovoDevCliIntegration:

    def test_single_turn_sync(self, tmp_path):
        inf = RovoDevCliInferencer(working_dir=str(tmp_path))
        result = inf("What is 2+2? Reply with just the number.")
        assert "4" in str(result)

    def test_output_file_captured(self, tmp_path):
        out = str(tmp_path / "output.txt")
        inf = RovoDevCliInferencer(working_dir=str(tmp_path), output_file=out)
        inf("What is 3+3? Reply with just the number.")
        assert Path(out).exists()
        assert "6" in Path(out).read_text()

    def test_multi_turn_resume(self, tmp_path):
        inf = RovoDevCliInferencer(working_dir=str(tmp_path))
        r1 = inf.new_session("Remember: my secret is 42")
        assert inf.active_session_id == "active"
        r2 = inf("What is my secret?")
        assert "42" in str(r2)
```

### 9.3 Phase 2 Verification

1. Manually run `acli rovodev serve 19100 --disable-session-token` and inspect SSE events
2. Verify server starts and healthcheck passes
3. Verify SSE streaming yields text chunks
4. Verify `POST /v3/reset` resets session
5. Verify `adisconnect()` cleanly terminates server

---

## 10. Implementation Order

| Step | Task | Phase | Est. |
|------|------|-------|------|
| 1 | Create directory + `__init__.py` | 1 | 5 min |
| 2 | Implement `common.py` | 1 | 20 min |
| 3 | Implement `rovodev_cli_inferencer.py` — class, `__attrs_post_init__`, `construct_command` | 1 | 45 min |
| 4 | Implement `parse_output`, `_yield_filter` | 1 | 30 min |
| 5 | Implement `_build_session_args`, `ainfer()`/`infer()` session overrides | 1 | 30 min |
| 6 | Update `agentic_inferencers/__init__.py` | 1 | 10 min |
| 7 | Write unit tests | 1 | 45 min |
| 8 | Write integration tests, smoke test | 1 | 30 min |
| **Phase 1 Total** | | | **~4 hrs** |
| 9 | Validate SSE event format by running serve | 2 | 30 min |
| 10 | Implement `rovodev_serve_inferencer.py` | 2 | 3 hrs |
| 11 | Write serve-mode tests | 2 | 1 hr |
| **Total (both phases)** | | | **~8.5 hrs** |

---

## 11. Feedback Validation Log

| Feedback | Valid? | Action Taken |
|---|---|---|
| `_yield_filter` wrong signature (Issue 1) | ✅ Valid | Fixed: async generator over `AsyncIterator[str]`, calls `super()._yield_filter()` |
| Missing `ainfer()`/`infer()` overrides (Issue 2) | ✅ Valid | Fixed: added both overrides with session management, `active_session_id = "active"` sentinel |
| `output_file` not a class attribute (Issue 3) | ✅ Valid | Fixed: added `output_file: str \| None = attrib(default=None)` as real attribute |
| `construct_command` duplicates session logic (Issue 4) | ✅ Valid | Fixed: session logic uses `_build_session_args()`, kwargs from `ainfer()`/`infer()` |
| `_ainfer_streaming` receives string not dict (Issue 5) | ✅ Valid observation, not a bug | Noted — `_extract_prompt` handles both |
| Temp file race condition (Issue 6) | ⚠️ Minor | Removed auto-temp-file; user controls `output_file` attribute instead |
| `__attrs_post_init__` order (Issue 7) | ✅ Already correct | No change needed |
| `auto_resume` redefinition (earlier) | ✅ Valid | Not redefined — uses inherited |
| `session_arg_name`/`resume_arg_name` override (earlier) | ⚠️ Partially valid | Not overridden — `_build_session_args()` is custom |
| `isinstance(result, dict)` check in ainfer/infer | ✅ Valid | Fixed: `result` is `TerminalInferencerResponse`, not dict. Changed to `getattr(result, "success", False)` |
| `--restore <session_id>` (other agent's plan) | ❌ Invalid | `--restore` is `bool \| None`, not a session ID arg |
| Missing `--jira` flag (other plan) | ✅ Valid | Added `jira` attribute |
| Missing `tool_use_idle_timeout_seconds` (other plan) | ✅ Valid | Added with 7200s default |
| Missing `respect_configured_permissions` (other plan) | ✅ Valid | Added to Phase 2 serve-mode attributes |

---

## 12. Open Questions

1. **Run mode stdout format**: Verify what `acli rovodev run "message"` outputs to stdout — Rich TUI or clean text?
2. **SSE event format (Phase 2)**: Validate by running serve server before implementing.
3. **Model selection**: No `--model` flag. Use `ROVODEV_MODEL_ID` env var on subprocess.
4. **System prompt**: No `--system-prompt` flag. Prepend to first user message.
5. **`--restore` multi-turn**: Verify consecutive `--restore` calls maintain conversation.
