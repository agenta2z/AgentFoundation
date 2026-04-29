# Plan: `get_final_output()` — Clean Final Response for CLI Streaming Inferencers

**Date:** 2026-04-13  
**Scope:** `AgentFoundation` (Steps 1–3) + `OpenStartup` (Steps 4–7)  
**Priority:** HIGH — root cause fix for conversation tool detection  
**Decision:** Full fix, no tech debt, no minimal viable workarounds.

---

## Problem Statement

CLI-based streaming inferencers (e.g., `RovoDevCliInferencer`) produce **two distinct outputs**:

| Output | Source | Quality | Timing |
|---|---|---|---|
| **Streamed stdout** | `acli` Rich TUI rendering | Noisy — ANSI codes, `────` separators, terminal-width line wrapping, potential code-fence stripping | Token-by-token during streaming |
| **Clean final output** | `--output-file` (legacy mode) or trailing JSON schema (non-legacy mode) | Exact LLM text — code fences intact, no wrapping, no noise | Written atomically when `acli` exits |

`StreamingInferencerBase` currently has **no concept** of this distinction. `ainfer_streaming()` yields noisy stdout chunks, and `ConversationalInferencer.run_agentic_loop()` runs `parse_conversation_response(raw_response)` on the accumulated noisy stdout.

This causes `parse_conversation_response()` to fail detecting conversation tools when:
1. Code fences (` ```json ToolsToInvoke ``` `) are stripped or mangled by Rich TUI rendering
2. Long JSON is hard-wrapped at terminal width (80 chars), breaking single-line regex patterns

**Root cause:** `run_agentic_loop()` uses the noisy accumulated stream for tool parsing instead of the clean `--output-file` output that is fully available after `_ainfer_single()` returns.

---

## Verified Timing Facts (Source-Confirmed)

Critical timing that both plans initially got wrong — now verified:

```
ainfer_streaming() [RovoDevCliInferencer]:
    Line 420: _current_output_file.set(auto_output_file)   ← set
    Line 444: result = await self._ainfer_single(...)      ← streaming happens here
              └── _ainfer_streaming() generator runs
              └── stream_token_batches() accumulates raw_response
              └── ConversationalInferencer calls get_final_output() ← DURING _ainfer_single
    Line 458: Path(auto_output_file).unlink(missing_ok=True)  ← file deleted AFTER
    Line 465: _current_output_file.set(None)                  ← cleared AFTER

```

**Conclusion:** `get_final_output()` is called from `ConversationalInferencer` BEFORE line 458 (file deletion) and BEFORE line 465 (contextvar clear). Both the file and the contextvar are valid. ✅

**Cache file timing:**
```
_ainfer_streaming() finally block:
    → cache overwrite (Step 1 addition)   ← runs first
    → _finalize_cache()                   ← writes marker + closes file
```
`_finalize_cache()` closes the cache file. The cache overwrite must happen BEFORE `_finalize_cache()`, reopening the file by name using `cache_file.name`. ✅

---

## Architecture

### Data Flow (After Fix)

```
acli starts
    │
    ├── stdout → _ainfer_streaming() → _yield_filter() → token_gen()
    │            → stream_token_batches() → WS "token" messages → frontend (live display)
    │            → raw_response (noisy, streaming display only)
    │            → _last_raw_stdout accumulated in _yield_filter() (non-legacy path)
    │
    └── --output-file → written atomically when acli response completes
                     ↑ AVAILABLE inside _ainfer_single() after generator exhausted

After stream_token_batches() returns (still inside _ainfer_single()):
    clean_output = base_inferencer.get_final_output()
        → legacy:     reads _current_output_file path → clean text ✅
        → non-legacy: extract_json_from_output(self._last_raw_stdout)["response"] ✅
        → base class: returns None → falls back to raw_response ✅
    
    parse_conversation_response(clean_output)  ← correct detection ✅
    add_message("assistant", clean_output)     ← clean history ✅
    on_clean_output_available(clean_output)    ← notify WebSocketInteractive ✅
    
    WebSocketInteractive sends "stream_correction" → frontend replaces display ✅
    cache file overwritten with clean_output ✅
    
    message_end.final_content = result.text (from clean conv_response.text) ✅
```

---

## Files to Modify

| # | File | Change |
|---|---|---|
| 1 | `streaming_inferencer_base.py` | `streams_differ_from_final_output` + `get_final_output()` + cache overwrite in `finally` |
| 2 | `rovodev_cli_inferencer.py` | Override `streams_differ_from_final_output=True`, `get_final_output()`, accumulate `_last_raw_stdout` inside existing `_yield_filter()` |
| 3 | `conversational_inferencer.py` | After `stream_token_batches()`: call `get_final_output()`, use for parsing + history + notify interactive |
| 4 | `websocket_interactive.py` | `on_clean_output_available()`, `_clean_output` storage, `clean_output` property |
| 5 | `manager_websocket_routes.py` | No change needed — `result.text` already flows to `final_content` |
| 6 | `useManagerChat.js` | Handle `stream_correction` WS event |
| 7 | Tests | Unit tests for `get_final_output()`, integration tests for CI clean output |

---

## Detailed Implementation

### Step 1 — `StreamingInferencerBase`: Add hook + cache overwrite

**File:** `AgentFoundation/src/agent_foundation/common/inferencers/streaming_inferencer_base.py`

**Add after `fallback_recovery_template_key` (line ~165):**

```python
# Whether streamed chunks differ from the authoritative final output.
# CLI-based inferencers (e.g., RovoDevCliInferencer) set True — their stdout
# is noisy TUI output while --output-file has clean LLM text.
# API-based inferencers (Claude, GPT) leave this False — stream IS the output.
streams_differ_from_final_output: bool = False

def get_final_output(self) -> Optional[str]:
    """Return clean final output if it differs from concatenated stream.

    Subclasses where streamed tokens (stdout) differ from the actual LLM
    output (e.g., CLI inferencers with --output-file) override this to
    return the clean version after streaming completes.

    Must only be called AFTER ainfer_streaming() has completed (i.e., after
    the generator is exhausted and _ainfer_single() is returning).
    Returns None if stream == final output (default for API inferencers).

    Returns:
        Clean final output string, or None if stream == final output.
    """
    return None
```

**Add cache overwrite in `_ainfer_streaming()` finally block (BEFORE `_finalize_cache()`):**

```python
finally:
    # If the clean final output differs from the noisy stream, overwrite the
    # cache file with clean content so recovery inferences use correct context.
    if self.streams_differ_from_final_output and cache_file:
        final = self.get_final_output()
        if final:
            try:
                cache_path = getattr(cache_file, 'name', None)
                if cache_path:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(final)
                        f.write("\n--- STREAM COMPLETED SUCCESSFULLY ---\n")
                    success = True  # mark success so _finalize_cache skips re-writing marker
            except OSError as e:
                logger.warning(
                    "[%s] Failed to replace cache with clean output: %s",
                    self.__class__.__name__, e
                )
    self._finalize_cache(cache_file, success, error)
```

**Note:** `success = True` prevents `_finalize_cache()` from writing a duplicate marker since we already wrote it. Alternatively, add a `already_finalized` flag — but setting `success = True` is simpler and semantically correct (the stream did succeed).

**Note on `attrib` vs class-level bool:** `StreamingInferencerBase` uses `@attrs`. Adding `streams_differ_from_final_output` as a **class-level bool** (NOT `attrib()`) is intentional — it's a structural property of the inferencer TYPE, not a per-instance config. Subclasses override it at the class level (`streams_differ_from_final_output: bool = True`). This avoids attrs inheritance complexity and prevents users from accidentally setting it per-instance.

---

### Step 2 — `RovoDevCliInferencer`: Override `get_final_output()` + accumulate `_last_raw_stdout`

**File:** `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/external/rovodev/rovodev_cli_inferencer.py`

**Add class-level override after line ~122:**

```python
streams_differ_from_final_output: bool = True
```

**Add `get_final_output()` method:**

```python
def get_final_output(self) -> Optional[str]:
    """Read clean output from --output-file (legacy) or trailing JSON (non-legacy).

    Called after ainfer_streaming() completes, before auto_output_file is
    deleted (line 458) and before _current_output_file is cleared (line 465).
    Both the file and contextvar are valid at the time this is called.
    """
    if self.enable_legacy:
        # Legacy mode: --output-file has clean LLM output (no TUI noise)
        output_path = _current_output_file.get(None) or self.output_file
        if output_path:
            p = Path(output_path)
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        logger.debug(
                            "[%s] get_final_output: read %d chars from %s",
                            self.__class__.__name__, len(content), p
                        )
                        return content
                except OSError as e:
                    logger.warning(
                        "[%s] get_final_output: failed to read %s: %s",
                        self.__class__.__name__, p, e
                    )
    else:
        # Non-legacy mode: clean output embedded as trailing JSON in stdout
        raw = getattr(self, "_last_raw_stdout", None)
        if raw:
            try:
                parsed = extract_json_from_output(raw)
                if parsed and "response" in parsed:
                    content = parsed["response"].strip()
                    if content:
                        return content
            except Exception as e:
                logger.warning(
                    "[%s] get_final_output: failed to extract non-legacy output: %s",
                    self.__class__.__name__, e
                )
    return None
```

**Update existing `_yield_filter()` to accumulate `_last_raw_stdout` for non-legacy path:**

The existing `_yield_filter()` at line 368 already filters session headers etc. We add accumulation INSIDE it, not by replacing it:

```python
async def _yield_filter(self, chunks: AsyncIterator[str], **kwargs) -> AsyncIterator[str]:
    """Filter stdout chunks, accumulating raw content for non-legacy get_final_output()."""
    self._last_raw_stdout = ""  # reset per-call
    async for chunk in super()._yield_filter(chunks, **kwargs):
        if not self.enable_legacy:
            # Accumulate for extract_json_from_output() in get_final_output()
            self._last_raw_stdout += chunk
        yield chunk
```

**Critical:** We call `super()._yield_filter()` to preserve existing session-header stripping logic. We only ADD accumulation on top. The accumulated content is post-filter (already ANSI-stripped) — sufficient for `extract_json_from_output()` which looks for trailing JSON.

---

### Step 3 — `ConversationalInferencer`: Use clean output for parsing + notify interactive

**File:** `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py`

**Replace the block after `stream_token_batches()` returns (currently lines ~219–234):**

```python
raw_response = await effective_interactive.stream_token_batches(
    token_gen(), session_id, send_stream_end=False, turn_number=turn_number,
)
last_raw_response = raw_response

# Get clean final output if base inferencer has one (e.g., --output-file).
# CLI-based inferencers (streams_differ_from_final_output=True) return the
# clean text from --output-file or trailing JSON. API-based return None.
clean_response = raw_response
if getattr(self.base_inferencer, "streams_differ_from_final_output", False):
    final = self.base_inferencer.get_final_output()
    if final:
        clean_response = final
        logger.debug(
            "[ConversationalInferencer] Using clean final output (%d chars) "
            "instead of noisy stream (%d chars) for parsing",
            len(clean_response), len(raw_response)
        )
        # Notify interactive so it can update cache + send stream_correction to frontend
        if hasattr(effective_interactive, "on_clean_output_available"):
            try:
                await effective_interactive.on_clean_output_available(clean_response)
            except Exception as e:
                logger.warning(
                    "[ConversationalInferencer] on_clean_output_available failed: %s", e
                )

# Keep existing on_prompt_rendered block as-is (disabled debug block)
if False and on_prompt_rendered:
    try:
        await on_prompt_rendered(self, raw_response)
    except Exception:
        pass

# Add CLEAN output to conversation history (not noisy stdout)
self.add_message("assistant", clean_response)

# Parse conversation/action tools from CLEAN output
conv_response = parse_conversation_response(clean_response)
```

**All downstream code (`_handle_conversation_tools`, `AgenticResult`, etc.) uses `conv_response` and `clean_response` — no further changes needed.**

---

### Step 4 — `WebSocketInteractive`: `on_clean_output_available()` callback

**File:** `OpenStartup/src/openteam/server/services/websocket_interactive.py`

```python
def __init__(self, send_callback, input_queue):
    self._send = send_callback
    self._input_queue = input_queue
    self._clean_output: str | None = None  # set by on_clean_output_available()

@property
def clean_output(self) -> str | None:
    return self._clean_output

async def on_clean_output_available(self, clean_output: str) -> None:
    """Called after streaming completes when cleaner final output is available.

    Stores the clean output and sends a 'stream_correction' WS event so the
    frontend can replace the noisy streamed display with the clean version.
    """
    self._clean_output = clean_output
    await self._send({
        "type": "stream_correction",
        "content": clean_output,
    })
```

---

### Step 5 — `manager_websocket_routes.py`: No changes needed

`run_conversation_turn()` returns `AgenticResult`. `result.text` comes from `conv_response.text` which is derived from the clean output (Step 3). `final_content = result.text` in the route handler is already correct. ✅

---

### Step 6 — Frontend: Handle `stream_correction` WS event

**File:** `OpenStartup/src/openteam/ui/src/hooks/useManagerChat.js`

Add to the `handleServerMessage` switch statement:

```javascript
case 'stream_correction': {
    // Server has a clean version of the streamed content (from --output-file).
    // Replace the in-progress streaming display with the clean version.
    // This fires before message_end, so the committed message will be clean.
    const rawClean = data.content || '';
    streamingContentRef.current = rawClean;

    // Re-parse the clean content through the same pipeline as regular tokens
    const parsed = parseResponseTags(rawClean);
    const phase = parsed.phase === 'pre_response' ? 'no_tags' : parsed.phase;
    let displayContent;
    if (phase === 'no_tags') {
        displayContent = stripAnsi(stripAcliNoise(stripToolsToInvoke(rawClean)));
    } else {
        displayContent = stripSessionContext(
            stripAnsi(stripAcliNoise(stripToolsToInvoke(parsed.responseContent)))
        );
    }

    setStreamingMessage(prev => prev ? {
        ...prev,
        content: rawClean,
        displayContent,
        thinkingContent: parsed.thinkingContent,
        responsePhase: phase,
    } : prev);
    break;
}
```

**Note:** `parseResponseTags`, `parseSessionContext`, `stripAnsi`, `stripAcliNoise`, `stripToolsToInvoke`, `stripSessionContext` are all already defined/imported in `useManagerChat.js`. No new imports needed.

---

### Step 7 — Tests

**New test file:** `AgentFoundation/test/agent_foundation/inferencers/test_get_final_output.py`

```python
"""Tests for get_final_output() / streams_differ_from_final_output."""

def test_base_class_returns_none():
    """StreamingInferencerBase.get_final_output() returns None by default."""
    ...

def test_base_class_streams_differ_is_false():
    """StreamingInferencerBase.streams_differ_from_final_output is False."""
    ...

def test_rovodev_streams_differ_is_true():
    """RovoDevCliInferencer.streams_differ_from_final_output is True."""
    ...

def test_rovodev_get_final_output_reads_legacy_file(tmp_path):
    """get_final_output() reads from --output-file when enable_legacy=True."""
    output_file = tmp_path / "output.md"
    output_file.write_text("Clean LLM output with ```json ToolsToInvoke``` fences")
    _current_output_file.set(str(output_file))
    inferencer = RovoDevCliInferencer(working_dir=str(tmp_path), enable_legacy=True)
    result = inferencer.get_final_output()
    assert result == "Clean LLM output with ```json ToolsToInvoke``` fences"

def test_rovodev_get_final_output_returns_none_when_no_file():
    """get_final_output() returns None when output file not set."""
    _current_output_file.set(None)
    inferencer = RovoDevCliInferencer(working_dir="/tmp", enable_legacy=True, output_file=None)
    assert inferencer.get_final_output() is None

def test_rovodev_get_final_output_nonlegacy(monkeypatch):
    """get_final_output() extracts from _last_raw_stdout for non-legacy mode."""
    inferencer = RovoDevCliInferencer(working_dir="/tmp", enable_legacy=False)
    inferencer._last_raw_stdout = 'some noise\n{"response": "Clean LLM output"}\n'
    result = inferencer.get_final_output()
    assert result == "Clean LLM output"

def test_conversational_inferencer_uses_clean_output(mock_base_inferencer):
    """ConversationalInferencer uses get_final_output() for parse_conversation_response()."""
    # mock_base_inferencer.streams_differ_from_final_output = True
    # mock_base_inferencer.get_final_output() returns clean text with code fences
    # Verify parse_conversation_response called with clean text, not raw_response
    ...
```

---

## What This Replaces (Workarounds to Remove Later)

Once implemented and verified, these workarounds become unnecessary:

| Workaround | Location | Action |
|---|---|---|
| AF `conversation_response_parser.py` Path 3 (bare JSON regex) | AF | Keep as belt-and-suspenders — handles edge cases |
| `stripToolsToInvoke` bare JSON line stripping | OpenStartup `ThinkingFold.js` | Keep — handles display edge cases |
| `enable_legacy=True` explicit in `conversation_service.py` | OpenStartup | Keep — legacy mode preferred |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| `get_final_output()` called before file exists | LOW | Timing verified: called inside `_ainfer_single()`, before file deletion at line 458 |
| `_last_raw_stdout` adds memory overhead | LOW | Single string per inference turn, reset on each `_yield_filter()` call |
| Cache overwrite fails mid-write | LOW | Catch `OSError`; recovery falls back to noisy cache (still works, just noisier) |
| `stream_correction` causes UI flicker | LOW | Content is cleaner version of same text; visual diff is minimal |
| `on_clean_output_available()` raises exception | NONE | Step 3 wraps in try/except; graceful degradation to existing behavior |
| Non-legacy `extract_json_from_output()` returns None | NONE | Falls back to `raw_response` (existing behavior) |
| API-based inferencers (Claude, GPT) | NONE | `streams_differ_from_final_output=False` → `get_final_output()` never called |

---

## Implementation Order

```
Step 1: streaming_inferencer_base.py    ← hook + cache overwrite
    ↓
Step 2: rovodev_cli_inferencer.py       ← override + _yield_filter accumulation
    ↓ (can do Steps 3+4 in parallel)
Step 3: conversational_inferencer.py    ← use clean output for parsing
Step 4: websocket_interactive.py        ← on_clean_output_available
    ↓
Step 6: useManagerChat.js               ← stream_correction handler
    ↓
Step 7: Tests
```

Steps 1–4 are the critical path. Steps 5 (no-op), 6, and 7 can follow.
