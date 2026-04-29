# Issue 001: Gateway Streaming Protocol Mismatch + Inferencer Init Hang

**Date:** 2026-04-17
**Severity:** High — `_stream_gateway()` silently drops all streaming data from the live gateway
**Status:** Open

---

## Summary

Two issues discovered when calling `OpenClawInferencer(mode="gateway")` against the live OpenClaw gateway at `ws://127.0.0.1:18789`:

1. **Init hang** — the inferencer hangs during Python initialization and never reaches the WebSocket call
2. **Protocol mismatch** — `_stream_gateway()` expects a frame format (`state=delta/final`) that the live gateway does not produce, so even if the init hang is resolved, streaming would silently yield nothing

---

## Issue A: Inferencer Init Hang

### Reproduction

```python
import sys
sys.path.insert(0, '.../AgentFoundation/src')
sys.path.insert(0, '.../RichPythonUtils/src')

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import OpenClawInferencer

inf = OpenClawInferencer(
    mode="gateway",
    gateway_url="ws://127.0.0.1:18789",
    auth_token="<valid-token>",
    session_id="test-session",
    timeout_seconds=120,
)

# This never executes — process hangs above
async for chunk in inf.ainfer_streaming("Hello"):
    print(chunk)
```

### Observed Behavior

- The Python process stays alive (visible in `ps aux`) but produces no output
- The gateway is confirmed reachable (raw WebSocket handshake completes in 0.04s)
- The hang occurs **before any WebSocket connection is made**

### Suspected Root Cause

The hang is in the Python import/initialization chain, not the gateway. Possible locations:

1. **Heavy import tree** — `agent_foundation` → `streaming_inferencer_base` → `rich_python_utils` may have import-time side effects or heavy dependency loading
2. **`StreamingInferencerBase.__attrs_post_init__()`** — called via `super().__attrs_post_init__()` at line 170; may perform blocking initialization
3. **Import-time global state** — some packages in the dependency tree may attempt network calls, lock acquisition, or heavy computation at import time

### Status

Not yet diagnosed to exact root cause. Needs step-by-step import profiling:
```python
print("step 1"); import agent_foundation
print("step 2"); from agent_foundation.common.inferencers... import OpenClawInferencer
print("step 3"); inf = OpenClawInferencer(...)
print("step 4"); # try calling
```

---

## Issue B: `_stream_gateway()` Protocol Mismatch

### The Code Expects (lines 382–509)

```python
# _stream_gateway() looks for:
data = frame.get("payload", {}).get("data", {})
state = data.get("state")                          # expects "state" key

if state == "delta":                                # expects state="delta"
    for item in data.get("message", {}).get("content", []):
        if item.get("type") == "text":
            chunk = item["text"]                    # expects nested message.content[].text

elif state == "final":                              # expects state="final"
    ...
```

### What the Live Gateway Actually Sends (protocol v3)

```
Lifecycle events:
  {"phase": "start", "startedAt": 1776447141334}
  {"phase": "end"}

Streaming text events:
  {"text": "Here are all my",           "delta": "Here are all my"}
  {"text": "Here are all my available", "delta": " available"}
  {"text": "...(cumulative)...",        "delta": "(incremental new chars)"}

Final result:
  type="res" frame (not type="event") with payload keys: [runId, status, summary, result]
```

### Key Differences

| Aspect | Code Expects | Gateway Actually Sends |
|--------|-------------|----------------------|
| State field | `data.state = "delta"` / `"final"` | No `state` field; uses `data.phase` for lifecycle |
| Text location | `data.message.content[].text` (cumulative) | `data.text` (cumulative) + `data.delta` (incremental) |
| Completion signal | `event` frame with `state="final"` | `res` frame with `summary="completed"` |
| Lifecycle | Not handled | `data.phase = "start"` / `"end"` |

### Impact

- `data.get("state")` returns `None` for every streaming frame
- None of the `if state == "delta"` / `elif state == "final"` branches ever execute
- **Result: `_stream_gateway()` yields zero chunks and hangs until timeout**

### Suggested Fix

```python
async def _stream_gateway(self, prompt: str, session_id: str) -> AsyncIterator[str]:
    ws = await self._ws_connect()
    req_id = str(uuid.uuid4())
    # ... send agent request (unchanged) ...

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout_seconds + 10)
        frame = json.loads(raw)
        ftype = frame.get("type")
        fevent = frame.get("event", "")

        # Agent request ack
        if ftype == "res" and frame.get("id") == req_id:
            if not frame.get("ok"):
                # ... error handling (unchanged) ...
            payload = frame.get("payload", {})
            # Check if this is the FINAL result frame (has 'summary' or 'result')
            if "summary" in payload or "result" in payload:
                break  # Agent run completed
            continue

        if ftype != "event" or fevent != "agent":
            continue

        data = frame.get("payload", {}).get("data", {})

        # Phase lifecycle
        if "phase" in data:
            if data["phase"] in ("end", "done", "complete"):
                break
            continue

        # Streaming text — use data.delta for incremental text
        if "delta" in data:
            delta = data["delta"]
            if delta:
                yield delta

        # Error handling
        if data.get("phase") in ("aborted", "error"):
            err_msg = data.get("errorMessage", "unknown error")
            if is_rate_limit_error(err_msg):
                raise OpenClawRateLimitError(...)
            raise OpenClawError(f"OpenClaw agent error: {err_msg}")
```

---

## Verification

A raw WebSocket call using the corrected protocol works perfectly:

```
Session: skills-fresh-004
TTFT: 8.9s
Total: 29.9s
Response: 2926 chars — full skills list returned successfully
```

---

## Files Affected

- `openclaw_inferencer.py` — `_stream_gateway()` method (lines 382–509)
- `openclaw_inferencer.py` — `__attrs_post_init__()` / import chain (init hang)
