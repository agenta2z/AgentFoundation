# OpenClaw Inferencer — Analysis & Implementation Plan

**Updated:** 2026-04-15  
**Status:** Revised — Single unified `OpenClawInferencer` class with `mode="gateway"` (streaming) and `mode="cli"` (subprocess)  
**Scope:** `AgentFoundation` framework; examples in `examples/...external/openclaw/`

---

## 1. What Is OpenClaw (Reality Check)

OpenClaw is a **TypeScript/Node.js** personal AI gateway. It is **not** a Python package and provides no Python SDK.

### How It Is Deployed (Production Setup — Confirmed by Live Testing)

OpenClaw runs as a **k3s Kubernetes cluster inside a Docker container** (`openshell-cluster-openshell`). Inside the cluster there are two key pods:

| Pod | Namespace | Role |
|---|---|---|
| `atlassian-openclaw-gateway` | `openshell` | Gateway process + `openclaw` CLI binary |
| `openshell-0` | `openshell` | Sandbox shell environment |

The gateway (`openclaw-gateway` process) listens on:
- `0.0.0.0:18789` — WebSocket gateway API (exposed to host via port-forward)
- `127.0.0.1:18791` — Internal IPC socket

### Auth Token (Critical Discovery)

Two different tokens exist — they are **not the same**:

| Source | Token |
|---|---|
| Printed by `run.sh` on host | The host-side gateway config token |
| Inside `atlassian-openclaw-gateway` pod | **Different token** — in `/sandbox/.openclaw/openclaw.json` |

The pod-internal token is `gateway.auth.token` in `/sandbox/.openclaw/openclaw.json`. The **web UI** must be accessed with `#token=<pod_token>` (hash fragment, not `?token=` query string — the UI strips query params).

---

## 2. Confirmed Working CLI Interface

After live testing, the **only working CLI invocation pattern** is:

```bash
docker exec openshell-cluster-openshell \
  kubectl exec -n openshell atlassian-openclaw-gateway -c agent -- sh -c \
  "OPENCLAW_CONFIG_PATH=/sandbox/.openclaw/openclaw.json \
   OPENCLAW_STATE_DIR=/sandbox/.openclaw \
   openclaw agent --local --session-id <session_id> --message '<prompt>' --json"
```

### Why `--local` is required

The `openclaw agent` command by default tries to reach the gateway via WebSocket (`ws://127.0.0.1:18789`). However, the CLI runs inside the **agent sidecar container** (`-c agent`) while the gateway runs in the main container of the same pod. They share the same network namespace but the WebSocket connection fails with "gateway closed (1006 abnormal closure)" because the agent sidecar cannot reach the gateway's loopback. Using `--local` bypasses the gateway entirely and runs the agent embedded in-process.

### Critical env vars

| Var | Value | Why |
|---|---|---|
| `OPENCLAW_CONFIG_PATH` | `/sandbox/.openclaw/openclaw.json` | Points to sandbox config (model provider, auth, gateway settings) |
| `OPENCLAW_STATE_DIR` | `/sandbox/.openclaw` | Points to sandbox agent dir (auth-profiles.json, agents/, sessions/) |

Without these, openclaw falls back to `/root/.openclaw/` which lacks `auth-profiles.json` → fails with "No API key found".

### Model provider chain (confirmed)

```
openclaw CLI → embedded agent → atlassian-ai-gateway-proxy
  → http://host.openshell.internal:29576/vertex/claude/v1
    → Proximity AI Gateway on host port 29576
      → Atlassian AI Gateway
        → Claude (claude-opus-4-6 by default)
```

### Full CLI options (relevant subset)

| Option | Type | Description |
|---|---|---|
| `--message <text>` | required | Prompt body |
| `--session-id <id>` | optional | Session ID (auto-creates if new) |
| `--agent <id>` | optional | Agent ID from config |
| `--thinking <level>` | optional | `off\|minimal\|low\|medium\|high\|xhigh` |
| `--timeout <seconds>` | optional | Default 600s |
| `--json` | flag | Output full JSON (recommended) |
| `--local` | flag | **Required** — run embedded without gateway |
| `--verbose <on\|off>` | optional | Verbose output |

### JSON output format (confirmed by live test)

```json
{
  "runId": "main",
  "status": "ok",
  "result": {
    "payloads": [
      { "text": "Hey there — I just came online, fresh and ready to go! 👋" }
    ],
    "sessionId": "main",
    "provider": "atlassian-ai-gateway-proxy",
    "model": "claude-opus-4-6",
    "usage": { "inputTokens": 1234, "outputTokens": 56 },
    "stopReason": "end_turn"
  }
}
```

### Session management (confirmed by live test)

- **New session**: use any new `--session-id` string; it auto-creates
- **Resume session**: reuse the same `--session-id`; context is preserved within a process lifetime
- **Session persistence**: with `--local`, sessions are **in-memory only** — they are NOT written to `sessions.json`. The `openclaw sessions` command shows 0 stored sessions.
- **Cross-run continuity**: NOT available with `--local` mode (no persistence). Each inferencer instance starts fresh.

---

## 3. Unified Inferencer Design: `OpenClawInferencer`

### Rationale: One Class, Two Transport Modes

Both the CLI subprocess and WebSocket gateway accomplish the same goal from the user's perspective — send a prompt to OpenClaw, get a response. The transport is an implementation detail. A single class with `mode` parameter is cleaner than two separate classes:

- **`mode="gateway"`** — WebSocket to `ws://127.0.0.1:18789` → true streaming + cross-run session restore ✅
- **`mode="cli"`** — Docker/kubectl subprocess `openclaw agent --local` → blocking, no streaming, no session persistence

Both modes share: `session_id`, `thinking`, `timeout_seconds`, `docker_container`, `kubectl_*` attributes.

### Base Class: `StreamingInferencerBase`

Both modes implement the `StreamingInferencerBase` interface. In CLI mode, `ainfer_streaming()` yields the full response as a single chunk (compatibility shim). In gateway mode it yields true token-by-token chunks.

### Attributes

```python
@attrs
class OpenClawInferencer(StreamingInferencerBase):
    # Mode
    mode: Literal["gateway", "cli"] = "gateway"
    # "gateway" → WebSocket streaming + persistent sessions (requires gateway running)
    # "cli"     → Docker subprocess blocking (always works, no streaming/session persist)

    # Docker/kubectl targeting (used by BOTH modes for token discovery + CLI exec)
    docker_container: str = "openshell-cluster-openshell"
    kubectl_namespace: str = "openshell"
    kubectl_pod: str = "atlassian-openclaw-gateway"
    kubectl_container: str = "agent"   # the sidecar container name

    # OpenClaw config paths (CLI mode: used as env vars; gateway mode: token discovery)
    openclaw_config_path: str = "/sandbox/.openclaw/openclaw.json"
    openclaw_state_dir: str = "/sandbox/.openclaw"

    # Gateway connection (gateway mode only)
    gateway_url: str = "ws://127.0.0.1:18789"
    auth_token: Optional[str] = None
    # If None → auto-discovered via docker exec at __attrs_post_init__

    # Agent params (used by BOTH modes)
    session_id: str = "main"
    agent_id: Optional[str] = None
    thinking: Optional[str] = None   # "off"|"minimal"|"low"|"medium"|"high"|"xhigh"
    timeout_seconds: int = 600
    deliver: bool = False             # gateway mode only
    verbose: Optional[str] = None    # cli mode only ("on"|"off")
    extra_cli_args: Optional[List[str]] = None  # cli mode only

    # Retry config (applies to both modes on rate limit / timeout)
    max_retries: int = 3
    retry_delay: float = 8.0         # base delay, multiplied by attempt number
    retry_continuation_prompt: str = (
        "You were interrupted. Please continue or re-answer: {original_prompt}"
    )
```

### `__attrs_post_init__` — Auto-discover token

```python
def __attrs_post_init__(self):
    if self.mode == "gateway" and self.auth_token is None:
        self.auth_token = read_gateway_token_from_pod(
            docker_container=self.docker_container,
            kubectl_namespace=self.kubectl_namespace,
            kubectl_pod=self.kubectl_pod,
            openclaw_config_path=self.openclaw_config_path,
        )
```

### `infer()` dispatch

```python
def infer(self, prompt: str, **kwargs) -> InferenceResult:
    if self.mode == "gateway":
        return asyncio.run(self._ainfer_gateway(prompt, **kwargs))
    else:
        return self._infer_cli(prompt, **kwargs)

async def ainfer(self, prompt: str, **kwargs) -> InferenceResult:
    if self.mode == "gateway":
        return await self._ainfer_gateway(prompt, **kwargs)
    else:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._infer_cli, prompt
        )

async def ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
    if self.mode == "gateway":
        async for chunk in self._stream_gateway(prompt, **kwargs):
            yield chunk
    else:
        # CLI has no streaming — yield full response as single chunk
        result = self._infer_cli(prompt, **kwargs)
        yield result.output
```

### CLI mode: `_infer_cli()`

```python
def _infer_cli(self, prompt: str, **kwargs) -> InferenceResult:
    session_id = kwargs.get("session_id", self.session_id)
    openclaw_cmd = (
        f"OPENCLAW_CONFIG_PATH={self.openclaw_config_path} "
        f"OPENCLAW_STATE_DIR={self.openclaw_state_dir} "
        f"openclaw agent --local --json"
        f" --session-id {shlex.quote(session_id)}"
        f" --message {shlex.quote(prompt)}"
    )
    if self.agent_id:
        openclaw_cmd += f" --agent {shlex.quote(self.agent_id)}"
    if self.thinking:
        openclaw_cmd += f" --thinking {self.thinking}"
    if self.timeout_seconds:
        openclaw_cmd += f" --timeout {self.timeout_seconds}"
    if self.verbose:
        openclaw_cmd += f" --verbose {self.verbose}"
    if self.extra_cli_args:
        openclaw_cmd += " " + " ".join(self.extra_cli_args)

    full_cmd = (
        f"docker exec {self.docker_container} "
        f"kubectl exec -n {self.kubectl_namespace} {self.kubectl_pod} "
        f"-c {self.kubectl_container} -- sh -c {shlex.quote(openclaw_cmd)}"
    )
    stdout, stderr, rc = run_subprocess(full_cmd, timeout=self.timeout_seconds + 60)
    return parse_cli_output(stdout, stderr, rc)
```

### Gateway mode: `_stream_gateway()`

See Section 11.7 for the full WebSocket streaming implementation. Key points:
- Connects with `Origin` header for scope auth
- Waits for `connect.challenge` before sending `connect` RequestFrame
- Uses `client.id="openclaw-control-ui"` + `mode="ui"` for write scope
- Yields incremental text chunks from `delta` events
- Accumulates text (delta events are cumulative, not incremental)

### Retry logic (both modes)

```python
async def _ainfer_with_retry(self, prompt: str, **kwargs) -> InferenceResult:
    original_prompt = prompt
    for attempt in range(1, self.max_retries + 1):
        try:
            if self.mode == "gateway":
                result = await self._ainfer_gateway(prompt, **kwargs)
            else:
                result = self._infer_cli(prompt, **kwargs)
            return result
        except (OpenClawRateLimitError, OpenClawTimeoutError) as e:
            if attempt == self.max_retries:
                raise
            wait = self.retry_delay * attempt
            await asyncio.sleep(wait)
            # Use continuation prompt on retry
            prompt = self.retry_continuation_prompt.format(
                original_prompt=original_prompt
            )
```

---

## 4. Files to Create

```
external/openclaw/
├── __init__.py                 ← Export OpenClawInferencer + exceptions
├── common.py                   ← Shared utilities:
│                                   constants (DEFAULT_DOCKER_CONTAINER, etc.)
│                                   read_gateway_token_from_pod()
│                                   read_gateway_token_from_config()
│                                   extract_json_from_output()
│                                   strip_ansi_codes()
│                                   strip_plugin_warnings()
│                                   check_docker_available()
│                                   run_subprocess()
│                                   GATEWAY_SCOPES, RATE_LIMIT_SIGNALS
│                                   OpenClawError, OpenClawRateLimitError,
│                                   OpenClawTimeoutError, OpenClawNotFoundError
└── openclaw_inferencer.py      ← OpenClawInferencer (single unified class)
                                    _infer_cli() — CLI mode
                                    _stream_gateway() — Gateway mode streaming
                                    _ainfer_gateway() — Gateway mode non-streaming
                                    _ainfer_with_retry() — retry logic
                                    _ws_connect() — WS handshake helper
```

---

## 5. Tests to Create

```
test/agent_foundation/common/inferencers/external/openclaw/
├── __init__.py
└── test_openclaw_inferencer.py
```

### Test classes (all unit — no Docker / no real WS needed):

| Class | Tests |
|---|---|
| `TestInit` | Default init (gateway mode), CLI mode init, auto-token discovery mock |
| `TestCLIMode` | `_build_cli_cmd()` minimal, with all options, shlex quoting, env vars |
| `TestCLIParseOutput` | Valid JSON, multi-payload, strips plugin warnings, fallback plain text, error rc |
| `TestGatewayConnectFrame` | Challenge → connect req → hello-ok frame structure, Origin header |
| `TestGatewayAgentRequest` | agent req frame, idempotencyKey, sessionId, thinking, deliver |
| `TestGatewayStreamParsing` | delta→new chunk, cumulative delta dedup, final→remainder, aborted/error→exception |
| `TestGatewaySessionRestore` | same session_id across two calls |
| `TestRetryLogic` | rate limit → retry with continuation prompt, max retries exceeded |
| `TestModeDispatch` | `infer()` routes to CLI vs gateway, `ainfer_streaming()` CLI yields single chunk |
| `TestTokenDiscovery` | `read_gateway_token_from_pod()` mock subprocess, `read_gateway_token_from_config()` |

---

## 6. Examples to Create

```
examples/agent_foundation/common/inferencers/agentic_inferencers/external/openclaw/
├── example_openclaw_cli_mode.py       ← mode="cli": sync query, multi-turn session
└── example_openclaw_gateway_mode.py   ← mode="gateway": streaming, session restore, retry
```

### Example pattern (follows kiro/claude_code exactly)

Each example:
1. Auto-adds `AgentFoundation/src` and `RichPythonUtils/src` to `sys.path`
2. Creates `OpenClawInferencer` with argparse for `--mode`, `--session-id`, `--thinking`
3. Demonstrates mode-specific features with timing output
4. Prints session_id, model, usage stats

---

## 7. Implementation Order

```
Step 1: common.py
        — Constants, exceptions, utilities
        — read_gateway_token_from_pod() / read_gateway_token_from_config()
        — extract_json_from_output(), strip_ansi_codes(), strip_plugin_warnings()
        — check_docker_available(), run_subprocess()
        — GATEWAY_SCOPES, RATE_LIMIT_SIGNALS
            ↓
Step 2: openclaw_inferencer.py
        — @attrs class OpenClawInferencer
        — __attrs_post_init__ (auto token discovery)
        — _build_cli_cmd(), _infer_cli(), parse_cli_output()
        — _ws_connect() (challenge → connect req → hello-ok)
        — _stream_gateway() (async generator)
        — _ainfer_gateway() (accumulates stream)
        — _ainfer_with_retry() (retry on rate limit)
        — infer() / ainfer() / ainfer_streaming() dispatch methods
            ↓
Step 3: __init__.py
        — Export OpenClawInferencer, OpenClawError, OpenClawRateLimitError
            ↓
Step 4: test_openclaw_inferencer.py
            ↓
Step 5: Examples
            ↓
Step 6: Live integration test
```

---

## 8. Comparison with Reference Implementations

| Feature | ClaudeCodeCliInferencer | OpenClawInferencer (cli mode) | OpenClawInferencer (gateway mode) |
|---|---|---|---|
| Base class | `TerminalSessionInferencerBase` | `StreamingInferencerBase` | `StreamingInferencerBase` |
| Binary location | PATH (`claude`) | Docker/kubectl exec | N/A (pure Python WS client) |
| Session resume | `--resume <id>` | `--session-id <id>` (in-memory) | `sessionId` param (persistent) |
| Streaming | ✅ ndjson events | ❌ Single chunk shim | ✅ True token streaming |
| Session persist | File-backed | ❌ In-memory only | ✅ Gateway `sessions.json` |
| Env vars required | None | `OPENCLAW_CONFIG_PATH`, `OPENCLAW_STATE_DIR` | None (WS only) |
| Rate limit handling | None | Retry with continuation prompt | Retry with continuation prompt |

---

## 9. Known Limitations & Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Rate limit on `claude-opus-4-6` | MEDIUM | Auto-retry + continuation prompt; switch to Haiku via `openclaw.json` |
| CLI mode sessions not persisted | MEDIUM | Documented; use `mode="gateway"` for persistence |
| Gateway must be running for gateway mode | MEDIUM | `check_gateway_reachable()` at init; clear error message |
| `openclaw-control-ui` client ID required for write scope | LOW | Hardcoded in `_ws_connect()`; documented in comments |
| SIGHUP kills gateway (no config reload) | LOW | Documented; never send SIGHUP to openclaw-gateway |
| Haiku model must be set in `openclaw.json` | LOW | Documented; `read_gateway_token_from_pod()` can also patch config |
| sessions dir permissions (sandbox user vs root) | LOW | `chmod 777` on first run; documented |

---

## 10. Target Usage API

```python
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    OpenClawInferencer,
)

# ── Gateway mode (default) — streaming + session restore ──────────────────────

# Simple one-shot (token auto-discovered from pod)
inf = OpenClawInferencer()
result = inf("what tools do you have access to?")
print(result.output)

# True async streaming
inf = OpenClawInferencer(session_id="my-project")
async for chunk in inf.ainfer_streaming("Summarize my open Jira tickets"):
    print(chunk, end="", flush=True)

# Cross-run session restore
inf1 = OpenClawInferencer(session_id="project-alpha")
inf1("My project is called Lighthouse.")   # run 1

inf2 = OpenClawInferencer(session_id="project-alpha")
r = inf2("What is my project called?")     # run 2 — remembers "Lighthouse" ✅
print(r.output)

# With thinking + custom gateway
inf = OpenClawInferencer(
    gateway_url="ws://127.0.0.1:18789",
    auth_token="8ae7f8154bad...",
    session_id="deep-analysis",
    thinking="high",
    timeout_seconds=300,
    max_retries=5,
)
result = inf("Analyze all Jira issues created this week")
print(result.output)

# ── CLI mode — always works, no gateway needed ────────────────────────────────

inf = OpenClawInferencer(mode="cli", session_id="local-test")
result = inf("say hello")
print(result.output)
print(result.model)    # → "claude-opus-4-6" (or haiku if configured)

# ── Classmethods for convenience ──────────────────────────────────────────────

# Auto-discover token from running pod (default behavior)
inf = OpenClawInferencer()  # token auto-discovered

# Read from local config file (if sandbox config is mounted)
inf = OpenClawInferencer.from_config(
    openclaw_json_path="/sandbox/.openclaw/openclaw.json"
)

# Availability checks
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
    check_docker_available,
    check_gateway_reachable,
)
check_docker_available()       # raises OpenClawNotFoundError if docker missing
check_gateway_reachable()      # raises OpenClawNotFoundError if WS unreachable
```


---

## 11. Gateway Mode Deep Dive: WebSocket Protocol & Streaming

### 11.1 Why Gateway Mode Enables What CLI Mode Cannot

| Capability | Phase 1 (`--local` CLI) | Phase 2 (Gateway WS) |
|---|---|---|
| True token streaming | ❌ Blocking subprocess | ✅ `delta` events over WS |
| Cross-run session restore | ❌ In-memory only | ✅ Gateway persists `sessions.json` |
| Session list (`openclaw sessions`) | ❌ Shows 0 | ✅ Stored persistently |
| Rate of connection setup | Fast (subprocess) | Slightly slower (WS handshake) |
| Requires gateway running | ❌ No (--local) | ✅ Yes (always running) |

### 11.2 The Exact Wire Protocol (Fully Reverse-Engineered from Source)

The gateway speaks a custom **JSON-RPC-over-WebSocket** protocol (not standard JSON-RPC 2.0). All frames are JSON objects.

#### Step 1 — WebSocket Upgrade

Connect to `ws://127.0.0.1:18789` (port-forwarded from k3s to host).

**Required HTTP header:** `Origin: http://127.0.0.1:18789` — without this, the server strips write scopes from the token auth and the `agent` method returns `missing scope: operator.write`.

```python
websockets.connect(
    "ws://127.0.0.1:18789",
    additional_headers={"Origin": "http://127.0.0.1:18789"}
)
```

#### Step 2 — Server sends `connect.challenge` event FIRST (client must wait)

**⚠️ Critical discovery from live testing**: The server sends a challenge event BEFORE the client sends anything. The client must wait for this first.

```json
{
  "type": "event",
  "event": "connect.challenge",
  "payload": { "nonce": "38a2d8e9-be8d-4e07-..." }
}
```

The `nonce` is informational — it does not need to be signed for token-mode auth. Simply wait for it, then proceed with the connect request.

#### Step 3 — Client sends `connect` as a `RequestFrame` (not bare params)

**⚠️ Critical discovery**: The connect is sent as a standard `RequestFrame` with `method: "connect"`, NOT as a bare `{"type":"connect",...}` object.

```json
{
  "type": "req",
  "id": "<uuid>",
  "method": "connect",
  "params": {
    "minProtocol": 1,
    "maxProtocol": 10,
    "auth": {
      "token": "<gateway_auth_token>"
    },
    "client": {
      "id": "openclaw-control-ui",
      "version": "1.0.0",
      "platform": "darwin",
      "mode": "ui"
    },
    "caps": [],
    "scopes": ["operator.admin", "operator.read", "operator.write", "operator.approvals", "operator.pairing"],
    "role": "operator"
  }
}
```

**⚠️ Critical discovery — Client ID must be `"openclaw-control-ui"` with `mode: "ui"`**: Using `"cli"` mode with token auth causes the server to strip all write scopes (due to missing device identity). The `"openclaw-control-ui"` + `"ui"` mode + `Origin` header combination is what the web UI uses, and this is what grants full operator scopes. A proper CLI client would need RSA device pairing (`openclaw devices approve`).

Auth options (in `params.auth`, exactly one required):
- `params.auth.token` — the `gateway.auth.token` value from `openclaw.json` (token mode)
- `params.auth.password` — the `gateway.auth.password` value (password mode)

#### Step 4 — `hello-ok` in `ResponseFrame` payload

The response is a standard `ResponseFrame` (not a bare `hello-ok` object):

```json
{
  "type": "res",
  "id": "<same-uuid-as-connect-req>",
  "ok": true,
  "payload": {
    "type": "hello-ok",
    "protocol": 3,
    "server": { "version": "2026.4.2", "connId": "abc123" },
    "features": {
      "methods": ["agent", "chat.send", "chat.history", "sessions.list", ...],
      "events": ["agent", "chat", "tick", "shutdown"]
    },
    "snapshot": { ... }
  }
}
```

Match on `frame["type"] == "res" and frame["id"] == connect_req_id` to identify this frame.

#### Step 5 — Send `agent` request

```json
{
  "type": "req",
  "id": "<uuid>",
  "method": "agent",
  "params": {
    "message": "your prompt here",
    "sessionId": "my-session",
    "idempotencyKey": "<uuid>",
    "thinking": "medium",
    "timeout": 600,
    "deliver": false
  }
}
```

Full `AgentParams` schema (all optional except `message` and `idempotencyKey`):

| Field | Type | Description |
|---|---|---|
| `message` | `string` (required) | The user prompt |
| `idempotencyKey` | `string` (required) | UUID for dedup |
| `sessionId` | `string` | Session ID — gateway creates if new, resumes if existing |
| `agentId` | `string` | Agent ID from config |
| `thinking` | `string` | `off\|minimal\|low\|medium\|high\|xhigh` |
| `timeout` | `integer` | Seconds (default 600) |
| `deliver` | `boolean` | Whether to deliver to channel (default false) |
| `extraSystemPrompt` | `string` | Append to system prompt |
| `lane` | `string` | Routing lane |

#### Step 6 — `res` frame (immediate ack from server)

```json
{
  "type": "res",
  "id": "<same-uuid>",
  "ok": true,
  "payload": { "runId": "run-abc123" }
}
```

This is just an **acknowledgment** — the run has been queued. The actual response comes as streaming events.

#### Step 7 — `event` frames with `agent` type (streaming)

The server sends a stream of `event` frames of type `"agent"`:

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "run-abc123",
    "seq": 0,
    "stream": "text-0",
    "ts": 1744685923456,
    "data": {
      "state": "delta",
      "message": {
        "content": [{ "type": "text", "text": "Hello" }]
      }
    }
  }
}
```

**`data.state` values:**

| `state` | Meaning | Action |
|---|---|---|
| `"delta"` | Incremental text chunk | Extract `data.message.content[].text` → yield to caller |
| `"final"` | Last event, full message | Extract full text, mark done |
| `"aborted"` | Run was aborted | Raise exception |
| `"error"` | Run failed | Raise exception with `data.errorMessage` |

**Text extraction from delta/final:**
```python
# From data.message.content
for item in data["message"]["content"]:
    if item["type"] == "text":
        yield item["text"]
```

**Session key:** The gateway uses `sessionId` to route to the correct conversation history. Same `sessionId` = continued conversation. Different `sessionId` = new conversation. Sessions persist across gateway restarts (stored in `sessions.json`).

#### Step 8 — Disconnect

Close the WebSocket (code 1000) after receiving the `"final"` state event.

### 11.3 Alternative: `chat.send` Method (Webchat API)

The gateway also supports `chat.send` (the webchat UI's method):

```json
{
  "type": "req",
  "id": "<uuid>",
  "method": "chat.send",
  "params": {
    "sessionKey": "session:my-session",
    "message": "your prompt",
    "idempotencyKey": "<uuid>"
  }
}
```

Response events arrive on the `"chat"` event type. The `"agent"` method is preferred for programmatic use (full parameter control, thinking level, timeout).

### 11.4 Base Class Pattern

`OpenClawInferencer` follows the **`RovoDevServeInferencer`** pattern for gateway mode:

| Feature | RovoDevServeInferencer | OpenClawInferencer (gateway) |
|---|---|---|
| Base class | `StreamingInferencerBase` | `StreamingInferencerBase` |
| Transport | HTTP SSE (streaming) | WebSocket JSON-RPC |
| Streaming events | SSE `text_delta` chunks | WS `event.agent.data.state=="delta"` |
| Session handling | `POST /v3/reset` | `sessionId` in `agent` params |
| Auth | None (local) | `authToken` in connect frame |
| Requires daemon | `acli rovodev serve` | `openclaw gateway` (already running in pod) |

### 11.5 Token Auto-Discovery (in `common.py`)

```python
def read_gateway_token_from_pod(
    docker_container: str = "openshell-cluster-openshell",
    kubectl_namespace: str = "openshell",
    kubectl_pod: str = "atlassian-openclaw-gateway",
    openclaw_config_path: str = "/sandbox/.openclaw/openclaw.json",
) -> str:
    """Read gateway auth token from the running pod via docker exec."""
    result = subprocess.run([
        "docker", "exec", docker_container,
        "kubectl", "exec", "-n", kubectl_namespace, kubectl_pod,
        "-c", "agent", "--",
        "python3", "-c",
        f"import json; d=json.load(open('{openclaw_config_path}')); "
        "print(d['gateway']['auth']['token'])"
    ], capture_output=True, text=True, timeout=10)
    token = result.stdout.strip()
    if not token:
        raise OpenClawNotFoundError("Could not read gateway auth token from pod")
    return token

def read_gateway_token_from_config(
    openclaw_json_path: str = "/sandbox/.openclaw/openclaw.json",
) -> str:
    """Read gateway auth token from a local openclaw.json file."""
    config = json.loads(Path(openclaw_json_path).read_text())
    return config["gateway"]["auth"]["token"]
```

### 11.6 `_ws_connect()` Helper (inside `OpenClawInferencer`)

```python
async def _ws_connect(self) -> websockets.WebSocketClientProtocol:
    """Open WebSocket, complete handshake, return authenticated connection."""
    origin = self.gateway_url.replace("ws://", "http://").replace("wss://", "https://")
    ws = await websockets.connect(
        self.gateway_url,
        max_size=25 * 1024 * 1024,
        additional_headers={"Origin": origin},
    )
    # 1. Wait for server challenge (server speaks first)
    challenge = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
    assert challenge.get("event") == "connect.challenge"

    # 2. Send connect as RequestFrame (NOT a bare connect frame)
    connect_id = str(uuid.uuid4())
    await ws.send(json.dumps({
        "type": "req", "id": connect_id, "method": "connect",
        "params": {
            "minProtocol": 1, "maxProtocol": 10,
            "auth": {"token": self.auth_token},
            "client": {
                "id": "openclaw-control-ui",  # required for write scope with token auth
                "version": "1.0.0",
                "platform": "python",
                "mode": "ui",                 # "cli" mode strips write scopes (no device identity)
            },
            "caps": [], "scopes": GATEWAY_SCOPES, "role": "operator",
        }
    }))

    # 3. Wait for hello-ok in ResponseFrame payload
    while True:
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if frame.get("type") == "res" and frame.get("id") == connect_id:
            if not frame.get("ok"):
                await ws.close()
                raise OpenClawError(f"Gateway connect failed: {frame.get('error')}")
            return ws


### 11.7 Async Streaming Implementation

```python
GATEWAY_SCOPES = [
    "operator.admin", "operator.read", "operator.write",
    "operator.approvals", "operator.pairing",
]

async def ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
    import uuid, websockets, json

    session_id = kwargs.get("session_id", self.session_id)
    idempotency_key = str(uuid.uuid4())
    connect_id = str(uuid.uuid4())
    req_id = str(uuid.uuid4())
    accumulated = ""

    # Origin header required — grants full operator write scopes for token auth
    async with websockets.connect(
        self.gateway_url,
        max_size=25 * 1024 * 1024,
        additional_headers={"Origin": self.gateway_url.replace("ws://", "http://").replace("wss://", "https://")}
    ) as ws:

        # Step 1: wait for connect.challenge from server
        challenge = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert challenge.get("event") == "connect.challenge", \
            f"Expected connect.challenge, got: {challenge.get('event')}"

        # Step 2: send connect as RequestFrame (method="connect")
        await ws.send(json.dumps({
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": {
                "minProtocol": 1,
                "maxProtocol": 10,
                "auth": {"token": self.auth_token} if self.auth_token else {"password": self.auth_password},
                "client": {
                    "id": "openclaw-control-ui",   # must be this for write scope with token auth
                    "version": "1.0.0",
                    "platform": "python",
                    "mode": "ui",                  # must be "ui" not "cli"
                },
                "caps": [],
                "scopes": GATEWAY_SCOPES,
                "role": "operator",
            }
        }))

        # Step 3: receive hello-ok in ResponseFrame payload
        while True:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if frame.get("type") == "res" and frame.get("id") == connect_id:
                if not frame.get("ok"):
                    raise RuntimeError(f"OpenClaw connect failed: {frame.get('error')}")
                # Connection established
                break

        # Step 4: send agent request
        await ws.send(json.dumps({
            "type": "req",
            "id": req_id,
            "method": "agent",
            "params": {
                "message": prompt,
                "sessionId": session_id,
                "idempotencyKey": idempotency_key,
                **( {"thinking": self.thinking} if self.thinking else {} ),
                "timeout": self.timeout_seconds,
                "deliver": self.deliver,
            }
        }))

        # Step 5: receive streaming events
        async for raw in ws:
            frame = json.loads(raw)
            ftype = frame.get("type")
            fevent = frame.get("event", "")

            # Agent req ack — check for immediate rejection
            if ftype == "res" and frame.get("id") == req_id:
                if not frame.get("ok"):
                    raise RuntimeError(f"OpenClaw agent rejected: {frame.get('error')}")
                continue

            # Skip non-agent events (tick, etc.)
            if ftype != "event" or fevent != "agent":
                continue

            payload = frame.get("payload", {})
            data = payload.get("data", {})
            state = data.get("state")

            if state == "delta":
                for item in data.get("message", {}).get("content", []):
                    if item.get("type") == "text":
                        chunk = item["text"]
                        # delta events send CUMULATIVE text — yield only the new part
                        if chunk.startswith(accumulated):
                            new_chunk = chunk[len(accumulated):]
                            accumulated = chunk
                        else:
                            new_chunk = chunk
                            accumulated += chunk
                        if new_chunk:
                            yield new_chunk

            elif state == "final":
                # Yield any remaining text not yet yielded
                for item in data.get("message", {}).get("content", []):
                    if item.get("type") == "text":
                        final_text = item["text"]
                        if final_text.startswith(accumulated):
                            remainder = final_text[len(accumulated):]
                            if remainder:
                                yield remainder
                break

            elif state in ("aborted", "error"):
                raise RuntimeError(
                    f"OpenClaw agent {state}: {data.get('errorMessage', 'unknown error')}"
                )
```

### 11.8 Session Restore (Cross-Run)

With the gateway, **same `session_id` = automatic resume**. The gateway stores conversation history in `sessions.json` inside the gateway pod. When you reconnect with the same `session_id`, the full history is loaded from disk.

```python
# Session A — first run (token auto-discovered from pod)
inf = OpenClawInferencer(session_id="project-alpha")
async for chunk in inf.ainfer_streaming("My project is called Lighthouse"):
    print(chunk, end="")

# Session A — second run (different process, days later)
inf2 = OpenClawInferencer(session_id="project-alpha")
async for chunk in inf2.ainfer_streaming("What is my project called?"):
    print(chunk, end="")
# → "Your project is called Lighthouse."  ✅ Session restored!
```

### 11.9 Notes on Session Permissions

After gateway restart, the `/sandbox/.openclaw/agents/main/sessions/` directory may be owned by `root` (from `--local` CLI runs) but the gateway process runs as `sandbox` user. Fix with:

```bash
docker exec openshell-cluster-openshell kubectl exec -n openshell atlassian-openclaw-gateway -- \
  sh -c "chmod 777 /sandbox/.openclaw/agents/main/sessions/ && chmod 666 /sandbox/.openclaw/agents/main/sessions/*.jsonl 2>/dev/null || true"
```

### 11.10 ⚠️ Never Send SIGHUP to `openclaw-gateway`

The gateway process does **not** support `SIGHUP` for config reload — it terminates the process instead. To change model config, edit `openclaw.json` and restart via `./run.sh start`.

---

## 12. All Files to Create

```
external/openclaw/
├── __init__.py               ← Export OpenClawInferencer + exceptions
├── common.py                 ← Constants, utilities, token helpers, exceptions
└── openclaw_inferencer.py    ← Unified class (CLI mode + Gateway mode)

test/agent_foundation/common/inferencers/external/openclaw/
├── __init__.py
└── test_openclaw_inferencer.py   ← All unit tests (CLI + gateway, mock subprocess + mock WS)

examples/agent_foundation/common/inferencers/agentic_inferencers/external/openclaw/
├── example_openclaw_cli_mode.py       ← mode="cli": sync query, multi-turn
└── example_openclaw_gateway_mode.py   ← mode="gateway": streaming, session restore, retry
```

---

## 13. Full Implementation Order

```
Step 1: common.py
        — Constants: DEFAULT_DOCKER_CONTAINER, DEFAULT_KUBECTL_*, GATEWAY_SCOPES,
          RATE_LIMIT_SIGNALS, PROTOCOL_VERSION_MIN/MAX
        — Exceptions: OpenClawError, OpenClawRateLimitError,
          OpenClawTimeoutError, OpenClawNotFoundError
        — Utilities: extract_json_from_output(), strip_ansi_codes(),
          strip_plugin_warnings(), run_subprocess()
        — Token helpers: read_gateway_token_from_pod(),
          read_gateway_token_from_config()
        — Checks: check_docker_available(), check_gateway_reachable()
            ↓
Step 2: openclaw_inferencer.py
        — @attrs class OpenClawInferencer(StreamingInferencerBase)
        — __attrs_post_init__: auto-discover token if gateway mode + no token
        — CLI mode: _build_cli_cmd(), _infer_cli(), parse_cli_output()
        — Gateway mode: _ws_connect(), _stream_gateway(), _ainfer_gateway()
        — Retry: _ainfer_with_retry() (rate limit + timeout detection)
        — Dispatch: infer(), ainfer(), ainfer_streaming()
        — Classmethod: from_config(openclaw_json_path)
            ↓
Step 3: __init__.py
        — Export OpenClawInferencer, OpenClawError, OpenClawRateLimitError
            ↓
Step 4: test_openclaw_inferencer.py
        — TestInit, TestCLIMode, TestCLIParseOutput
        — TestGatewayConnectFrame, TestGatewayAgentRequest
        — TestGatewayStreamParsing, TestGatewaySessionRestore
        — TestRetryLogic, TestModeDispatch, TestTokenDiscovery
            ↓
Step 5: Examples
        — example_openclaw_cli_mode.py
        — example_openclaw_gateway_mode.py
            ↓
Step 6: Live integration test with running Docker cluster
```
