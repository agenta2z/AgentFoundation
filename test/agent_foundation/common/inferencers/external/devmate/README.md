# Devmate SDK Inferencer Tests

This directory contains tests for `DevmateSDKInferencer`, which wraps the Devmate SDK for agentic tasks.

## ⚠️ IMPORTANT: Environment Requirements

**The Devmate SDK requires a specific runtime environment that is NOT available in plain `buck2 run` contexts.**

The SDK spawns a Rust bridge process (`srconveyor devmate-cli-bridge`) that requires:
- Proper authentication tokens (crypto auth tokens)
- Access to Devmate backend services
- Running from a Devmate-enabled context (VSCode with Devmate extension, or within Devmate CLI)

**If you see this error, it's an environment issue, NOT a code bug:**
```
BridgeStartupError: Bridge process produced no output for 60.0 seconds during startup.
This may indicate an error in the configuration or prompt file.
```

### How to Run Devmate Tests

**Option 1: Run from VSCode with Devmate Extension**
- Open VSCode with the Devmate extension enabled
- Run the test from within that environment

**Option 2: Use Devmate CLI Session**
- Start a Devmate CLI session first: `devmate`
- Then run the tests from within that session

**Option 3: Use Mock-Based Unit Tests (Recommended for CI)**
```bash
buck2 test --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer
```

## Test Files

| File | Type | Description | Status |
|------|------|-------------|--------|
| `test_devmate_sdk_inferencer.py` | Unit | Mock-based unit tests | ✅ Works anywhere |
| `test_devmate_sdk_inferencer_real.py` | Integration | Real SDK calls | ⚠️ Requires Devmate environment |

---

## Quick Start

### Run Real Integration Tests

```bash
# All tests (sync + async) with default query
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer_real

# With custom query
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer_real -- --query "What is Python?"

# Sync mode only
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer_real -- --mode sync

# Async mode only
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer_real -- --mode async
```

### Run Unit Tests

```bash
buck2 test @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer
```

---

## Test Coverage (Real Integration Tests)

| Test | Mode | What It Tests |
|------|------|---------------|
| **Sync Single Call** | sync | `inferencer(query)` - basic callable syntax via async bridge |
| **Sync Multiple Calls** | sync | Two consecutive sync calls |
| **Async Single Call** | async | `await inferencer.ainfer(query)` |
| **Async Multiple Calls** | async | Multiple async calls |
| **SDK Response Format** | async | `return_sdk_response=True` returns `SDKInferencerResponse` |

---

## Implementation Details

### Per-Call Client Model

Unlike ClaudeCodeInferencer (which maintains a persistent connection), DevmateSDKInferencer creates a **fresh SDK client for each query**:

```python
async def _ainfer(self, inference_input, inference_config=None, **kwargs):
    # Creates new client for each call
    client = DevmateSDKClient(
        config_file_path=self.config_file_path,
        usecase=self.usecase,
        config_vars=config_vars,
        repo_root=repo_root_path,
    )
    # ... execute query ...
```

### Event-Driven Response Collection

The inferencer uses SDK event handlers to collect responses:

```python
async def on_session(session):
    """Handle session status changes."""
    ...

async def on_action(action):
    """Collect response text from action outputs."""
    ...

async def on_step(step):
    """Track step progress."""
    ...

async def on_error(error):
    """Handle SDK errors."""
    ...
```

### v3 Fixes Applied

1. **Path wrapper (v3 FIX #3):** SDK expects `Path | None`, not `str`
   ```python
   repo_root_path = Path(self.root_folder) if self.root_folder else None
   ```

2. **Local session_id (v3 FIX #9):** Uses closure-local variable for concurrent safety
   ```python
   local_session_id = None  # Local to this call, not self._session_id
   ```

### Sync Bridge

The sync `_infer` method bridges to async via `_run_async`:

```python
def _infer(self, inference_input, inference_config=None, **kwargs):
    """Sync wrapper for async _ainfer()."""
    from rich_python_utils.common_utils.async_function_helper import _run_async
    return _run_async(self._ainfer(inference_input, inference_config, **kwargs))
```

---

## Dependencies

The BUCK target includes:
```python
deps = [
    "//rankevolve/src/agentic_foundation:agentic_foundation",
    "//rankevolve/src/utils:utils",
    "//devai/devmate_sdk/python:devmate_python_sdk",  # Required!
]
```

---

## Known Issues

### Bridge Startup Timeout

The Devmate SDK requires specific environment configuration. You may see:

```
Bridge startup timed out after 60.0 seconds of silence.
This may indicate an error in the configuration or prompt file.
```

**Causes:**
- Missing SDK environment configuration
- Network connectivity issues to Devmate services
- Invalid configuration parameters

**Solutions:**
1. Verify Devmate SDK is properly configured
2. Check network access to Devmate services
3. Increase `timeout_seconds` if needed
4. Check SDK-specific logs for details

---

## Troubleshooting

### "Transport endpoint is not connected"

Use `--prefer-remote` flag:
```bash
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_devmate_sdk_inferencer_real
```

### "Can't instantiate abstract class"

Ensure `_infer` method is implemented in the inferencer class.

### Empty responses after timeout

Check:
1. SDK configuration (`config_file_path`, `usecase`)
2. Network connectivity
3. Timeout values (`timeout_seconds`, `idle_timeout_seconds`)

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_folder` | `None` | Working directory for Devmate agent |
| `config_file_path` | `"freeform"` | Path to config or "freeform" mode |
| `usecase` | `"sdk_inferencer"` | Usecase identifier |
| `model_name` | `"claude-sonnet-4-5"` | Model to use |
| `timeout_seconds` | `1800` (30 min) | Max session time |
| `idle_timeout_seconds` | `600` (10 min) | Max idle time |
| `config_vars` | `{}` | Additional config variables |

---

*Last updated: February 2026*
