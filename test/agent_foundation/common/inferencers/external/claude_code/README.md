# Claude Code Inferencer Tests

This directory contains tests for `ClaudeCodeInferencer`, which wraps the Claude Code SDK for agentic tasks.

## Test Files

| File | Type | Description | Status |
|------|------|-------------|--------|
| `test_claude_code_inferencer.py` | Unit | Mock-based unit tests | ✅ |
| `test_claude_code_inferencer_real.py` | Integration | **Real SDK calls** | ✅ **ALL 5 PASSED** |
| `test_claude_code_quick.py` | Sanity | Quick import/init checks | ✅ |

---

## Quick Start

### Run Real Integration Tests (Recommended)

```bash
# All tests (sync + async) with default query
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real

# With custom query
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --query "What is Python?"

# Sync mode only
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode sync

# Async mode only
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode async
```

### Run Unit Tests

```bash
buck2 test @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer
```

---

## Test Coverage (Real Integration Tests)

| Test | Mode | What It Tests | Status |
|------|------|---------------|--------|
| **Sync Single Call** | sync | `inferencer(query)` - basic callable syntax | ✅ PASSED |
| **Sync Multiple Calls** | sync | Two consecutive `inferencer(q1); inferencer(q2)` - tests **v4 stale-loop detection** | ✅ PASSED |
| **Async Single Call** | async | `await inferencer.ainfer(query)` | ✅ PASSED |
| **Async Context Manager** | async | `async with inferencer:` - persistent connection, multiple queries | ✅ PASSED |
| **SDK Response Format** | async | `return_sdk_response=True` returns `SDKInferencerResponse` object | ✅ PASSED |

---

## Example Output

```
============================================================
CLAUDE CODE INFERENCER - REAL INTEGRATION TESTS
============================================================

Query: What is 2+2? Just answer with the number.
Mode: all

============================================================
TEST: Sync Single Call
============================================================
✓ Created inferencer

Sending SYNC query: 'What is 2+2? Just answer with the number.'

✓ Got response in 8.13s:
  Response type: <class 'str'>
  Response length: 1 chars
----------------------------------------
4
----------------------------------------

✅ SYNC TEST PASSED!

============================================================
TEST: Sync Multiple Calls (Stale-Loop Detection)
============================================================

First SYNC call: 'What is 2+2?'
✓ Response 1 in 7.62s: 4...

Second SYNC call: 'What is 3+3?'
✓ Response 2 in 6.96s: 6...

✅ SYNC MULTIPLE CALLS PASSED!

============================================================
TEST: Async Context Manager (Multiple Calls, Persistent Connection)
============================================================

Using async context manager (single connection, multiple queries)...
✓ Connected via context manager

First ASYNC query: 'What is 2+2?'
✓ Response 1 in 2.42s: 4...

Second ASYNC query: 'What is 3+3?'
✓ Response 2 in 2.67s: 6...
✓ Disconnected via context manager

✅ ASYNC CONTEXT MANAGER PASSED!

============================================================
TEST SUMMARY
============================================================
  Sync Single Call: ✅ PASSED
  Sync Multiple Calls: ✅ PASSED
  Async Single Call: ✅ PASSED
  Async Context Manager: ✅ PASSED
  SDK Response Format: ✅ PASSED

Total: 5 passed, 0 failed
```

---

## Implementation Details

### v4 Stale-Loop Detection

When using sync mode (`inferencer(query)`), each call uses `asyncio.run()` which closes the event loop afterward. The v4 implementation detects this and reconnects automatically:

```python
# In _infer():
if self._connected_loop is not None and self._connected_loop.is_closed():
    logger.debug("Previous event loop is closed — clearing stale client")
    self._client = None
    self._disconnect_fn = None
    self._connected_loop = None
```

This allows multiple consecutive sync calls to work correctly.

### Async Context Manager

For optimal performance with multiple queries, use the async context manager:

```python
async with ClaudeCodeInferencer(root_folder="/tmp") as inferencer:
    # Single connection, multiple queries
    result1 = await inferencer.ainfer("Query 1")
    result2 = await inferencer.ainfer("Query 2")
    result3 = await inferencer.ainfer("Query 3")
# Automatic disconnect
```

This maintains a single connection across all queries, significantly reducing latency (2-3s per query vs 7-8s for new connections).

### SDKInferencerResponse

When `return_sdk_response=True`, returns a structured response:

```python
@attrs
class SDKInferencerResponse:
    content: str           # The actual response text
    session_id: str        # SDK session identifier
    tool_uses: int         # Number of tool uses in response
    tokens_received: int   # Token count
    raw_response: Any      # Raw SDK response object
```

---

## Dependencies

The BUCK target includes:
```python
deps = [
    "//rankevolve/src/agentic_foundation:agentic_foundation",
    "//rankevolve/src/utils:utils",
    "fbsource//third-party/pypi/claude-agent-sdk:claude-agent-sdk",  # Required!
]
```

---

## Troubleshooting

### "Transport endpoint is not connected"

Use `--prefer-remote` flag:
```bash
buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real
```

### "No module named 'claude_agent_sdk'"

Ensure the SDK dependency is in the BUCK file.

### Response latency

- **Sync mode:** ~7-8s per query (new connection each time)
- **Async context manager:** ~2-3s per query (reuses connection)

---

*Last updated: February 2026*
