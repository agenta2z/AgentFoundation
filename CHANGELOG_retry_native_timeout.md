# Changelog: retry-native-timeout

## Behavioral Change: `1 + max_retry` Attempts

`InferencerBase.fallback_mode` defaults to `FallbackMode.ON_FIRST_FAILURE`.

**Async path** (`for attempt in range(max_retry)`):
- Before: `Inf(max_retry=3)` → 3 attempts total.
- After: `Inf(max_retry=3)` → 1 primary + 3 recovery = 4 total.

**Sync path** (`while True` + `attempts >= max_retry: break` = initial + max_retry):
- Before: `Inf(max_retry=3)` → 4 attempts total.
- After: `Inf(max_retry=3)` → 1 primary + 4 recovery = 5 total.

**Restoring old behavior:**
```python
inf = SomeInferencer(max_retry=3, fallback_mode=FallbackMode.NEVER)
```

**Per-call override:**
```python
result = inf.infer("prompt", fallback_mode=FallbackMode.NEVER)
```

## New Features

- `total_timeout` and `attempt_timeout` parameters on both `execute_with_retry` and `async_execute_with_retry`
- `FallbackMode` enum (`NEVER`, `ON_EXHAUSTED`, `ON_FIRST_FAILURE`) for fallback chain control
- `fallback_func`, `fallback_on_exceptions`, `on_fallback_callback` parameters on retry helpers
- `attempt_timeout_seconds`, `fallback_inferencer`, `fallback_mode` attributes on `InferencerBase`
- `_infer_recovery` / `_ainfer_recovery` overridable recovery methods on `InferencerBase`
- `FallbackInferMode` enum (`CONTINUE`, `REFERENCE`, `RESTART`) on `StreamingInferencerBase`
- Cache-aware and session-aware recovery in `StreamingInferencerBase`
- `_current_fallback_state` ContextVar for per-task-safe fallback state transport

## Cost Impact

For paid APIs, the extra attempt per failure is a real cost increase. Audit cost/quota assumptions. Use `fallback_mode=FallbackMode.NEVER` to restore exact previous behavior.
