"""Shared fixtures, markers, skip logic, and helpers for real integration tests.

These tests invoke actual ``claude`` and ``kiro-cli`` CLI tools on the machine,
spawning real subprocesses that call real models. They are slow, cost money,
and require the CLI tools to be installed and authenticated.
"""

import glob
import hashlib
import os
import subprocess
import time
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Configurable timeout (default 120s, override via env var)
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = int(os.environ.get("INTEGRATION_TEST_TIMEOUT_SECONDS", "120"))


# ---------------------------------------------------------------------------
# Pytest marker registration
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: real CLI integration tests (slow, costs money)"
    )


# ---------------------------------------------------------------------------
# CLI availability detection
# ---------------------------------------------------------------------------
def _cli_available(command: str) -> bool:
    """Check if a CLI tool is available and functional via ``--version``."""
    try:
        result = subprocess.run(
            f"{command} --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


CLAUDE_AVAILABLE = _cli_available("claude")
KIRO_AVAILABLE = _cli_available("kiro-cli")

skip_claude = pytest.mark.skipif(
    not CLAUDE_AVAILABLE,
    reason="claude CLI not available (not in PATH or non-zero exit)",
)
skip_kiro = pytest.mark.skipif(
    not KIRO_AVAILABLE,
    reason="kiro-cli CLI not available (not in PATH or non-zero exit)",
)
skip_both = pytest.mark.skipif(
    not (CLAUDE_AVAILABLE and KIRO_AVAILABLE),
    reason="Both claude and kiro-cli required",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide isolated temp directories for cache, checkpoint, and workspace."""
    dirs = {
        "cache": tmp_path / "cache",
        "checkpoint": tmp_path / "checkpoint",
        "workspace": tmp_path / "workspace",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    return dirs


@pytest.fixture
def claude_inferencer(tmp_workspace):
    """Create a :class:`ClaudeCodeCliInferencer` with cache enabled."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
        ClaudeCodeCliInferencer,
    )

    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=60,
    )
    # Set permission_mode if the attribute exists on the class
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    return ClaudeCodeCliInferencer(**kwargs)


@pytest.fixture
def kiro_inferencer(tmp_workspace):
    """Create a :class:`KiroCliInferencer` with cache enabled."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
        KiroCliInferencer,
    )

    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="auto",
        resume_with_saved_results=True,
        idle_timeout_seconds=60,
    )
    # Set trust_mode if the attribute exists on the class
    if hasattr(KiroCliInferencer, "trust_mode"):
        kwargs["trust_mode"] = "all"
    return KiroCliInferencer(**kwargs)


# ---------------------------------------------------------------------------
# Timing-based verification helpers
# ---------------------------------------------------------------------------
def assert_cached_skip(start_time: float, max_seconds: float = 2.0):
    """Assert that execution was fast enough to be a cache hit."""
    elapsed = time.monotonic() - start_time
    assert elapsed < max_seconds, (
        f"Expected cache hit (<{max_seconds}s), took {elapsed:.1f}s"
    )


def assert_real_call(start_time: float, min_seconds: float = 3.0):
    """Assert that execution was slow enough to be a real CLI call."""
    elapsed = time.monotonic() - start_time
    assert elapsed >= min_seconds, (
        f"Expected real CLI call (>{min_seconds}s), took {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# Cache file inspection helpers
# ---------------------------------------------------------------------------
def count_cache_files(cache_folder: str, prompt: str, class_name: str) -> int:
    """Count cache files matching a prompt hash for a given inferencer class."""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return len(
        glob.glob(os.path.join(cache_folder, class_name, "*", f"stream_*_{h}.txt"))
    )


def read_latest_cache(
    cache_folder: str, prompt: str, class_name: str
) -> Optional[str]:
    """Read the most recent cache file for a prompt. Returns ``None`` if absent."""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    matches = glob.glob(
        os.path.join(cache_folder, class_name, "*", f"stream_*_{h}.txt")
    )
    if not matches:
        return None
    latest = max(matches, key=os.path.getmtime)
    with open(latest) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tier C coding puzzle prompt constants
# ---------------------------------------------------------------------------
PUZZLE_PROMPT = """\
Write a Python module with these three functions. Each has subtle edge cases \
that most implementations get wrong:

1. `deep_flatten(nested)` - Flatten arbitrarily nested iterables (lists, tuples, \
generators) into a single list. Must handle: circular references (raise ValueError), \
strings (don't iterate into characters), dict (iterate keys only), \
and generators (consume only once, don't restart).

2. `lru_cache_with_ttl(maxsize, ttl_seconds)` - Decorator that combines LRU eviction \
with time-based expiration. Must handle: thread safety, unhashable arguments \
(raise TypeError with clear message), ttl=0 means no caching, negative ttl raises \
ValueError, and cache.clear() method on the decorated function.

3. `retry_with_backoff(max_retries, base_delay, max_delay, jitter)` - Decorator for \
retrying failed calls with exponential backoff. Must handle: async functions \
(detect and use await), generators (raise TypeError - can't retry a generator), \
base_delay=0 means no delay, and the decorated function must expose \
.retry_stats (dict with 'attempts', 'total_delay', 'last_exception').

Include comprehensive docstrings, type hints, and handle all edge cases listed above.
"""


# ---------------------------------------------------------------------------
# Tier C verification helpers
# ---------------------------------------------------------------------------
def _exec_generated(code: str) -> dict:
    """Execute generated code in an isolated namespace and return it."""
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns


def verify_deep_flatten(code: str) -> None:
    """Verify ``deep_flatten`` from generated code handles key edge cases."""
    ns = _exec_generated(code)
    fn = ns["deep_flatten"]

    # Basic nesting
    assert fn([1, [2, [3, [4]]]]) == [1, 2, 3, 4]

    # Strings are NOT iterated into characters
    assert fn("hello") == ["hello"]

    # Dict iterates keys only
    result = fn({"a": 1, "b": 2})
    assert set(result) == {"a", "b"}

    # Circular reference detection
    circular: list = [1]
    circular.append(circular)
    try:
        fn(circular)
        raise AssertionError("Expected ValueError for circular reference")
    except ValueError:
        pass


def verify_lru_cache_with_ttl(code: str) -> None:
    """Verify ``lru_cache_with_ttl`` from generated code handles key edge cases."""
    ns = _exec_generated(code)
    decorator = ns["lru_cache_with_ttl"]

    # Basic caching works
    call_count = 0

    @decorator(maxsize=2, ttl_seconds=10)
    def add_one(x):
        nonlocal call_count
        call_count += 1
        return x + 1

    assert add_one(1) == 2
    assert add_one(1) == 2  # cached
    assert call_count == 1

    # Negative ttl raises ValueError
    try:
        @decorator(maxsize=2, ttl_seconds=-1)
        def bad_fn(x):
            return x
        raise AssertionError("Expected ValueError for negative ttl")
    except ValueError:
        pass

    # Unhashable arguments raise TypeError
    @decorator(maxsize=2, ttl_seconds=10)
    def identity(x):
        return x

    try:
        identity([1, 2, 3])  # list is unhashable
        raise AssertionError("Expected TypeError for unhashable argument")
    except TypeError:
        pass


def verify_retry_with_backoff(code: str) -> None:
    """Verify ``retry_with_backoff`` from generated code handles key edge cases."""
    ns = _exec_generated(code)
    decorator = ns["retry_with_backoff"]

    # Basic retry works
    attempt_count = 0

    @decorator(max_retries=3, base_delay=0, max_delay=1, jitter=False)
    def flaky():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert flaky.retry_stats["attempts"] == 3

    # Generator raises TypeError
    try:

        @decorator(max_retries=3, base_delay=0, max_delay=1, jitter=False)
        def gen():
            yield 1

        gen()
        raise AssertionError("Expected TypeError for generator function")
    except TypeError:
        pass
