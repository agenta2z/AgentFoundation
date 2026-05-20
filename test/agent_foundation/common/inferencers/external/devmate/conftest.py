"""Shared fixtures + async-test wiring for devmate tests.

Provides:
- Auto-wrapping of ``async def test_*`` functions so they run via
  ``asyncio.run()`` without needing ``pytest-asyncio`` (which is at
  version 1.3.0 in the third-party tree and doesn't handle modern
  async test patterns).
- Skip markers for tests that require external dependencies which
  aren't always available outside of Buck builds (notably the
  ``devai.devmate_sdk`` Python package).
- Skip marker for tests that invoke real ``devmate`` or ``claude``
  CLIs (slow, env-dependent — gated behind ``DEVMATE_RUN_REAL`` /
  ``CLAUDE_RUN_REAL`` env vars so unit-test runs stay fast).

Mirrors the pattern used by ``../claude_code/conftest.py``.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import inspect
import os

import pytest

DEFAULT_QUERY = "Reply with one word: hello"


@pytest.fixture
def query() -> str:
    """Default query for devmate real integration tests."""
    return DEFAULT_QUERY


def _has_devai_sdk() -> bool:
    """Return True if ``devai.devmate_sdk.python.devmate_client`` is importable.

    ``find_spec`` raises ``ModuleNotFoundError`` when a parent package is
    missing (rather than returning ``None``), so we catch that explicitly.
    """
    try:
        return importlib.util.find_spec("devai.devmate_sdk.python.devmate_client") is not None
    except ModuleNotFoundError:
        return False


# Test files that ALWAYS contain real-devmate calls; every test in them is
# skipped unless ``DEVMATE_RUN_REAL=1`` is set. Each call typically costs
# 15+ seconds, which would push CI time into the minutes.
_REAL_DEVMATE_FILE_MARKERS: tuple[str, ...] = (
    "test_devmate_cli_inferencer_real.py",
    "test_devmate_sdk_inferencer_real.py",
    "test_model_id_real_inference.py",
)

# Individual test names that hit a real ``devmate`` CLI / SDK in mixed files;
# skipped unless ``DEVMATE_RUN_REAL=1`` is set.
_REAL_DEVMATE_TESTS: tuple[str, ...] = (
    "test_cross_session_cli",
    "test_cross_session_sdk",
)

# Test names that hit a real Claude Code CLI; skipped unless CLAUDE_RUN_REAL is set.
_REAL_CLAUDE_TESTS: tuple[str, ...] = (
    "test_cross_session_claude",
)


def pytest_collection_modifyitems(config, items):
    """Wrap async tests + apply skip markers based on environment."""
    skip_devai = pytest.mark.skip(
        reason="devai.devmate_sdk Python package not available (Buck-only dep)"
    )
    skip_real_devmate = pytest.mark.skip(
        reason="real devmate CLI test; set DEVMATE_RUN_REAL=1 to opt in"
    )
    skip_real_claude = pytest.mark.skip(
        reason="real claude CLI test; set CLAUDE_RUN_REAL=1 to opt in"
    )

    devai_available = _has_devai_sdk()
    run_real_devmate = os.environ.get("DEVMATE_RUN_REAL") == "1"
    run_real_claude = os.environ.get("CLAUDE_RUN_REAL") == "1"

    for item in items:
        nodeid = item.nodeid
        in_real_devmate_file = any(m in nodeid for m in _REAL_DEVMATE_FILE_MARKERS)
        is_sdk_real_file = "test_devmate_sdk_inferencer_real.py" in nodeid

        # Skip devai-dependent SDK tests when the package isn't importable.
        if (item.name == "test_cross_session_sdk" or is_sdk_real_file) and not devai_available:
            item.add_marker(skip_devai)
            continue

        # Skip real-devmate tests unless opted in (whole-file or named).
        if (in_real_devmate_file or item.name in _REAL_DEVMATE_TESTS) and not run_real_devmate:
            item.add_marker(skip_real_devmate)

        # Skip real-claude tests unless opted in.
        if item.name in _REAL_CLAUDE_TESTS and not run_real_claude:
            item.add_marker(skip_real_claude)

    # Wrap any remaining async test functions so they execute via asyncio.run().
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            original = item.obj

            @functools.wraps(original)
            def make_sync(fn=original):
                sig = inspect.signature(fn)

                def wrapper(**kwargs):
                    return asyncio.run(fn(**kwargs))

                wrapper.__signature__ = sig
                return wrapper

            item.obj = make_sync()
