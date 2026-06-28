"""Shared fixtures + async-test wiring for metamate tests.

Mirrors ``../devmate/conftest.py``:

- Provides a ``query`` fixture for the real integration scripts.
- Auto-wraps ``async def test_*`` functions so they run via ``asyncio.run()``
  without needing ``pytest-asyncio``.
- Skips the live-service integration tests by default. EVERY existing
  ``test_metamate_*`` file in this directory drives MetaMate end-to-end —
  either via ``buck run :query_metamate`` (CLI) or the
  ``msl.metamate`` / ``metamate_standalone`` GraphQL client (SDK) — against
  the live MetaMate service. None of that is available outside a Buck build
  with service connectivity, so they are skipped unless ``METAMATE_RUN_REAL=1``.
  Offline unit tests (``test_metamate_offline.py``) always run.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os

import pytest

DEFAULT_QUERY = "What is 2 + 2? Reply with just the number."


@pytest.fixture
def query() -> str:
    """Default query for metamate real integration tests."""
    return DEFAULT_QUERY


# Files whose every test calls the live MetaMate service (CLI buck target or
# SDK GraphQL client). Skipped unless ``METAMATE_RUN_REAL=1``.
_REAL_METAMATE_FILE_MARKERS: tuple[str, ...] = (
    "test_metamate_cli_inferencer_real.py",
    "test_metamate_cli_deep_research.py",
    "test_metamate_sdk_inferencer_real.py",
    "test_metamate_standalone_inferencer.py",
    "test_deep_research_from_file.py",
)


def pytest_collection_modifyitems(config, items):
    """Skip live integration tests by default + wrap async tests."""
    skip_real = pytest.mark.skip(
        reason="real MetaMate integration test (needs buck/msl + live service); "
        "set METAMATE_RUN_REAL=1 to opt in"
    )
    run_real = os.environ.get("METAMATE_RUN_REAL") == "1"

    for item in items:
        if not run_real and any(m in item.nodeid for m in _REAL_METAMATE_FILE_MARKERS):
            item.add_marker(skip_real)

    # Wrap any async test functions so they execute via asyncio.run().
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "obj", None)):
            original = item.obj

            @functools.wraps(original)
            def make_sync(fn=original):
                sig = inspect.signature(fn)

                def wrapper(**kwargs):
                    return asyncio.run(fn(**kwargs))

                wrapper.__signature__ = sig
                return wrapper

            item.obj = make_sync()
