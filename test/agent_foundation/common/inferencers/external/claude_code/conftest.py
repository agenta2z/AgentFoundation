"""Shared fixtures for Claude Code integration tests.

Provides:
- ``query`` fixture with a default test query
- Auto-wrapping of ``async def test_*`` functions so they run via
  ``asyncio.run()`` without needing ``pytest-asyncio`` (which is at
  version 1.3.0 and doesn't handle modern async test patterns).
"""

import asyncio
import functools
import inspect

import pytest

DEFAULT_QUERY = "What is Python? Answer in one sentence."


@pytest.fixture
def query() -> str:
    """Default query for Claude Code real integration tests."""
    return DEFAULT_QUERY


def pytest_collection_modifyitems(items):
    """Wrap async test functions with asyncio.run() so they work as sync tests.

    This avoids dependency on a modern pytest-asyncio version. Each async
    test function is replaced with a sync wrapper that calls asyncio.run().
    """
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            original = item.obj

            @functools.wraps(original)
            def make_sync(fn=original):
                # Get the function's parameter names to handle fixtures
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())

                def wrapper(**kwargs):
                    return asyncio.run(fn(**kwargs))

                # Preserve signature for pytest fixture injection
                wrapper.__signature__ = sig
                return wrapper

            item.obj = make_sync()
