"""Test hygiene for the run_context suite.

Several tests here drive coroutines via ``asyncio.run(...)``, which CLOSES the
global event loop on completion. Other modules in the wider suite still use the
deprecated ``asyncio.get_event_loop()`` and would then raise "There is no current
event loop". To keep the suite order-independent, restore a fresh event loop and
clear any leaked active RunContext after every test in this package.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _restore_event_loop_and_ctx():
    yield
    # Clear any leaked active context (defensive — tests use enter/exit, but a
    # mid-test failure could leave one set).
    try:
        from agent_foundation.common.inferencers.run_context import bridge

        bridge._active_ctx.set(None)
    except Exception:
        pass
    # Restore a usable global event loop for downstream modules.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
