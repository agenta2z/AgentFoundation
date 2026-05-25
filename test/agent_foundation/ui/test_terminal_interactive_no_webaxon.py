"""Test that TerminalInteractive can be imported without webaxon installed.

Phase 0 made webaxon a lazy import inside _send_response(). This test
verifies the import succeeds even when webaxon is not available.
"""
import sys
import pytest


def test_terminal_interactive_imports_without_webaxon(monkeypatch):
    """Importing TerminalInteractive succeeds when webaxon is not installed."""
    # Remove webaxon from sys.modules if present
    webaxon_modules = [k for k in sys.modules if k.startswith('webaxon')]
    saved = {}
    for k in webaxon_modules:
        saved[k] = sys.modules.pop(k)

    # Block webaxon from being imported
    monkeypatch.setitem(sys.modules, 'webaxon', None)
    monkeypatch.setitem(sys.modules, 'webaxon.html_utils', None)
    monkeypatch.setitem(sys.modules, 'webaxon.html_utils.common', None)

    # Force re-import of terminal_interactive
    if 'agent_foundation.ui.terminal_interactive' in sys.modules:
        del sys.modules['agent_foundation.ui.terminal_interactive']

    # This should NOT raise ImportError
    from agent_foundation.ui.terminal_interactive import TerminalInteractive
    assert TerminalInteractive is not None
