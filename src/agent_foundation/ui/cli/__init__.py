"""CLI UI framework for agent_foundation.

This package provides Rich-powered terminal components for SOP-driven
agent interactions.  **All heavy imports (Rich, prompt_toolkit) are
deferred** so that ``import agent_foundation`` never triggers Rich
loading.  Concrete classes are resolved lazily via ``__getattr__``.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# Mapping of public names -> (module_path, attribute_name)
_LAZY_IMPORTS = {
    "RichTerminalInteractive": ("agent_foundation.ui.cli.rich_terminal", "RichTerminalInteractive"),
    "StreamingPanel": ("agent_foundation.ui.cli.streaming", "StreamingPanel"),
    "ThemeManager": ("agent_foundation.ui.cli.theme", "ThemeManager"),
    "COLORS": ("agent_foundation.ui.cli.theme", "COLORS"),
    "ask_confirm": ("agent_foundation.ui.cli.prompts", "ask_confirm"),
    "ask_single_choice": ("agent_foundation.ui.cli.prompts", "ask_single_choice"),
    "ask_text": ("agent_foundation.ui.cli.prompts", "ask_text"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod_path, attr_name = _LAZY_IMPORTS[name]
        mod = import_module(mod_path)
        attr = getattr(mod, attr_name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module 'agent_foundation.ui.cli' has no attribute {name!r}")


if TYPE_CHECKING:
    from agent_foundation.ui.cli.rich_terminal import RichTerminalInteractive as RichTerminalInteractive  # noqa: F401
    from agent_foundation.ui.cli.streaming import StreamingPanel as StreamingPanel  # noqa: F401
    from agent_foundation.ui.cli.theme import ThemeManager as ThemeManager, COLORS as COLORS  # noqa: F401
    from agent_foundation.ui.cli.prompts import ask_confirm as ask_confirm, ask_single_choice as ask_single_choice, ask_text as ask_text  # noqa: F401

__all__ = list(_LAZY_IMPORTS.keys())
