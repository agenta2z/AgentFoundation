"""Public API for agent_foundation.ui.

Lightweight contract types load eagerly. Heavy transports
(TerminalInteractive, QueueInteractive, WebUIInteractive) load lazily
on first attribute access, so consumers that only handle wire-format
messages (WidgetMessage / InputModeConfig round-trips) do not need
transport-specific deps (webaxon, queue_service backends, WebSocket
machinery) installed at all. NOTE: this does not avoid rich_python_utils
or attr — both are required by the contract layer itself (interactive_base
imports rich_python_utils.common_objects.debuggable directly).
"""
from __future__ import annotations
from importlib import import_module
from typing import TYPE_CHECKING

# --- Eager: lightweight contract types ---
from agent_foundation.ui.input_modes import (
    InputMode,
    InputModeConfig,
    ChoiceOption,
    press_to_continue,
    exact_string,
    single_choice,
    multiple_choices,
)
from agent_foundation.ui.widget_protocol import (
    WidgetMessage,
    WidgetResponse,
    WidgetField,
    WIDGET_TEXT_INPUT,
    WIDGET_SINGLE_CHOICE,
    WIDGET_MULTIPLE_CHOICE,
    WIDGET_DROPDOWN,
    WIDGET_TOGGLE,
    WIDGET_TOOL_ARGUMENT_FORM,
    WIDGET_CONFIRMATION,
    WIDGET_MULTI_INPUT,
    WIDGET_GROUPED,
    WIDGET_DEFAULT,
    WIDGET_TYPES,
)
from agent_foundation.ui.interactive_base import (
    InteractiveBase,
    InteractionFlags,
)
from agent_foundation.ui.rich_interactive_base import RichInteractiveBase
from agent_foundation.ui.interactive_checkpoint import (
    CheckpointResult,
    run_checkpoint,
    checkpoint_plan_review,
    checkpoint_breakdown_review,
    checkpoint_results_review,
)

# --- Lazy: heavy transports (PEP 562) ---
_LAZY = {
    'TerminalInteractive': 'agent_foundation.ui.terminal_interactive',
    'QueueInteractive': 'agent_foundation.ui.queue_interactive',
    'WebUIInteractive': 'agent_foundation.ui.web_interactive',
}


def __getattr__(name: str):
    if name in _LAZY:
        mod = import_module(_LAZY[name])
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'agent_foundation.ui' has no attribute {name!r}")


if TYPE_CHECKING:
    from agent_foundation.ui.terminal_interactive import TerminalInteractive  # noqa: F401
    from agent_foundation.ui.queue_interactive import QueueInteractive  # noqa: F401
    from agent_foundation.ui.web_interactive import WebUIInteractive  # noqa: F401

__all__ = [
    # contract
    'InputMode', 'InputModeConfig', 'ChoiceOption',
    'press_to_continue', 'exact_string', 'single_choice', 'multiple_choices',
    'WidgetMessage', 'WidgetResponse', 'WidgetField',
    'WIDGET_TEXT_INPUT', 'WIDGET_SINGLE_CHOICE', 'WIDGET_MULTIPLE_CHOICE',
    'WIDGET_DROPDOWN', 'WIDGET_TOGGLE', 'WIDGET_TOOL_ARGUMENT_FORM',
    'WIDGET_CONFIRMATION', 'WIDGET_MULTI_INPUT', 'WIDGET_GROUPED',
    'WIDGET_DEFAULT', 'WIDGET_TYPES',
    'InteractiveBase', 'InteractionFlags', 'RichInteractiveBase',
    'CheckpointResult', 'run_checkpoint', 'checkpoint_plan_review',
    'checkpoint_breakdown_review', 'checkpoint_results_review',
    # transports (lazy)
    'TerminalInteractive', 'QueueInteractive', 'WebUIInteractive',
]
