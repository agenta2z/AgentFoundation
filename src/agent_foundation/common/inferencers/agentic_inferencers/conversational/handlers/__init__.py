# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Built-in conversation tool handlers + default registry factory.

Registers the 5 generic handlers that ship with the framework.
Domain-specific handlers should be registered by downstream consumers.
"""

from __future__ import annotations

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_registry import (
    ConversationToolHandlerRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.clarification import (
    ClarificationHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.confirmation import (
    ConfirmationHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.multiple_choice import (
    MultipleChoiceHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.single_choice import (
    SingleChoiceHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.tool_argument_form import (
    ToolArgumentFormHandler,
)


def default_registry() -> ConversationToolHandlerRegistry:
    """Build the framework-default registry of the 5 generic handlers.

    Domain-specific handlers should be registered by downstream consumers
    via `reg.register(...)` after calling this factory.
    """
    reg = ConversationToolHandlerRegistry()
    reg.register(ClarificationHandler())
    reg.register(SingleChoiceHandler())
    reg.register(MultipleChoiceHandler())
    reg.register(ConfirmationHandler())
    reg.register(ToolArgumentFormHandler())
    return reg
