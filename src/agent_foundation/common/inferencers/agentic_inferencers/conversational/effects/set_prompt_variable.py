# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SetPromptVariable effect — typed-Protocol replacement for variable_manager duck call.

Forwards to `inferencer.prompt_renderer.set_variable(name, value)` (the typed
Protocol method introduced in Diff 1a). Used by SINGLE_CHOICE / MULTIPLE_CHOICE
to bind `output_vars[0]` to the user's choice value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )


@dataclass
class SetPromptVariable:
    name: str
    value: str

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        if inferencer.prompt_renderer is None:
            return
        inferencer.prompt_renderer.set_variable(self.name, self.value)
