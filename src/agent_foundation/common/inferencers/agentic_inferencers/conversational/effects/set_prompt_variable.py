# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SetPromptVariable effect — typed-Protocol replacement for variable_manager duck call.

Forwards to `inferencer.prompt_renderer.set_variable(name, value)` (the typed
Protocol method introduced in Diff 1a). The default
`TemplateManagerPromptRenderer` has no `set_variable`, and the value never
reaches `prior_context` (the render feed), so widget handlers publish their
answers with `PublishSessionVariablesEffect` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    effect_target,
    EffectTarget,
)


@dataclass
class SetPromptVariable:
    name: str
    value: str

    async def apply(self, inferencer: EffectTarget) -> None:
        renderer = effect_target(inferencer).prompt_renderer
        if renderer is None:
            return
        renderer.set_variable(self.name, self.value)
