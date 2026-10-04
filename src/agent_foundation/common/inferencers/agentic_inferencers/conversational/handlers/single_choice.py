# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SingleChoiceHandler — single-choice widget with AF composite-input support.

Extended behaviors beyond the RE port:

- ``variable_override``: if the user edited the choice content in-widget, the
  response carries ``variable_override: {name: value}``; publish those instead
  of the default output-var binding.
- Composite input: a ``ChoiceItem`` may declare a nested ``InputFieldSpec``
  (path/typed input embedded under a choice). When the selected choice has one
  and the response carries ``inputs: {spec.name: raw}``, decode via
  ``finalize_input_value`` and return in ``HandlerResult.bindings`` (decode
  RESULT, per §A1.a — not a side-effect). The dispatch loop merges these
  bindings into the aggregate ``collected`` dict.

Also emits ``PublishSessionVariablesEffect`` (with tool_type namespacing) for
each of ``tool.output_vars`` — matches AF's inline ``set_session_variables``
semantics and the ``<tool_type>__<var>`` alias convention (A1.b).
"""

from __future__ import annotations

from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    finalize_input_value,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    PublishSessionVariablesEffect,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
    HandlerContext,
    HandlerResult,
    InferencerEffect,
)
from agent_foundation.common.ui.input_modes import (
    ChoiceOption,
    InputModeConfig,
    single_choice,
)


class SingleChoiceHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.SINGLE_CHOICE

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        # Mirrors inline `_choice_option_from` — preserves description AND
        # serialized composite `input` spec so ChoiceItem.input survives into
        # InputModeConfig.to_dict().
        options = [
            ChoiceOption(
                label=c.label,
                value=c.value,
                description=getattr(c, "description", "") or "",
                input=c.input.to_dict()
                if getattr(c, "has_input", False) and c.input is not None
                else None,
            )
            for c in tool.choices
        ]
        return single_choice(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
        )

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        if not isinstance(response, dict):
            text = str(response)
            effects: list[InferencerEffect] = []
            if tool.output_vars:
                effects.append(
                    PublishSessionVariablesEffect(
                        {v: text for v in tool.output_vars},
                        tool_type=tool.tool_type,
                    )
                )
            return HandlerResult(text=text, effects=effects)

        choice_idx = response.get("choice_index")
        if (
            choice_idx is not None
            and tool.choices
            and 0 <= choice_idx < len(tool.choices)
        ):
            choice_value = tool.choices[choice_idx].value
        else:
            choice_value = response.get("content") or response.get("custom_text") or ""

        effects = []

        # variable_override wins over the default output-var binding.
        variable_override = response.get("variable_override")
        if variable_override and isinstance(variable_override, dict):
            effects.append(
                PublishSessionVariablesEffect(
                    variable_override,
                    tool_type=tool.tool_type,
                )
            )
        elif tool.output_vars:
            effects.append(
                PublishSessionVariablesEffect(
                    {v: str(choice_value) for v in tool.output_vars},
                    tool_type=tool.tool_type,
                )
            )

        # Composite input: nested InputFieldSpec on the selected choice.
        bindings: dict[str, Any] = {}
        if (
            choice_idx is not None
            and tool.choices
            and 0 <= choice_idx < len(tool.choices)
            and tool.choices[choice_idx].has_input
        ):
            spec = tool.choices[choice_idx].input
            inputs = response.get("inputs")
            raw = None
            if isinstance(inputs, dict):
                raw = inputs.get(spec.name, next(iter(inputs.values()), None))
            elif "content" in response:
                raw = response.get("content")
            nested_val = finalize_input_value(
                raw,
                expected_input_type=spec.expected_input_type,
                prefix=spec.prefix,
                allow_multiple_input=spec.allow_multiple_input,
                serialization=spec.serialization,
                session_root=ctx.session_root or "",
            )
            if spec.name:
                effects.append(
                    PublishSessionVariablesEffect(
                        {spec.name: nested_val},
                        tool_type=tool.tool_type,
                    )
                )
                bindings[spec.name] = nested_val

        return HandlerResult(text=str(choice_value), effects=effects, bindings=bindings)
