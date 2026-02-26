"""Intermediate base class adding structured input mode support.

Sits between InteractiveBase and transport implementations (e.g. QueueInteractive).
Handles:
  - send_response side: input_mode threading via instance state
  - get_input side: postprocessing structured UI responses -> semantic values

Transport subclasses read self._current_input_mode in their _send_response().
UI clients send structured dicts; this class maps them to agent-ready values.
"""
from typing import Any, Dict, List, Tuple, Union

from attr import attrs, attrib

from science_modeling_tools.ui.input_modes import InputMode, InputModeConfig
from science_modeling_tools.ui.interactive_base import (
    InteractiveBase, InteractionFlags,
)


@attrs
class RichInteractiveBase(InteractiveBase):
    """Intermediate base class that adds structured input mode support.

    Two responsibilities:
    1. send_response(input_mode=) -> threads input_mode to transport via instance state
    2. get_input() -> postprocesses structured UI responses into semantic values

    UI clients send structured dicts for choice modes:
        SINGLE_CHOICE:  {"choice_index": int}
                        {"choice_index": int, "follow_up_value": str}
                        {"custom_text": str}
        MULTIPLE_CHOICES: {"selections": [{"choice_index": int, "follow_up_value"?: str} | {"custom_text": str}, ...]}
        FREE_TEXT / PRESS_TO_CONTINUE / EXACT_STRING: plain string (passthrough)

    This class maps structured input -> semantic values so the agent gets clean data
    regardless of which UI implementation collected the input.
    """

    # Internal state -- not constructor params
    _current_input_mode: Any = attrib(default=None, init=False)
    _pending_input_mode: Any = attrib(default=None, init=False)

    # -- send_response side ---------------------------------------------------

    def send_response(
        self,
        response: Union[Any, List, Tuple],
        flag: InteractionFlags = InteractionFlags.TurnCompleted,
        input_mode=None,
    ) -> None:
        """Override to support input_mode parameter.

        Stores input_mode as instance state so transport subclasses can access
        self._current_input_mode in their _send_response() implementation.
        """
        self._current_input_mode = input_mode
        super().send_response(response, flag=flag)

        # Save for get_input() postprocessing if waiting for user input
        if flag == InteractionFlags.PendingInput and input_mode is not None:
            self._pending_input_mode = input_mode

        self._current_input_mode = None

    # -- get_input side -------------------------------------------------------

    def get_input(self):
        """Override to apply postprocessing based on pending input mode."""
        raw = super().get_input()

        if self._pending_input_mode is not None and raw is not None:
            processed = self._postprocess_input(raw, self._pending_input_mode)
            self._pending_input_mode = None
            return processed

        self._pending_input_mode = None
        return raw

    def _postprocess_input(self, raw_input: Any, input_mode: InputModeConfig) -> Any:
        """Map structured UI response -> semantic value based on input_mode.

        For queue-based transports, raw_input is typically a dict like:
            {"user_input": <structured_data>, "session_id": ..., "timestamp": ...}
        This method extracts user_input, processes it, and returns the modified dict.

        For direct transports, raw_input may be the structured data itself.
        """
        # Handle queue wrapper dict: extract user_input, process, put back
        if isinstance(raw_input, dict) and 'user_input' in raw_input:
            user_input = raw_input['user_input']
            processed = self._resolve_structured_input(user_input, input_mode)
            return {**raw_input, 'user_input': processed}

        # Direct transport: process the value itself
        return self._resolve_structured_input(raw_input, input_mode)

    def _resolve_structured_input(self, user_input: Any, input_mode: InputModeConfig) -> Any:
        """Core mapping logic: structured UI data -> semantic value.

        Returns a string that the agent can directly use.
        """
        mode = input_mode.mode

        if mode == InputMode.SINGLE_CHOICE and isinstance(user_input, dict):
            return self._resolve_single_choice(user_input, input_mode)

        elif mode == InputMode.MULTIPLE_CHOICES and isinstance(user_input, dict):
            return self._resolve_multiple_choices(user_input, input_mode)

        elif mode == InputMode.EXACT_STRING:
            # Validate server-side (UI may also validate for UX)
            expected = input_mode.expected_string
            if input_mode.case_sensitive:
                return user_input if user_input == expected else expected
            return user_input if str(user_input).lower() == expected.lower() else expected

        # FREE_TEXT, PRESS_TO_CONTINUE: passthrough
        return user_input

    def _resolve_single_choice(self, data: Dict, input_mode: InputModeConfig) -> str:
        """Map SINGLE_CHOICE structured input -> value string.

        Structured input formats:
            {"choice_index": 1}                              -> options[1].value
            {"choice_index": 1, "follow_up_value": "pass"}  -> "pass"
            {"custom_text": "something"}                     -> "something"
        """
        if 'follow_up_value' in data:
            return data['follow_up_value']

        if 'choice_index' in data:
            idx = data['choice_index']
            if 0 <= idx < len(input_mode.options):
                return input_mode.options[idx].value
            return str(idx)  # out of range fallback

        if 'custom_text' in data:
            return data['custom_text']

        return str(data)  # fallback

    def _resolve_multiple_choices(self, data: Dict, input_mode: InputModeConfig) -> str:
        """Map MULTIPLE_CHOICES structured input -> pipe-delimited value string.

        Structured input format:
            {"selections": [
                {"choice_index": 0},
                {"choice_index": 1, "follow_up_value": "val"},
                {"custom_text": "extra"}
            ]}
        Returns: "option0_value|val|extra"
        """
        selections = data.get('selections', [])
        values = []

        for sel in selections:
            if 'follow_up_value' in sel:
                values.append(sel['follow_up_value'])
            elif 'choice_index' in sel:
                idx = sel['choice_index']
                if 0 <= idx < len(input_mode.options):
                    values.append(input_mode.options[idx].value)
            elif 'custom_text' in sel:
                values.append(sel['custom_text'])

        return '|'.join(values) if values else str(data)  # fallback
