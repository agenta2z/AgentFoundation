"""Structured input mode definitions for the interactive protocol.

When an agent sends a PendingInput response, it can include an ``input_mode``
field telling the client *how* to collect the next user input.
"""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional


class InputMode(StrEnum):
    FREE_TEXT = 'free_text'
    PRESS_TO_CONTINUE = 'press_to_continue'
    EXACT_STRING = 'exact_string'
    SINGLE_CHOICE = 'single_choice'
    MULTIPLE_CHOICES = 'multiple_choices'


@dataclass
class ChoiceOption:
    label: str
    value: str
    follow_up_prompt: str = ''  # If non-empty, prompt for additional input after selection
    needs_user_copilot: bool = False  # If True, user interacted with browser; discard stale action_results


@dataclass
class InputModeConfig:
    mode: InputMode = InputMode.FREE_TEXT
    prompt: str = ''
    expected_string: str = ''
    case_sensitive: bool = False
    options: List[ChoiceOption] = field(default_factory=list)
    allow_custom: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {'mode': self.mode.value}
        if self.prompt:
            d['prompt'] = self.prompt
        if self.mode == InputMode.EXACT_STRING:
            d['expected_string'] = self.expected_string
            d['case_sensitive'] = self.case_sensitive
        elif self.mode in (InputMode.SINGLE_CHOICE, InputMode.MULTIPLE_CHOICES):
            d['options'] = [
                {'label': o.label, 'value': o.value,
                 **(({'follow_up_prompt': o.follow_up_prompt} if o.follow_up_prompt else {})),
                 **(({'needs_user_copilot': True} if o.needs_user_copilot else {}))}
                for o in self.options
            ]
            d['allow_custom'] = self.allow_custom
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'InputModeConfig':
        if not d:
            return cls()
        mode = InputMode(d.get('mode', 'free_text'))
        config = cls(mode=mode, prompt=d.get('prompt', ''))
        if mode == InputMode.EXACT_STRING:
            config.expected_string = d.get('expected_string', '')
            config.case_sensitive = d.get('case_sensitive', False)
        elif mode in (InputMode.SINGLE_CHOICE, InputMode.MULTIPLE_CHOICES):
            config.options = [
                ChoiceOption(label=o['label'], value=o['value'],
                             follow_up_prompt=o.get('follow_up_prompt', ''),
                             needs_user_copilot=o.get('needs_user_copilot', False))
                for o in d.get('options', [])
            ]
            config.allow_custom = d.get('allow_custom', True)
        return config


# Convenience constructors
def press_to_continue(prompt: str = '') -> InputModeConfig:
    return InputModeConfig(mode=InputMode.PRESS_TO_CONTINUE, prompt=prompt)


def exact_string(expected: str, prompt: str = '', case_sensitive: bool = False) -> InputModeConfig:
    return InputModeConfig(mode=InputMode.EXACT_STRING, expected_string=expected,
                           prompt=prompt, case_sensitive=case_sensitive)


def single_choice(options: List[ChoiceOption], allow_custom: bool = True, prompt: str = '') -> InputModeConfig:
    return InputModeConfig(mode=InputMode.SINGLE_CHOICE, options=options,
                           allow_custom=allow_custom, prompt=prompt)


def multiple_choices(options: List[ChoiceOption], allow_custom: bool = True, prompt: str = '') -> InputModeConfig:
    return InputModeConfig(mode=InputMode.MULTIPLE_CHOICES, options=options,
                           allow_custom=allow_custom, prompt=prompt)
