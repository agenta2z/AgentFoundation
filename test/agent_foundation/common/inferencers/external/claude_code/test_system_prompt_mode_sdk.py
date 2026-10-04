"""Unit tests: ClaudeCodeSdkInferencer ``system_prompt_mode``.

``"legacy_empty"`` (default) keeps the historical behavior — ``system_prompt``
is the whole system prompt, so an empty one leaves Claude Code with none.
``"replace"`` replaces Claude Code's system prompt only with non-empty text.
``"preset_append"`` keeps Claude Code's system prompt and appends
``system_prompt``. Verified on the options the inferencer builds and on the
CLI flags the SDK transport emits for them, without spawning the SDK
subprocess.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)

_PRESET = {"type": "preset", "preset": "claude_code"}
_EMPTY_PROMPTS = ("", None, "  \n")


def _options_kwargs(inferencer: ClaudeCodeSdkInferencer) -> dict:
    """What ``aconnect()`` passes to ``ClaudeAgentOptions``."""
    import claude_agent_sdk

    captured: dict = {}

    def fake_options(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ClaudeAgentOptions")

    client = MagicMock(name="ClaudeSDKClient")
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    async def run() -> None:
        with (
            patch.object(claude_agent_sdk, "ClaudeAgentOptions", fake_options),
            patch.object(claude_agent_sdk, "ClaudeSDKClient", lambda options: client),
        ):
            await inferencer.aconnect()
        await inferencer.adisconnect()

    asyncio.run(run())
    return captured


def _system_prompt_args(system_prompt) -> list:
    """The system-prompt flags, each with its value, that the SDK transport
    emits for ``system_prompt``."""
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport(
        prompt="hi", options=ClaudeAgentOptions(system_prompt=system_prompt)
    )
    transport._cli_path = "claude"
    cmd = transport._build_command()
    return [
        (flag, value)
        for flag, value in zip(cmd, cmd[1:])
        if flag.startswith("--") and "system-prompt" in flag
    ]


def _sdk_system_prompt(system_prompt, **kwargs):
    inferencer = ClaudeCodeSdkInferencer(system_prompt=system_prompt, **kwargs)
    return _options_kwargs(inferencer)["system_prompt"]


class LegacyEmptyModeTest(unittest.TestCase):
    def test_is_the_default(self) -> None:
        self.assertEqual(ClaudeCodeSdkInferencer().system_prompt_mode, "legacy_empty")

    def test_text_replaces_the_system_prompt(self) -> None:
        option = _sdk_system_prompt("Be terse.")
        self.assertEqual(option, "Be terse.")
        self.assertEqual(
            _system_prompt_args(option), [("--system-prompt", "Be terse.")]
        )

    def test_empty_prompt_blanks_claude_codes_prompt(self) -> None:
        for prompt in ("", None):
            with self.subTest(prompt=prompt):
                option = _sdk_system_prompt(prompt)
                self.assertEqual(option, prompt)
                self.assertEqual(_system_prompt_args(option), [("--system-prompt", "")])


class ReplaceModeTest(unittest.TestCase):
    def test_text_replaces_the_system_prompt(self) -> None:
        option = _sdk_system_prompt("Be terse.", system_prompt_mode="replace")
        self.assertEqual(option, "Be terse.")
        self.assertEqual(
            _system_prompt_args(option), [("--system-prompt", "Be terse.")]
        )

    def test_empty_prompt_keeps_claude_codes_prompt(self) -> None:
        for prompt in _EMPTY_PROMPTS:
            with self.subTest(prompt=prompt):
                option = _sdk_system_prompt(prompt, system_prompt_mode="replace")
                self.assertEqual(option, _PRESET)
                self.assertEqual(_system_prompt_args(option), [])


class PresetAppendModeTest(unittest.TestCase):
    def test_appends_to_claude_codes_prompt(self) -> None:
        option = _sdk_system_prompt("Be terse.", system_prompt_mode="preset_append")
        self.assertEqual(option, {**_PRESET, "append": "Be terse."})
        self.assertEqual(
            _system_prompt_args(option), [("--append-system-prompt", "Be terse.")]
        )

    def test_empty_prompt_sends_no_system_prompt_flag(self) -> None:
        for prompt in _EMPTY_PROMPTS:
            with self.subTest(prompt=prompt):
                option = _sdk_system_prompt(prompt, system_prompt_mode="preset_append")
                self.assertEqual(option, _PRESET)
                self.assertEqual(_system_prompt_args(option), [])


class SystemPromptModeValidationTest(unittest.TestCase):
    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClaudeCodeSdkInferencer(system_prompt_mode="append")
