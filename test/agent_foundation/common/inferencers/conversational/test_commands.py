"""Tests for @command decorator, CommandRegistry, and CI command integration."""

import asyncio
import unittest

from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    CommandMeta,
    CommandRegistry,
    UnknownCommand,
    command,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs(slots=False)
class MockBase(InferencerBase):
    async def _ainfer(self, inp, cfg=None, **kw):
        return "mock"

    def _infer(self, inp, cfg=None, **kw):
        return "mock"


def _make_ci(**kwargs):
    return ConversationalInferencer(base_inferencer=MockBase(), **kwargs)


class TestCommandDecorator(unittest.TestCase):
    def test_decorator_sets_metadata(self):
        @command("test", description="A test", aliases=("t",))
        async def _handler(self):
            return "ok"

        meta = _handler.__command__
        assert isinstance(meta, CommandMeta)
        assert meta.name == "test"
        assert meta.description == "A test"
        assert meta.aliases == ("t",)

    def test_decorator_defaults(self):
        @command("bare")
        async def _handler(self):
            return "ok"

        meta = _handler.__command__
        assert meta.description == ""
        assert meta.aliases == ()
        assert meta.requires_active_sop is False


class TestCommandRegistry(unittest.TestCase):
    def test_builtin_commands_discovered(self):
        ci = _make_ci()
        cmds = ci._commands.list_commands()
        names = [m.name for m in cmds]
        assert "help" in names
        assert "status" in names
        assert "clear" in names
        assert "sop" in names
        assert "pause_sop" in names
        assert "resume_sop" in names
        assert "exit_sop" in names

    def test_command_count(self):
        ci = _make_ci()
        # help, status, clear, sop, pause_sop, exit_sop, resume_sop, model, root, target
        assert len(ci._commands.list_commands()) == 10

    def test_is_command_slash(self):
        ci = _make_ci()
        assert ci._commands.is_command("/help")
        assert ci._commands.is_command("/status")
        assert ci._commands.is_command("/clear")
        assert ci._commands.is_command("/sop")
        assert ci._commands.is_command("/pause_sop")
        assert ci._commands.is_command("/resume_sop")

    def test_is_command_alias(self):
        ci = _make_ci()
        assert ci._commands.is_command("/?")
        assert ci._commands.is_command("/s")

    def test_not_command_without_slash(self):
        ci = _make_ci()
        assert not ci._commands.is_command("help")
        assert not ci._commands.is_command("status")

    def test_unknown_not_command(self):
        ci = _make_ci()
        assert not ci._commands.is_command("/unknown_xyz")

    def test_dispatch_help(self):
        ci = _make_ci()
        result = asyncio.get_event_loop().run_until_complete(
            ci._commands.dispatch("/help")
        )
        assert "Available commands:" in result
        assert "/help" in result

    def test_dispatch_status_no_sop(self):
        ci = _make_ci()
        result = asyncio.get_event_loop().run_until_complete(
            ci._commands.dispatch("/status")
        )
        assert "No active SOP" in result

    def test_dispatch_clear(self):
        ci = _make_ci()
        ci.add_message("user", "hello")
        assert len(ci._messages) == 1
        asyncio.get_event_loop().run_until_complete(ci._commands.dispatch("/clear"))
        assert len(ci._messages) == 0

    def test_requires_active_sop_guard(self):
        ci = _make_ci()
        result = asyncio.get_event_loop().run_until_complete(
            ci._commands.dispatch("/pause_sop")
        )
        assert "requires an active SOP" in result

    def test_unknown_command_raises(self):
        ci = _make_ci()
        with self.assertRaises(UnknownCommand):
            asyncio.get_event_loop().run_until_complete(
                ci._commands.dispatch("/nonexistent")
            )


class TestCommandDispatchInLoop(unittest.TestCase):
    def test_command_bypasses_llm(self):
        ci = _make_ci()
        result = asyncio.get_event_loop().run_until_complete(
            ci.run_agentic_loop("/help")
        )
        assert "Available commands:" in result.text
        assert result.iterations_used == 0

    def test_command_recorded_in_messages(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(ci.run_agentic_loop("/status"))
        assert len(ci._messages) == 2
        assert ci._messages[0]["role"] == "user"
        assert ci._messages[0]["content"] == "/status"
        assert ci._messages[1]["role"] == "assistant"

    def test_unknown_slash_falls_through(self):
        ci = _make_ci()
        assert not ci._commands.is_command("/unknown_thing")


if __name__ == "__main__":
    unittest.main()
