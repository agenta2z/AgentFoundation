"""Tests for @command decorator, CommandRegistry, and CI command integration."""

import asyncio
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    command,
    CommandMeta,
    CommandRegistry,
    UnknownCommand,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrs


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


class _OrderBase:
    @command("alpha")
    async def _cmd_alpha(self) -> str:
        return "base alpha"

    @command("beta", aliases=("b",))
    async def _cmd_beta(self) -> str:
        return "base beta"

    @command("gamma")
    async def _cmd_gamma(self) -> str:
        return "base gamma"


class _OrderDerived(_OrderBase):
    @command("delta")
    async def _cmd_delta(self) -> str:
        return "derived delta"

    @command("beta", description="derived", aliases=("b",))
    async def _cmd_beta(self) -> str:
        return "derived beta"

    @command("gamma", description="shadow")
    async def _cmd_gamma_shadow(self) -> str:
        return "derived gamma"


def _listed(registry: CommandRegistry) -> list[str]:
    return [
        line.split("`")[1].lstrip("/")
        for line in registry.render_for_prompt().splitlines()
    ]


class CommandOrderTest(unittest.TestCase):
    def test_classic_prompt_lists_commands_in_declaration_order(self) -> None:
        self.assertEqual(
            _listed(_make_ci()._commands),
            [
                "help",
                "status",
                "clear",
                "sop <args>",
                "pause_sop",
                "exit_sop",
                "resume_sop <args>",
                "model <model_name>",
                "root <path>",
                "target <path>",
            ],
        )

    def test_override_and_shadow_keep_the_base_position(self) -> None:
        registry = CommandRegistry(_OrderDerived())
        self.assertEqual(_listed(registry), ["alpha", "beta", "gamma", "delta"])
        descriptions = {m.name: m.description for m in registry.list_commands()}
        self.assertEqual(descriptions["beta"], "derived")
        self.assertEqual(descriptions["gamma"], "shadow")
        loop = asyncio.new_event_loop()
        try:
            beta = loop.run_until_complete(registry.dispatch("/b"))
            gamma = loop.run_until_complete(registry.dispatch("/gamma"))
        finally:
            loop.close()
        self.assertEqual(beta, "derived beta")
        self.assertEqual(gamma, "derived gamma")


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
