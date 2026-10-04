"""The native UI mirror (``get_messages``) records the raw text of every
vendor turn once (plan §6.1 step 5.2)."""

from __future__ import annotations

from fakes import text
from helpers import make_native
from later.unittest import TestCase


def _contents(native) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in native.get_messages()]


class MirrorTest(TestCase):
    async def test_a_programmatic_host_gets_its_user_turns_mirrored(self) -> None:
        native, _, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            await native.run_agentic_loop("first")
            await native.run_agentic_loop("second")
            self.assertEqual(
                _contents(native),
                [
                    ("user", "first"),
                    ("assistant", "one"),
                    ("user", "second"),
                    ("assistant", "two"),
                ],
            )

    async def test_a_host_that_records_user_input_gets_no_duplicate(self) -> None:
        # OpenStartup appends the user's message to its session first and
        # hands the whole history over with set_messages every fresh turn.
        native, factory, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            history = [{"role": "user", "content": "first"}]
            native.set_messages(history)
            await native.run_agentic_loop("first")
            history += [
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "second"},
            ]
            native.set_messages(history)
            await native.run_agentic_loop("second")
            self.assertEqual(
                _contents(native),
                [
                    ("user", "first"),
                    ("assistant", "one"),
                    ("user", "second"),
                    ("assistant", "two"),
                ],
            )
            self.assertEqual(factory.last.turn_requests[-1].text, "second")

    async def test_the_request_of_an_entered_sop_is_mirrored_after_the_command(
        self,
    ) -> None:
        native, _, _, _ = make_native([[text("On it.")]])
        async with native:
            await native.run_agentic_loop("/sop mini_research quantum sensors")
            roles_and_texts = _contents(native)
            self.assertEqual(
                roles_and_texts[0], ("user", "/sop mini_research quantum sensors")
            )
            self.assertEqual(roles_and_texts[1][0], "assistant")
            self.assertEqual(
                roles_and_texts[2:],
                [("user", "quantum sensors"), ("assistant", "On it.")],
            )

    async def test_a_recap_carries_the_programmatic_hosts_user_turns(self) -> None:
        native, factory, _, _ = make_native([[text("noted")], [text("42")]])
        async with native:
            await native.run_agentic_loop("remember canary-55")
            native.backend.cwd = native.backend.cwd + "_moved"  # session lost (D9)
            await native.run_agentic_loop("what was it?")
            recap = factory.last.l2_seen[0]
            self.assertIn('type="recap"', recap)
            self.assertIn("user: remember canary-55", recap)
            self.assertIn("assistant: noted", recap)
            self.assertNotIn("user: what was it?", recap)
