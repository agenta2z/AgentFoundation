#!/usr/bin/env python3
"""RovoChat Inferencer — Session Isolation & Resume Integration Tests.

End-to-end tests that verify:
1. Sessions are isolated — each conversation has its own context
2. Sessions can be resumed by ID — even from a fresh inferencer instance
3. Cross-session contamination does NOT occur — Session A cannot recall Session B's secret
4. Three concurrent sessions maintain independent state
5. Chitchat interleavings don't break session context

These are integration tests that make real API calls to RovoChat.
They require valid credentials via environment variables.

Prerequisites:
    export JIRA_URL=https://hello.atlassian.net
    export JIRA_EMAIL=you@atlassian.com
    export JIRA_API_TOKEN=your-token

Usage:
    # Run all tests:
    python test_rovochat_session_isolation.py

    # Run with verbose output:
    python test_rovochat_session_isolation.py -v

    # Run a specific test:
    python test_rovochat_session_isolation.py -k test_two_sessions
"""

import asyncio
import os
import sys
import unittest

# Auto-add AgentFoundation/src and RichPythonUtils/src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_test_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
_agent_foundation_root = os.path.normpath(os.path.join(_test_root, ".."))
_src_dir = os.path.join(_agent_foundation_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)


def _has_credentials() -> bool:
    """Check if RovoChat credentials are available."""
    has_basic = bool(
        (os.environ.get("ROVOCHAT_EMAIL") or os.environ.get("JIRA_EMAIL"))
        and (os.environ.get("ROVOCHAT_API_TOKEN") or os.environ.get("JIRA_API_TOKEN"))
    )
    has_uct = bool(os.environ.get("ROVOCHAT_UCT_TOKEN"))
    return has_basic or has_uct


def _create_inferencer(**kwargs):
    """Create a fresh RovoChatInferencer with no session state."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
        RovoChatInferencer,
    )

    return RovoChatInferencer(**kwargs)


async def _query(inferencer, message: str, session_id: str = "") -> str:
    """Send a query and return the response text."""
    if session_id:
        resp = await inferencer.ainfer(
            message, session_id=session_id, return_sdk_response=True
        )
        return resp.content
    else:
        parts = []
        async for chunk in inferencer.ainfer_streaming(message):
            parts.append(chunk)
        return "".join(parts)


@unittest.skipUnless(_has_credentials(), "RovoChat credentials not available")
class TestRovoChatSessionIsolation(unittest.TestCase):
    """Integration tests for session isolation and resume."""

    def test_two_sessions_isolated_secrets(self):
        """Two sessions with different secrets should each recall only their own.

        Steps:
            1. Session A: "secret word is banana"
            2. Session B: "secret word is dragon"
            3. Resume A: ask for secret → must say "banana", must NOT say "dragon"
            4. Resume B: ask for secret → must say "dragon", must NOT say "banana"
        """

        async def _run():
            # --- Session A: set secret ---
            inf_a = _create_inferencer()
            await _query(
                inf_a,
                "I'm telling you a secret word. Remember it. "
                "The secret word is: banana. Confirm you remember it.",
            )
            session_a = inf_a.active_session_id
            self.assertIsNotNone(session_a, "Session A should have an ID")

            # --- Session B: set different secret ---
            inf_b = _create_inferencer()
            await _query(
                inf_b,
                "I'm telling you a secret word. Remember it. "
                "The secret word is: dragon. Confirm you remember it.",
            )
            session_b = inf_b.active_session_id
            self.assertIsNotNone(session_b, "Session B should have an ID")

            # Sessions must be different
            self.assertNotEqual(session_a, session_b, "Sessions should have different IDs")

            # --- Resume A: recall secret ---
            inf_resume_a = _create_inferencer()
            response_a = await _query(
                inf_resume_a,
                "What is the secret word I told you? Reply with just the single word.",
                session_id=session_a,
            )
            self.assertIn("banana", response_a.lower(), f"Session A should recall 'banana', got: {response_a}")
            self.assertNotIn("dragon", response_a.lower(), f"Session A should NOT mention 'dragon', got: {response_a}")

            # --- Resume B: recall secret ---
            inf_resume_b = _create_inferencer()
            response_b = await _query(
                inf_resume_b,
                "What is the secret word I told you? Reply with just the single word.",
                session_id=session_b,
            )
            self.assertIn("dragon", response_b.lower(), f"Session B should recall 'dragon', got: {response_b}")
            self.assertNotIn("banana", response_b.lower(), f"Session B should NOT mention 'banana', got: {response_b}")

        asyncio.run(_run())

    def test_three_sessions_with_chitchat(self):
        """Three sessions with interleaved chitchat maintain independent context.

        Steps:
            1. Session A: secret is "piano"
            2. Session B: secret is "rocket"
            3. Chitchat in Session A (unrelated question)
            4. Session C: secret is "mango"
            5. Chitchat in Session B (unrelated question)
            6. Resume all three → each recalls only its own secret
        """

        async def _run():
            # --- Create three sessions with secrets ---
            secrets = {"A": "piano", "B": "rocket", "C": "mango"}
            sessions = {}

            for label, secret in secrets.items():
                inf = _create_inferencer()
                await _query(
                    inf,
                    f"Remember this secret word carefully: {secret}. "
                    f"Just confirm you've memorized it.",
                )
                sessions[label] = inf.active_session_id
                self.assertIsNotNone(sessions[label], f"Session {label} should have an ID")

            # All session IDs must be unique
            ids = list(sessions.values())
            self.assertEqual(len(set(ids)), 3, "All three sessions should have unique IDs")

            # --- Chitchat in Session A (shouldn't affect secret) ---
            inf_chat_a = _create_inferencer()
            await _query(
                inf_chat_a,
                "What's the capital of France? Just give a short answer.",
                session_id=sessions["A"],
            )

            # --- Chitchat in Session B (shouldn't affect secret) ---
            inf_chat_b = _create_inferencer()
            await _query(
                inf_chat_b,
                "What is 7 times 8? Just give the number.",
                session_id=sessions["B"],
            )

            # --- Resume all three and verify secrets ---
            for label, secret in secrets.items():
                other_secrets = [s for l, s in secrets.items() if l != label]

                inf_resume = _create_inferencer()
                response = await _query(
                    inf_resume,
                    "What is the secret word I told you at the beginning? "
                    "Reply with just the single word, nothing else.",
                    session_id=sessions[label],
                )

                self.assertIn(
                    secret,
                    response.lower(),
                    f"Session {label} should recall '{secret}', got: {response}",
                )
                for other in other_secrets:
                    self.assertNotIn(
                        other,
                        response.lower(),
                        f"Session {label} should NOT mention '{other}', got: {response}",
                    )

        asyncio.run(_run())

    def test_session_resume_after_multiple_turns(self):
        """A session with multiple turns can still be resumed and recall context.

        Steps:
            1. Tell secret "sunflower"
            2. Ask an unrelated question
            3. Ask another unrelated question
            4. Create a new inferencer, resume, ask for the secret
        """

        async def _run():
            inf = _create_inferencer()

            # Turn 1: set secret
            await _query(
                inf,
                "Remember this secret word: sunflower. Confirm you remember it.",
            )
            session_id = inf.active_session_id

            # Turn 2: chitchat
            await _query(inf, "What programming language is Python named after?")

            # Turn 3: more chitchat
            await _query(inf, "How many continents are there?")

            # Turn 4: resume from fresh inferencer
            inf_fresh = _create_inferencer()
            response = await _query(
                inf_fresh,
                "What was the secret word I told you? Just say the word.",
                session_id=session_id,
            )

            self.assertIn(
                "sunflower",
                response.lower(),
                f"Should recall 'sunflower' after chitchat, got: {response}",
            )

        asyncio.run(_run())

    def test_new_session_does_not_recall_old_secret(self):
        """A brand new session should NOT recall secrets from other sessions.

        Steps:
            1. Session A: secret is "telescope"
            2. Create a completely new session (no session_id)
            3. Ask the new session for "the secret word" → should NOT know it
        """

        async def _run():
            # Set secret in Session A
            inf_a = _create_inferencer()
            await _query(
                inf_a,
                "The secret word is: telescope. Remember it.",
            )

            # Brand new session — should have no context
            inf_new = _create_inferencer()
            response = await _query(
                inf_new,
                "What is the secret word? If you don't know, just say 'I don't know'.",
            )

            self.assertNotIn(
                "telescope",
                response.lower(),
                f"New session should NOT know 'telescope', got: {response}",
            )

        asyncio.run(_run())


if __name__ == "__main__":
    print()
    print("🔬 RovoChat Session Isolation & Resume Tests")
    print("=" * 60)

    if not _has_credentials():
        print()
        print("⚠️  No credentials found. Set environment variables:")
        print("   export JIRA_URL=https://hello.atlassian.net")
        print("   export JIRA_EMAIL=you@atlassian.com")
        print("   export JIRA_API_TOKEN=your-token")
        print()
        sys.exit(1)

    print()
    unittest.main(verbosity=2)
