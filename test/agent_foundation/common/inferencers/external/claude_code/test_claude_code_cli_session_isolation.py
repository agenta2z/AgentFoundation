#!/usr/bin/env python3
"""Claude Code CLI Inferencer — Session Isolation & Resume Integration Tests.

Verifies that multi-turn conversation memory and session isolation work
correctly by telling different secrets in different sessions, then
resuming each to verify the correct secret is recalled without cross-leak.

Run:
    /opt/homebrew/anaconda3/bin/python -m pytest \\
        test/agent_foundation/common/inferencers/external/claude_code/test_claude_code_cli_session_isolation.py -v

    # Or directly:
    /opt/homebrew/anaconda3/bin/python \\
        test/agent_foundation/common/inferencers/external/claude_code/test_claude_code_cli_session_isolation.py -v

Prerequisites:
    - Claude Code CLI available (auto-detected)
    - Proximity proxy running on port 29576
"""

import os
import sys
import time
import unittest

# Auto-add AgentFoundation/src and RichPythonUtils/src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_foundation_root = os.path.normpath(
    os.path.join(_script_dir, "..", "..", "..", "..", "..")
)
_src_dir = os.path.join(_agent_foundation_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)


MODEL = os.environ.get("CLAUDE_TEST_MODEL", "sonnet")
TARGET_PATH = os.environ.get("CLAUDE_TEST_TARGET_PATH", "/tmp")


def _create_inferencer():
    """Create a fresh ClaudeCodeCliInferencer with no session state."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeCliInferencer,
    )

    return ClaudeCodeCliInferencer(
        target_path=TARGET_PATH,
        model_name=MODEL,
        auto_resume=True,
    )


def _send(inferencer, message: str) -> tuple:
    """Send a message and return (output_text, session_id)."""
    response = inferencer(message)
    output = response.output if response.success else ""
    session_id = getattr(response, "session_id", None)
    return output, session_id


def _tell_secret(secret_word: str) -> tuple:
    """Create a new session, tell it a secret, return (response, session_id)."""
    inf = _create_inferencer()
    output, session_id = _send(
        inf,
        f'I am going to tell you a secret word. The secret word is "{secret_word}". '
        "Please remember it. Reply with just: Understood, the secret is [word].",
    )
    return output, session_id


def _ask_secret(session_id: str) -> str:
    """Resume a session and ask for the secret. Returns the response text."""
    inf = _create_inferencer()
    inf.active_session_id = session_id
    output, _ = _send(
        inf,
        "What is the secret word I told you earlier? "
        "Reply with just the single word, nothing else.",
    )
    return output


class TestClaudeCodeCliSessionIsolation(unittest.TestCase):
    """Integration tests for session isolation and resume in Claude Code CLI."""

    def test_two_sessions_isolated_secrets(self):
        """Two sessions with different secrets recall their own secret only."""
        # Create two sessions with different secrets
        resp_a, sid_a = _tell_secret("banana")
        self.assertIsNotNone(sid_a, "Session A must return a session_id")
        self.assertIn("banana", resp_a.lower(), "Session A should confirm 'banana'")

        resp_b, sid_b = _tell_secret("dragon")
        self.assertIsNotNone(sid_b, "Session B must return a session_id")
        self.assertIn("dragon", resp_b.lower(), "Session B should confirm 'dragon'")

        self.assertNotEqual(sid_a, sid_b, "Session IDs must be different")

        # Resume each session — should recall its own secret
        recall_a = _ask_secret(sid_a)
        self.assertIn("banana", recall_a.lower(), "Session A should recall 'banana'")
        self.assertNotIn("dragon", recall_a.lower(), "Session A must NOT leak 'dragon'")

        recall_b = _ask_secret(sid_b)
        self.assertIn("dragon", recall_b.lower(), "Session B should recall 'dragon'")
        self.assertNotIn("banana", recall_b.lower(), "Session B must NOT leak 'banana'")

    def test_three_sessions_with_chitchat(self):
        """Three sessions with secrets survive interleaved chitchat."""
        # Create three sessions
        _, sid_1 = _tell_secret("piano")
        self.assertIsNotNone(sid_1)
        _, sid_2 = _tell_secret("rocket")
        self.assertIsNotNone(sid_2)
        _, sid_3 = _tell_secret("mango")
        self.assertIsNotNone(sid_3)

        # Chitchat in session 1 (shouldn't displace the secret)
        inf_chat = _create_inferencer()
        inf_chat.active_session_id = sid_1
        _send(inf_chat, "What is 2 + 2? Just say the number.")

        # Now recall secrets from all three
        recall_1 = _ask_secret(sid_1)
        self.assertIn("piano", recall_1.lower(), "Session 1 should recall 'piano'")

        recall_2 = _ask_secret(sid_2)
        self.assertIn("rocket", recall_2.lower(), "Session 2 should recall 'rocket'")

        recall_3 = _ask_secret(sid_3)
        self.assertIn("mango", recall_3.lower(), "Session 3 should recall 'mango'")

    def test_new_session_does_not_recall_old_secret(self):
        """A brand new session should NOT know secrets from other sessions."""
        _, sid = _tell_secret("telescope")
        self.assertIsNotNone(sid)

        # Verify the original session knows it
        recall = _ask_secret(sid)
        self.assertIn("telescope", recall.lower())

        # A new session should NOT know "telescope"
        inf_new = _create_inferencer()
        output, _ = _send(
            inf_new,
            "What is the secret word? Reply with just the word if you know it, "
            "or say 'I don\\'t know any secret word' if you don\\'t.",
        )
        self.assertNotIn(
            "telescope", output.lower(),
            "New session must NOT know 'telescope' from another session",
        )

    def test_session_resume_after_multiple_turns(self):
        """Secret survives multiple turns of unrelated conversation."""
        # Tell the secret
        _, sid = _tell_secret("sunflower")
        self.assertIsNotNone(sid)

        # Two turns of unrelated chitchat
        inf = _create_inferencer()
        inf.active_session_id = sid
        _send(inf, "What color is the sky? One word.")
        _send(inf, "Name a planet in our solar system. One word.")

        # Now recall the secret from a fresh inferencer
        recall = _ask_secret(sid)
        self.assertIn(
            "sunflower", recall.lower(),
            "Secret 'sunflower' should survive multiple turns of chitchat",
        )


if __name__ == "__main__":
    unittest.main()
