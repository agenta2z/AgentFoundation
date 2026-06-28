"""Offline unit tests for the MetaMate inferencers.

These run WITHOUT the live MetaMate service, buck, or ``msl.metamate`` — they
exercise the pure, in-process surface: CLI command construction + output
parsing, SDK session/option handling, and the shared ``common`` parsing
helpers. They also lock in four fixes:

- B: ``query_metamate``/``common.parse_assistant_text`` read
  ``inline_reasoning.content`` (the real Thrift field), not ``markdown_content``.
- C: ``MetamateSDKInferencer.reset_session()`` clears the MetaMate-specific
  ``_conversation_uuid`` / ``_conversation_fbid``.
- D: ``MetamateSDKInferencer(api_key=None)`` constructs without crashing.
- F: ``MetamateCliInferencer.construct_command`` shell-quotes ``--api-key`` /
  ``--agent-name`` (not just ``--query``).
"""

import shlex
from types import SimpleNamespace as NS

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
    MetamateCliInferencer,
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
    common,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
)

_DELIM = "-" * 72


# ── CLI: construct_command ────────────────────────────────────────────────

def test_cli_construct_command_basic():
    cli = MetamateCliInferencer(api_key="key123")
    cmd = cli.construct_command("hello world")
    assert "buck run " in cmd and ":query_metamate --" in cmd
    assert f"--query {shlex.quote('hello world')}" in cmd
    assert "--api-key key123" in cmd
    # not deep research by default → no --deep-research, no --agent-name
    assert "--deep-research" not in cmd
    assert "--agent-name" not in cmd


def test_cli_construct_command_deep_research_defaults_agent():
    cli = MetamateCliInferencer(deep_research=True)
    cmd = cli.construct_command("q")
    assert "--deep-research" in cmd
    assert f"--agent-name {common.MetamateAgent.DEEP_RESEARCH.value}" in cmd


def test_cli_construct_command_timeout_and_extra_args():
    cli = MetamateCliInferencer(timeout_seconds=42, extra_cli_args=["--foo", "bar"])
    cmd = cli.construct_command("q")
    assert "--timeout 42" in cmd
    assert cmd.rstrip().endswith("--foo bar")


def test_cli_construct_command_shell_quotes_untrusted_values():
    # F: api_key / agent_name must be shell-quoted, like --query.
    cli = MetamateCliInferencer(api_key="k$(rm -rf /)", agent_name="a;b c")
    cmd = cli.construct_command("p`whoami`")
    assert shlex.quote("k$(rm -rf /)") in cmd
    assert shlex.quote("a;b c") in cmd
    assert shlex.quote("p`whoami`") in cmd
    # the raw dangerous substrings must NOT appear unquoted
    assert "--api-key k$(rm -rf /)" not in cmd


# ── CLI: parse_output ─────────────────────────────────────────────────────

def test_cli_parse_output_extracts_response_block():
    stdout = "\n".join(
        ["[polling] ...", _DELIM, "RESPONSE", _DELIM,
         "The answer is 4.", "Second line.", _DELIM, "metadata: ignore"]
    )
    resp = MetamateCliInferencer().parse_output(stdout, "", 0)
    assert resp.output == "The answer is 4.\nSecond line."
    assert resp.return_code == 0
    assert resp.raw_output == stdout


def test_cli_parse_output_fallback_when_no_marker():
    stdout = "just some text without the marker"
    resp = MetamateCliInferencer().parse_output(stdout, "", 0)
    assert resp.output == "just some text without the marker"


def test_cli_build_session_args_is_single_turn():
    # CLI is single-turn: session args are always empty.
    assert MetamateCliInferencer()._build_session_args("sid-123", True) == ""
    assert MetamateCliInferencer()._build_session_args("", False) == ""


def test_cli_sync_subprocess_timeout_floor():
    # A: the sync subprocess path gets a wall-clock floor from the shared
    # constant, never below the server-side --timeout (timeout_seconds), and
    # an explicit timeout always wins.
    assert MetamateCliInferencer().timeout == max(
        common.DEFAULT_TIMEOUT, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    )
    assert MetamateCliInferencer(timeout_seconds=3600).timeout == 3600  # >= server budget
    assert MetamateCliInferencer(timeout=42).timeout == 42  # explicit wins


# ── SDK: session + option handling ────────────────────────────────────────

def test_sdk_reset_session_clears_conversation_state():
    # C: reset_session must clear the MetaMate-specific conversation handles,
    # not just _session_id, or auto_resume silently resumes the old convo.
    sdk = MetamateSDKInferencer()
    sdk._conversation_uuid = "conv-abc"
    sdk._conversation_fbid = "fbid-xyz"
    sdk._session_id = "conv-abc"
    sdk.reset_session()
    assert sdk._conversation_uuid is None
    assert sdk._conversation_fbid is None
    assert sdk.active_session_id is None


def test_sdk_constructs_with_none_api_key():
    # D: api_key=None must not crash __attrs_post_init__ (it logs api_key[:8]).
    sdk = MetamateSDKInferencer(api_key=None)
    assert sdk.api_key is None


def test_sdk_defaults():
    sdk = MetamateSDKInferencer()
    assert sdk.surface == common.DEFAULT_SURFACE
    assert sdk.mode == common.DEFAULT_MODE
    assert sdk.auto_continue is True
    assert sdk.max_continuations == common.MAX_CONTINUATIONS
    assert sdk.poll_interval_seconds == common.DEFAULT_POLL_INTERVAL


# ── common.parse_assistant_text (getattr duck-typing) ─────────────────────

def _assistant_bridge(block_uuid: str, content) -> list:
    """Build a minimal [message, block] bridge_outputs list for one ASSISTANT block."""
    message_out = NS(message=NS(role="ASSISTANT", block_uuids=[block_uuid], status="COMPLETED"))
    block_out = NS(block=NS(uuid=block_uuid, content=content))
    return [message_out, block_out]


def test_parse_assistant_text_markdown_and_text_string():
    md = _assistant_bridge("b1", NS(markdown=NS(value="# Title")))
    assert common.parse_assistant_text(md) == "# Title"
    ts = _assistant_bridge("b2", NS(text_string=NS(value="plain text")))
    assert common.parse_assistant_text(ts) == "plain text"


def test_parse_assistant_text_agent_message():
    am = _assistant_bridge("b3", NS(agent_message=NS(markdown="agent says hi")))
    assert common.parse_assistant_text(am) == "agent says hi"


def test_parse_assistant_text_inline_reasoning_content():
    # B: inline_reasoning text lives on `.content` (Thrift
    # BlockContentInlineReasoning.content), NOT `.markdown_content`.
    good = _assistant_bridge("b4", NS(inline_reasoning=NS(content="thinking...")))
    assert common.parse_assistant_text(good) == "thinking..."
    # A `.markdown_content`-only object yields nothing (proves the old field
    # name silently dropped text — the bug that was fixed).
    bad = _assistant_bridge("b5", NS(inline_reasoning=NS(markdown_content="dropped")))
    assert common.parse_assistant_text(bad) == ""


def test_parse_assistant_text_code_interpreter():
    ci = _assistant_bridge(
        "b6", NS(code_interpreter=NS(code="print(1)", language="python", output="1", summary="ran"))
    )
    out = common.parse_assistant_text(ci)
    assert "```python\nprint(1)\n```" in out and "1" in out and "ran" in out


def test_parse_assistant_text_excludes_non_assistant_blocks():
    # A USER message's block must be ignored.
    user_msg = NS(message=NS(role="USER", block_uuids=["ub"], status="COMPLETED"))
    user_block = NS(block=NS(uuid="ub", content=NS(markdown=NS(value="user said this"))))
    assert common.parse_assistant_text([user_msg, user_block]) == ""


# ── common: status + continuation helpers ─────────────────────────────────

def test_get_assistant_message_status():
    outs = [NS(message=NS(role="ASSISTANT", status="MessageStatus.COMPLETED", block_uuids=[]))]
    assert common.get_assistant_message_status(outs) == "COMPLETED"
    assert common.get_assistant_message_status([NS(message=None)]) is None


def test_needs_continuation():
    assert common.needs_continuation("Should I proceed with the research?") is True
    assert common.needs_continuation("Which one are you interested in?") is True
    assert common.needs_continuation("Here is the complete answer with all details.") is False
    assert common.needs_continuation("") is False


def test_metamate_agent_enum_values():
    assert common.MetamateAgent.DEEP_RESEARCH.value == "SPACES_DEEP_RESEARCH_AGENT"
    assert common.MetamateAgent.DEFAULT.value == "DEFAULT"
