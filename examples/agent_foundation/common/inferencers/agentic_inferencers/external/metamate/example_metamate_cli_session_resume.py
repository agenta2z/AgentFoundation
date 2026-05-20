#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""MetaMate CLI Inferencer — Single-Turn Limitation Demo.

The MetaMate CLI (``query_metamate``) is intentionally single-turn — the
underlying ``engine_start_v2`` GraphQL call is fired once per CLI invocation
and no conversation state is persisted across ``buck run`` calls. The
inferencer's ``_build_session_args()`` returns an empty string and logs a
warning whenever ``session_id`` is passed.

This demo proves that behavior empirically so users don't try to chain
context through the CLI and waste time debugging:

  1. Session A: Tell MetaMate a secret word ("banana")
  2. Reuse the *same* inferencer to ask "What's the secret?" — MetaMate
     should NOT recall "banana" because each CLI invocation starts a fresh
     server-side conversation.

For real multi-turn workflows, use ``MetamateSDKInferencer`` — see
``example_metamate_sdk_session_resume.py``.

Run (via buck — recommended):
    buck2 run @//mode/dbgo \\
      //_tony_dev/CoreProjects/AgentFoundation/examples/agent_foundation/common/inferencers/agentic_inferencers/external/metamate:example_metamate_cli_session_resume

Prerequisites:
    - ``query_metamate`` Buck target reachable
    - MetaMate backend reachable
"""

import argparse
import os
import sys
import time

# Auto-add AgentFoundation/src and RichPythonUtils/src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_foundation_root = os.path.normpath(
    os.path.join(_script_dir, "..", "..", "..", "..", "..", "..", "..")
)
_src_dir = os.path.join(_agent_foundation_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)


def create_inferencer(args):
    """Create a fresh MetamateCliInferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateCliInferencer,
    )

    return MetamateCliInferencer(
        agent_name=args.agent_name,
        deep_research=False,  # session demo should be fast; skip deep research
        timeout_seconds=args.timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )


def send_and_print(inferencer, message: str, label: str = ""):
    """Send a message via sync infer and print the response. Returns the text."""
    if label:
        print(f"  [{label}]")
    print(f"  You:      {message}")

    start = time.time()
    response = inferencer(message)
    elapsed = time.time() - start

    output_text = getattr(response, "output", None) or str(response)
    return_code = getattr(response, "return_code", None)

    display = output_text if len(output_text) < 250 else output_text[:250] + "..."
    print(f"  MetaMate: {display}")
    print(f"  [{elapsed:.1f}s | return_code={return_code}]")
    print()

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="MetaMate CLI — Single-Turn Limitation Demo"
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Optional MetaMate agent name override.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Per-CLI-invocation timeout (default: 180s).",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=180,
        help="Streaming idle timeout (default: 180s).",
    )
    args = parser.parse_args()

    print()
    print("MetaMate CLI — Single-Turn Limitation Demo")
    print("=" * 60)
    print(
        "The MetaMate CLI is intentionally single-turn. This demo proves\n"
        "that even reusing the same inferencer does NOT preserve context\n"
        "between calls — every ``buck run`` starts a fresh server-side\n"
        "conversation. For multi-turn workflows use MetamateSDKInferencer.\n"
    )

    inferencer = create_inferencer(args)

    # -- Step 1: Tell MetaMate a secret ----------------------------------
    print("STEP 1: Tell MetaMate a secret word in the FIRST CLI call")
    print("-" * 60)
    send_and_print(
        inferencer,
        'My secret word is "banana". Please remember it. '
        "Reply with: Understood, the secret is [word].",
        label="Call 1 — set secret",
    )

    # -- Step 2: Ask MetaMate to recall — should NOT know it -------------
    print("STEP 2: Reuse the SAME inferencer; ask MetaMate to recall")
    print("-" * 60)
    response2 = send_and_print(
        inferencer,
        "What is the secret word I told you in our previous exchange? "
        "Reply with just the word, or say \"I don't know\" if you weren't told one.",
        label="Call 2 — recall (expected: NO memory of 'banana')",
    )

    # -- Results ---------------------------------------------------------
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    has_banana = "banana" in response2.lower()

    if has_banana:
        print("  UNEXPECTED: MetaMate recalled 'banana' across CLI calls.")
        print("  This contradicts the documented single-turn behavior.")
        print("  Check whether server-side session state was implicitly")
        print("  attached (e.g. via cookies on the API key).")
    else:
        print("  EXPECTED: MetaMate did NOT recall 'banana' — confirms the")
        print("  CLI is single-turn. Each call starts a fresh conversation.")

    print()
    print("  For multi-turn workflows, see:")
    print("    example_metamate_sdk_session_resume.py")
    print()


if __name__ == "__main__":
    main()
