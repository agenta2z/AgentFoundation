#!/usr/bin/env python3
"""Claude Code SDK Inferencer -- Multi-Turn Session & Isolation Demo.

Proves multi-turn conversation memory and session isolation via the SDK:
  1. Session A: Tell Claude a secret word ("banana") -- via anew_session()
  2. Follow-up in Session A: Ask Claude to recall it -- via ainfer() (auto-resume)
  3. Session B: Tell Claude a different secret ("dragon") -- via anew_session()
  4. Follow-up in Session B: Ask Claude to recall it -- proves no cross-leak
  5. New Session C: Ask for secret -- should NOT know either

Key difference from CLI version:
    - SDK maintains a persistent connection within ``async with`` -- queries
      are fast (no subprocess startup per call).
    - Session management is handled via anew_session() and auto_resume=True.
    - The SDK client handles session context internally, so multi-turn "just works"
      within the same async with block.

Run:
    python examples/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/example_claude_code_sdk_session_resume.py

Prerequisites:
    - claude-agent-sdk package installed (pip install claude-agent-sdk)
    - Claude Code CLI authenticated on this machine
"""

import argparse
import asyncio
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


def _prepare_env():
    """Unset CLAUDECODE to bypass the nested-session guard.

    Note: ANTHROPIC_API_KEY is handled automatically by ClaudeCodeInferencer's
    ``prefer_subscription=True`` default.
    """
    if os.environ.pop("CLAUDECODE", None):
        print("  (Cleared CLAUDECODE env var to allow nested SDK usage)")


async def send_and_print(inferencer, message: str, label: str = "") -> str:
    """Send a message via ainfer and print the response. Returns response text."""
    if label:
        print(f"  [{label}]")
    print(f"  You:    {message}")

    start = time.time()
    result = await inferencer.ainfer(message)
    elapsed = time.time() - start

    output = str(result)
    display = output if len(output) < 200 else output[:200] + "..."
    sid = inferencer.active_session_id
    sid_short = (sid[:12] + "...") if sid else "none"
    print(f"  Claude: {display}")
    print(f"  [{elapsed:.1f}s | session={sid_short}]")
    print()

    return output


async def main_async(args):
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    print()
    print("Claude Code SDK -- Multi-Turn Session & Isolation Demo")
    print("=" * 60)
    print()

    # -- Step 1: Session A with secret "banana" --------------------------

    print("STEP 1: Create Session A -- tell secret 'banana'")
    print("-" * 40)

    async with ClaudeCodeInferencer(
        root_folder=args.root_folder, allowed_tools=[], auto_resume=True
    ) as inf_a:
        response_a1 = await send_and_print(
            inf_a,
            'I am going to tell you a secret word. The secret word is "banana". '
            "Please remember it. Reply with just: Understood, the secret is [word].",
            label="Session A -- New",
        )
        session_id_a = inf_a.active_session_id
        print(f"  Session A ID: {session_id_a}")
        print()

        # -- Step 2: Follow-up in Session A (auto-resume) ----------------

        print("STEP 2: Follow-up in Session A -- recall the secret")
        print("-" * 40)

        response_a2 = await send_and_print(
            inf_a,
            "What is the secret word I told you? Reply with just the word, nothing else.",
            label="Session A -- Follow-up (same connection)",
        )

    # -- Step 3: Session B with secret "dragon" --------------------------

    print("STEP 3: Create Session B -- tell secret 'dragon'")
    print("-" * 40)

    async with ClaudeCodeInferencer(
        root_folder=args.root_folder, allowed_tools=[], auto_resume=True
    ) as inf_b:
        response_b1 = await send_and_print(
            inf_b,
            'I am going to tell you a secret word. The secret word is "dragon". '
            "Please remember it. Reply with just: Understood, the secret is [word].",
            label="Session B -- New",
        )
        session_id_b = inf_b.active_session_id
        print(f"  Session B ID: {session_id_b}")
        print()

        # -- Step 4: Follow-up in Session B -------------------------

        print("STEP 4: Follow-up in Session B -- recall the secret")
        print("-" * 40)

        response_b2 = await send_and_print(
            inf_b,
            "What is the secret word I told you? Reply with just the word, nothing else.",
            label="Session B -- Follow-up (same connection)",
        )

    # -- Step 5: New Session C -- should not know secrets -----------------

    print("STEP 5: New Session C -- should NOT know any secret")
    print("-" * 40)

    async with ClaudeCodeInferencer(
        root_folder=args.root_folder, allowed_tools=[]
    ) as inf_c:
        response_c = await send_and_print(
            inf_c,
            "What is the secret word? Reply with just the word if you know it, "
            'or say "I don\'t know any secret word" if you don\'t.',
            label="Session C -- New (no prior context)",
        )

    # -- Results ----------------------------------------------------------

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    a2_lower = response_a2.lower()
    b2_lower = response_b2.lower()
    c_lower = response_c.lower()

    a_has_banana = "banana" in a2_lower
    a_no_dragon = "dragon" not in a2_lower
    b_has_dragon = "dragon" in b2_lower
    b_no_banana = "banana" not in b2_lower
    c_no_secrets = "banana" not in c_lower and "dragon" not in c_lower

    print(f"  Session A recall 'banana':     {'PASS' if a_has_banana else 'FAIL'}")
    print(f"  Session A no cross-leak:       {'PASS' if a_no_dragon else 'FAIL (leaked dragon)'}")
    print(f"  Session B recall 'dragon':     {'PASS' if b_has_dragon else 'FAIL'}")
    print(f"  Session B no cross-leak:       {'PASS' if b_no_banana else 'FAIL (leaked banana)'}")
    print(f"  Session C no secret knowledge: {'PASS' if c_no_secrets else 'UNEXPECTED (knew a secret)'}")
    print()

    all_pass = a_has_banana and a_no_dragon and b_has_dragon and b_no_banana and c_no_secrets
    if all_pass:
        print("  ALL CHECKS PASSED -- Session isolation and multi-turn work correctly!")
    else:
        print("  Some checks failed -- see details above.")

    print()
    print(f"  Session A ID: {session_id_a}")
    print(f"  Session B ID: {session_id_b}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code SDK -- Multi-Turn Session & Isolation Demo"
    )
    parser.add_argument(
        "-r", "--root-folder",
        default=os.path.expanduser("~"),
        help="Working directory for Claude Code agent (default: home dir)",
    )
    args = parser.parse_args()

    _prepare_env()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
