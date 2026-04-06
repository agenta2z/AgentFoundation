#!/usr/bin/env python3
"""Claude Code CLI Inferencer — Session Resume & Isolation Demo.

Proves multi-turn conversation memory and session isolation by:
  1. Session A: Tell Claude a secret word ("banana")
  2. Session B: Tell Claude a different secret word ("dragon")
  3. Resume Session A: Ask for the secret — should say "banana", NOT "dragon"
  4. Resume Session B: Ask for the secret — should say "dragon", NOT "banana"
  5. New Session C: Ask for a secret — should NOT know either secret

This uses SYNC mode (not streaming) because session_id is only available
in the JSON output from --output-format json. Streaming mode does not
return session metadata.

Run:
    /opt/homebrew/anaconda3/bin/python examples/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/example_claude_code_cli_session_resume.py

Prerequisites:
    - Claude Code CLI available (auto-detected: native binary or Node.js)
    - Proximity proxy running on port 29576 (for Vertex routing)
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
    """Create a fresh ClaudeCodeCliInferencer (no session state)."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeCliInferencer,
    )

    return ClaudeCodeCliInferencer(
        target_path=args.target_path,
        model_name=args.model,
        auto_resume=True,
    )


def send_and_print(inferencer, message, label=""):
    """Send a message via sync infer and print the response.

    Returns (response_text, session_id).
    """
    if label:
        print(f"  [{label}]")
    print(f"  You:    {message}")

    start = time.time()
    response = inferencer(message)
    elapsed = time.time() - start

    output = response.output if response.success else response.get("error", "Error")
    session_id = response.session_id if hasattr(response, "session_id") else None

    # Truncate long responses for readability
    display = output if len(output) < 200 else output[:200] + "..."
    print(f"  Claude: {display}")
    print(f"  [{elapsed:.1f}s | session={session_id[:12] + '...' if session_id else 'none'}]")
    print()

    return output, session_id


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code CLI — Session Resume & Isolation Demo"
    )
    parser.add_argument(
        "-m", "--model",
        default="sonnet",
        help="Model name/alias (default: sonnet)",
    )
    parser.add_argument(
        "-t", "--target-path",
        default="/tmp",
        help="Working directory for Claude Code CLI (default: /tmp)",
    )
    args = parser.parse_args()

    print()
    print("🔐 Claude Code CLI — Session Resume & Isolation Demo")
    print("=" * 60)
    print()

    # ── Step 1: Create Session A with secret "banana" ────────────────

    print("STEP 1: Create Session A")
    print("-" * 40)

    inf_a = create_inferencer(args)
    _, session_id_a = send_and_print(
        inf_a,
        'I am going to tell you a secret word. The secret word is "banana". '
        "Please remember it. Reply with just: Understood, the secret is [word].",
        label="Session A — New",
    )

    if not session_id_a:
        print("❌ FAILED: No session_id returned for Session A.")
        print("   Session resume requires --output-format json (sync mode).")
        print("   Make sure Claude Code CLI is working properly.")
        return

    print(f"  ✅ Session A created: {session_id_a}")
    print()

    # ── Step 2: Create Session B with secret "dragon" ────────────────

    print("STEP 2: Create Session B")
    print("-" * 40)

    inf_b = create_inferencer(args)
    _, session_id_b = send_and_print(
        inf_b,
        'I am going to tell you a secret word. The secret word is "dragon". '
        "Please remember it. Reply with just: Understood, the secret is [word].",
        label="Session B — New",
    )

    if not session_id_b:
        print("❌ FAILED: No session_id returned for Session B.")
        return

    print(f"  ✅ Session B created: {session_id_b}")
    print()

    # ── Step 3: Resume Session A — should recall "banana" ────────────

    print("STEP 3: Resume Session A — recall the secret")
    print("-" * 40)

    inf_resume_a = create_inferencer(args)
    inf_resume_a.active_session_id = session_id_a  # Set the session to resume
    response_a, _ = send_and_print(
        inf_resume_a,
        "What is the secret word I told you? Reply with just the word, nothing else.",
        label=f"Resuming Session A ({session_id_a[:12]}...)",
    )

    # ── Step 4: Resume Session B — should recall "dragon" ────────────

    print("STEP 4: Resume Session B — recall the secret")
    print("-" * 40)

    inf_resume_b = create_inferencer(args)
    inf_resume_b.active_session_id = session_id_b  # Set the session to resume
    response_b, _ = send_and_print(
        inf_resume_b,
        "What is the secret word I told you? Reply with just the word, nothing else.",
        label=f"Resuming Session B ({session_id_b[:12]}...)",
    )

    # ── Step 5: New Session C — should NOT know any secret ───────────

    print("STEP 5: New Session C — should NOT know any secret")
    print("-" * 40)

    inf_c = create_inferencer(args)
    response_c, _ = send_and_print(
        inf_c,
        "What is the secret word? Reply with just the word if you know it, "
        'or say "I don\'t know any secret word" if you don\'t.',
        label="Session C — New (no prior context)",
    )

    # ── Results ──────────────────────────────────────────────────────

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    response_a_lower = response_a.lower()
    response_b_lower = response_b.lower()
    response_c_lower = response_c.lower()

    # Check Session A
    a_has_banana = "banana" in response_a_lower
    a_no_dragon = "dragon" not in response_a_lower
    a_pass = a_has_banana and a_no_dragon

    # Check Session B
    b_has_dragon = "dragon" in response_b_lower
    b_no_banana = "banana" not in response_b_lower
    b_pass = b_has_dragon and b_no_banana

    # Check Session C (should not know either secret)
    c_no_secrets = "banana" not in response_c_lower and "dragon" not in response_c_lower

    print(f"  Session A recall 'banana':     {'✅ PASS' if a_has_banana else '❌ FAIL'}")
    print(f"  Session A no cross-leak:       {'✅ PASS' if a_no_dragon else '❌ FAIL (leaked dragon)'}")
    print(f"  Session B recall 'dragon':     {'✅ PASS' if b_has_dragon else '❌ FAIL'}")
    print(f"  Session B no cross-leak:       {'✅ PASS' if b_no_banana else '❌ FAIL (leaked banana)'}")
    print(f"  Session C no secret knowledge: {'✅ PASS' if c_no_secrets else '⚠️  UNEXPECTED (knew a secret)'}")
    print()

    all_pass = a_pass and b_pass and c_no_secrets
    if all_pass:
        print("  🎉 ALL CHECKS PASSED — Session isolation and resume work correctly!")
    else:
        print("  ⚠️  Some checks failed — see details above.")
        if not a_pass or not b_pass:
            print("     Session resume may not be working with this Claude Code version.")

    print()
    print(f"  Session A ID: {session_id_a}")
    print(f"  Session B ID: {session_id_b}")
    print()


if __name__ == "__main__":
    main()
