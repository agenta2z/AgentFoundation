#!/usr/bin/env python3
"""Kiro CLI Inferencer — Session Resume Demo.

Demonstrates multi-turn conversation memory using Kiro CLI's ``--resume``
flag, which resumes the most recent session from the working directory.

The Test:
  1. Turn 1: Tell Kiro a secret word ("banana") in a fresh session
  2. Turn 2: Resume the session and ask for the secret — should say "banana"
  3. Turn 3: Ask a follow-up in the same resumed session
  4. New session: Start fresh — should NOT know the secret

Note on Kiro CLI session model:
    Unlike Claude Code CLI (which returns a session UUID for explicit resume),
    Kiro CLI manages sessions per-directory. ``--resume`` always resumes the
    most recent session from the current working directory. This means:
    - Each working directory has its own session history
    - ``--resume`` picks up where the last conversation left off
    - There's no way to resume a specific session by ID via CLI flags

Run:
    python examples/agent_foundation/common/inferencers/agentic_inferencers/external/kiro/example_kiro_cli_session_resume.py

Prerequisites:
    - ``kiro-cli`` installed and in PATH
    - Authenticated (``kiro-cli login``)
"""

import argparse
import os
import sys
import tempfile
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


def create_inferencer(target_path: str, model: str = "auto"):
    """Create a KiroCliInferencer with auto_resume enabled."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro import (
        KiroCliInferencer,
    )

    return KiroCliInferencer(
        target_path=target_path,
        model_name=model,
        auto_resume=True,
    )


def send_and_print(inferencer, message: str, label: str = "") -> str:
    """Send a message via sync infer and print the response.

    Returns the response text.
    """
    if label:
        print(f"  [{label}]")
    print(f"  You:  {message}")

    start = time.time()
    response = inferencer(message)
    elapsed = time.time() - start

    output = response.output if response.success else (response.error or "Error")

    # Truncate long responses for readability
    display = output if len(output) < 200 else output[:200] + "..."
    print(f"  Kiro: {display}")
    print(f"  [{elapsed:.1f}s | success={response.success}]")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Kiro CLI — Session Resume Demo"
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        help="Model name/alias (default: auto)",
    )
    parser.add_argument(
        "-t", "--target-path",
        help="Working directory for Kiro CLI (default: temp dir)",
    )
    args = parser.parse_args()

    # Use a dedicated temp directory so sessions don't collide
    session_dir = args.target_path or tempfile.mkdtemp(prefix="kiro_session_demo_")

    # Auth check
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro import (
        KiroCliInferencer,
    )

    auth_check = KiroCliInferencer(target_path=session_dir)
    if not auth_check.check_auth(timeout=15.0):
        print("❌ Not authenticated. Run 'kiro-cli login' first.")
        return

    print()
    print("🔐 Kiro CLI — Session Resume Demo")
    print("=" * 60)
    print(f"   Working dir: {session_dir}")
    print(f"   Model:       {args.model}")
    print()

    try:
        # ── Step 1: Start a new session with a secret ────────────────

        print("STEP 1: Start a new session — tell Kiro a secret")
        print("-" * 40)

        inferencer = create_inferencer(session_dir, args.model)
        response_1 = send_and_print(
            inferencer,
            'I am going to tell you a secret word. The secret word is "banana". '
            "Please remember it. Reply with: Understood, the secret is [word].",
            label="Turn 1 — New session",
        )

        # ── Step 2: Resume and ask for the secret ────────────────────

        print("STEP 2: Resume the session — ask for the secret")
        print("-" * 40)

        # Create a new inferencer pointing to the same directory.
        # With auto_resume=True, the inferencer will use --resume
        # to continue the most recent session from this directory.
        #
        # We need to give it a fake active_session_id so the session
        # management logic triggers the --resume flag.
        inferencer_resume = create_inferencer(session_dir, args.model)
        inferencer_resume.active_session_id = "resumed"

        response_2 = send_and_print(
            inferencer_resume,
            "What is the secret word I told you? Reply with just the word.",
            label="Turn 2 — Resumed session",
        )

        # ── Step 3: Another follow-up in the same session ────────────

        print("STEP 3: Continue the resumed session — follow-up question")
        print("-" * 40)

        response_3 = send_and_print(
            inferencer_resume,
            "Can you spell that secret word backwards?",
            label="Turn 3 — Same resumed session",
        )

        # ── Step 4: New session — should NOT know the secret ─────────

        print("STEP 4: Start a completely new session — should NOT know the secret")
        print("-" * 40)

        # Use a DIFFERENT temp directory to ensure no session carryover
        fresh_dir = tempfile.mkdtemp(prefix="kiro_fresh_")
        inferencer_fresh = create_inferencer(fresh_dir, args.model)

        response_4 = send_and_print(
            inferencer_fresh,
            "What is the secret word? Reply with just the word if you know it, "
            'or say "I don\'t know any secret word" if you don\'t.',
            label="New session — no prior context",
        )

        # ── Results ──────────────────────────────────────────────────

        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print()

        r2_lower = response_2.lower()
        r4_lower = response_4.lower()

        recall_ok = "banana" in r2_lower
        isolation_ok = "banana" not in r4_lower

        print(f"  Session recall 'banana':       {'✅ PASS' if recall_ok else '❌ FAIL'}")
        print(f"  New session isolation:          {'✅ PASS' if isolation_ok else '⚠️  UNEXPECTED'}")
        print()

        if recall_ok and isolation_ok:
            print("  🎉 Session resume and isolation work correctly!")
        elif recall_ok:
            print("  ⚠️  Resume works but new session unexpectedly knew the secret.")
        else:
            print("  ⚠️  Session resume did not recall the secret.")
            print("     This may be a model behavior issue, not an inferencer bug.")

        print()
        print(f"  Session dir:  {session_dir}")
        print(f"  Fresh dir:    {fresh_dir}")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
