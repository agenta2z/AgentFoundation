#!/usr/bin/env python3
"""Rovo Dev CLI Inferencer — Session Save & Restore Demo.

Demonstrates that Rovo Dev sessions can be saved and restored by their
UUID, enabling true multi-turn conversations across separate invocations.

The Test:
    1. **Session A**: Tell Rovo Dev a favorite color ("blue")
    2. **Session B**: Tell Rovo Dev a different favorite color ("green")
    3. **Resume Session A** by UUID: Ask for the color -> should say "blue"
    4. **Resume Session B** by UUID: Ask for the color -> should say "green"
    5. **Cross-check**: Prove sessions are truly isolated.

This works because ``acli rovodev legacy --restore <uuid>`` restores a
specific session by its UUID from ``~/.rovodev/sessions/``.

Note:
    The working directory should be a git repository for session workspace
    matching to work correctly.

Prerequisites:
    - ``acli`` installed and in PATH (``brew install atlassian-cli``)
    - ``acli auth login`` completed (authenticated session)

Usage:
    python example_rovodev_session_restore.py
    python example_rovodev_session_restore.py --working-dir /path/to/repo
"""

import argparse
import os
import subprocess
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


def create_inferencer(working_dir: str, output_file: str):
    """Create a RovoDevCliInferencer instance."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
        RovoDevCliInferencer,
    )

    return RovoDevCliInferencer(
        working_dir=working_dir,
        output_file=output_file,
        idle_timeout_seconds=600,
        tool_use_idle_timeout_seconds=600,
    )


def send_and_print(inferencer, message: str, label: str = "") -> str:
    """Send a message and print the response."""
    if label:
        print(f"  [{label}]")
    print(f"  You: {message}")

    start = time.time()
    result = inferencer(message)
    elapsed = time.time() - start

    print(f"  Rovo Dev: {result.output}")
    print(f"  [{elapsed:.1f}s | session={inferencer.active_session_id}]")
    print()
    return result.output


def main():
    parser = argparse.ArgumentParser(
        description="Rovo Dev CLI Inferencer — Session Save & Restore Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--working-dir",
        help="Working directory for the agent (default: temp git repo)",
    )
    args = parser.parse_args()

    if args.working_dir:
        working_dir = args.working_dir
    else:
        # Create a temp git repo (required for session workspace matching)
        working_dir = tempfile.mkdtemp(prefix="rovodev_session_")
        subprocess.run(
            ["git", "init", "-q"], cwd=working_dir, check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "init"],
            cwd=working_dir, check=True, capture_output=True,
        )

    # Verify acli is installed
    import shutil

    if not shutil.which("acli"):
        print("  acli not found. Install with: brew install atlassian-cli")
        sys.exit(1)

    out_file = os.path.join(working_dir, "rovodev_output.txt")

    print()
    print("Rovo Dev Session Save & Restore Demo")
    print(f"   Working dir: {working_dir}")
    print()

    try:
        # =================================================================
        # Phase 1: Create two independent sessions with different colors
        # =================================================================

        print("=" * 70)
        print("Phase 1: Create two sessions with different favorite colors")
        print("=" * 70)
        print()

        # --- Session A ---
        inferencer_a = create_inferencer(working_dir, out_file)
        print("  Creating Session A...")
        result_a = inferencer_a.new_session(
            "My favorite color is BLUE. "
            "Confirm: your favorite color is BLUE."
        )
        print(f"  Rovo Dev: {result_a.output[:80]}")
        session_a_id = inferencer_a.active_session_id
        print(f"  Session A ID: {session_a_id}")
        print()

        if not session_a_id:
            print("  Failed to capture Session A ID. Aborting.")
            sys.exit(1)

        # --- Session B ---
        inferencer_b = create_inferencer(working_dir, out_file)
        print("  Creating Session B...")
        result_b = inferencer_b.new_session(
            "My favorite color is GREEN. "
            "Confirm: your favorite color is GREEN."
        )
        print(f"  Rovo Dev: {result_b.output[:80]}")
        session_b_id = inferencer_b.active_session_id
        print(f"  Session B ID: {session_b_id}")
        print()

        if not session_b_id:
            print("  Failed to capture Session B ID. Aborting.")
            sys.exit(1)

        assert session_a_id != session_b_id, (
            f"Sessions should have different IDs! A={session_a_id}, B={session_b_id}"
        )
        print(f"  Sessions are distinct: A={session_a_id[:12]}... B={session_b_id[:12]}...")
        print()

        # =================================================================
        # Phase 2: Resume each session and verify isolation
        # =================================================================

        print("=" * 70)
        print("Phase 2: Resume sessions by UUID and verify context isolation")
        print("=" * 70)
        print()

        recall_prompt = (
            "What is my favorite color? "
            "Reply with ONLY the color, nothing else."
        )

        # --- Resume Session A ---
        print("  Resuming Session A...")
        inferencer_resume_a = create_inferencer(working_dir, out_file)
        inferencer_resume_a.active_session_id = session_a_id
        response_a = send_and_print(
            inferencer_resume_a, recall_prompt, label="Session A recall"
        )

        # --- Resume Session B ---
        print("  Resuming Session B...")
        inferencer_resume_b = create_inferencer(working_dir, out_file)
        inferencer_resume_b.active_session_id = session_b_id
        response_b = send_and_print(
            inferencer_resume_b, recall_prompt, label="Session B recall"
        )

        # =================================================================
        # Phase 3: Verify results
        # =================================================================

        print("=" * 70)
        print("Phase 3: Verification")
        print("=" * 70)
        print()

        passed = True

        if "BLUE" in response_a.upper():
            print("  [PASS] Session A correctly recalled: BLUE")
        else:
            print(f"  [FAIL] Session A: Expected BLUE, got: {response_a!r}")
            passed = False

        if "GREEN" in response_b.upper():
            print("  [PASS] Session B correctly recalled: GREEN")
        else:
            print(f"  [FAIL] Session B: Expected GREEN, got: {response_b!r}")
            passed = False

        if session_a_id != session_b_id:
            print(f"  [PASS] Session IDs are different (isolated)")
        else:
            print(f"  [FAIL] Session IDs are the same (not isolated!)")
            passed = False

        print()
        if passed:
            print("All checks passed! Sessions are properly isolated and restorable.")
        else:
            print("Some checks failed. See output above.")
            sys.exit(1)

        # =================================================================
        # Bonus: Show session IDs for manual restoration
        # =================================================================

        print()
        print("=" * 70)
        print("Manual session restore commands")
        print("=" * 70)
        print()
        print(f"  # Resume Session A:")
        print(f"  acli rovodev legacy --restore {session_a_id}")
        print()
        print(f"  # Resume Session B:")
        print(f"  acli rovodev legacy --restore {session_b_id}")
        print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
    except Exception as e:
        print(f"\n  Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
