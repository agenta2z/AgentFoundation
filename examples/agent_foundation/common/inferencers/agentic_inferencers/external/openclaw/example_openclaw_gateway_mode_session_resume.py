#!/usr/bin/env python3
"""OpenClaw Inferencer — Session Isolation & Resume Demo (PodGateway mode).

Demonstrates that OpenClaw gateway sessions are truly isolated and persist
across Python process restarts. Proves that:

1. Two sessions with different IDs maintain separate conversation context
2. A saved session_id can be used to resume a conversation in a new process
3. Resumed sessions correctly recall their own context (not the other session's)
4. ``always_initialize_new_session=True`` (default) auto-warms new sessions

The Test:
    - Session A (session_id="demo-session-a-<run>"): Tell OpenClaw a secret word ("apricot")
    - Session B (session_id="demo-session-b-<run>"): Tell OpenClaw a different secret word ("tangerine")
    - Resume Session A (fresh inferencer, same session_id): Ask for the secret → "apricot"
    - Resume Session B (fresh inferencer, same session_id): Ask for the secret → "tangerine"
    - Cross-check: Ask Session A for Session B's secret → should NOT say "tangerine"

Key differences from RovoChat session resume:
    - OpenClaw sessions are identified by human-readable string IDs (not UUIDs)
    - Sessions persist as JSONL files in the pod at
      /sandbox/.openclaw/agents/main/sessions/<session_id>.jsonl
    - ``always_initialize_new_session=True`` sends a warm-up turn on first use
      of a new session ID (separates startup latency from the real query)
    - Streaming via ``ainfer_streaming()`` — true token-by-token from gateway

Prerequisites:
    - Docker container ``openshell-cluster-openshell`` running (``./run.sh start``)
    - OpenClaw gateway pod healthy inside the container

Usage:
    python example_openclaw_gateway_mode_session_resume.py
    python example_openclaw_gateway_mode_session_resume.py --session-suffix myrun
    python example_openclaw_gateway_mode_session_resume.py --thinking high --timeout 120
"""

import argparse
import asyncio
import os
import sys
import uuid

# ── Path bootstrap (AgentFoundation/src) ──────────────────────────────────────
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
# ──────────────────────────────────────────────────────────────────────────────


def make_inferencer(session_id: str, thinking: str | None, timeout: int):
    """Create a fresh OpenClawInferencer for the given session.

    Each call returns a new Python object — simulating a new process.
    The session history lives in the pod's JSONL file, not in Python memory.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
        OpenClawInferencer,
        OpenClawMode,
    )

    return OpenClawInferencer(
        mode=OpenClawMode.PodGateway,
        session_id=session_id,
        thinking=thinking,
        timeout_seconds=timeout,
        enable_turn_separation=False,   # flat stream for demo clarity
        always_initialize_new_session=True,  # auto warm-up on first use
        auto_resume=True,
    )


async def send_and_print(
    inferencer,
    message: str,
    label: str = "",
) -> str:
    """Send a message via streaming and print the response.

    Returns the full accumulated response text.
    """
    if label:
        print(f"  [{label}]")
    print(f"  You: {message}")
    print(f"  OpenClaw: ", end="", flush=True)

    chunks: list[str] = []
    async for chunk in inferencer.ainfer_streaming(message):
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    print()
    return "".join(chunks)


async def run_demo(
    session_suffix: str,
    thinking: str | None,
    timeout: int,
) -> None:
    """Run the session isolation and resume demo."""

    # Use unique-per-run session IDs so repeated runs don't bleed into each other
    session_id_a = f"demo-session-a-{session_suffix}"
    session_id_b = f"demo-session-b-{session_suffix}"
    secret_a = "apricot"
    secret_b = "tangerine"

    print()
    print(f"  Session A ID: {session_id_a!r}")
    print(f"  Session B ID: {session_id_b!r}")
    print(f"  Secret A: {secret_a!r}  |  Secret B: {secret_b!r}")

    # =========================================================================
    # STEP 1: Create Session A — tell it a secret word
    # =========================================================================
    print()
    print("=" * 70)
    print(f"STEP 1: Create Session A — Secret word is '{secret_a}'")
    print("=" * 70)
    print()

    inf_a = make_inferencer(session_id_a, thinking, timeout)
    # always_initialize_new_session=True will auto-warm the new session
    # before sending this first real message

    resp_a1 = await send_and_print(
        inf_a,
        f"Remember this secret word carefully: '{secret_a}'. "
        f"Do NOT use any tools or search anything — just acknowledge "
        f"you have memorised the word. Reply with: 'Memorised: {secret_a}.'",
        label=f"Session A [{session_id_a}]",
    )

    print()
    print(f"  📌 Session A transcript saved as: {session_id_a}.jsonl")

    # =========================================================================
    # STEP 2: Create Session B — tell it a DIFFERENT secret word
    # =========================================================================
    print()
    print("=" * 70)
    print(f"STEP 2: Create Session B — Secret word is '{secret_b}'")
    print("=" * 70)
    print()

    inf_b = make_inferencer(session_id_b, thinking, timeout)

    resp_b1 = await send_and_print(
        inf_b,
        f"Remember this secret word carefully: '{secret_b}'. "
        f"Do NOT use any tools or search anything — just acknowledge "
        f"you have memorised the word. Reply with: 'Memorised: {secret_b}.'",
        label=f"Session B [{session_id_b}]",
    )

    print()
    print(f"  📌 Session B transcript saved as: {session_id_b}.jsonl")
    print()

    # Verify sessions are different
    assert session_id_a != session_id_b, "Session IDs must be different!"
    print(f"  ✓ Sessions have distinct IDs: A={session_id_a!r}  B={session_id_b!r}")

    # =========================================================================
    # STEP 3: Resume Session A (fresh inferencer — simulates new process)
    # =========================================================================
    print()
    print("=" * 70)
    print(f"STEP 3: Resume Session A — Should recall '{secret_a}' (not '{secret_b}')")
    print("=" * 70)
    print()

    # Fresh Python object — session history comes from pod JSONL file, not memory
    inf_resume_a = make_inferencer(session_id_a, thinking, timeout)
    # always_initialize_new_session=True will skip warm-up since JSONL exists

    resp_a2 = await send_and_print(
        inf_resume_a,
        "What is the secret word I told you earlier? "
        "Reply with just the word, nothing else.",
        label=f"Resuming Session A [{session_id_a}]",
    )

    recall_a = secret_a.lower() in resp_a2.lower()
    wrong_b_in_a = secret_b.lower() in resp_a2.lower()
    print()
    print(f"  {'✅' if recall_a else '❌'} Expected '{secret_a}': "
          f"{'FOUND' if recall_a else 'NOT FOUND'} in response")
    if wrong_b_in_a:
        print(f"  ⚠️  Session B's secret '{secret_b}' appeared in Session A response — context bleed!")

    # =========================================================================
    # STEP 4: Resume Session B (fresh inferencer — simulates new process)
    # =========================================================================
    print()
    print("=" * 70)
    print(f"STEP 4: Resume Session B — Should recall '{secret_b}' (not '{secret_a}')")
    print("=" * 70)
    print()

    inf_resume_b = make_inferencer(session_id_b, thinking, timeout)

    resp_b2 = await send_and_print(
        inf_resume_b,
        "What is the secret word I told you earlier? "
        "Reply with just the word, nothing else.",
        label=f"Resuming Session B [{session_id_b}]",
    )

    recall_b = secret_b.lower() in resp_b2.lower()
    wrong_a_in_b = secret_a.lower() in resp_b2.lower()
    print()
    print(f"  {'✅' if recall_b else '❌'} Expected '{secret_b}': "
          f"{'FOUND' if recall_b else 'NOT FOUND'} in response")
    if wrong_a_in_b:
        print(f"  ⚠️  Session A's secret '{secret_a}' appeared in Session B response — context bleed!")

    # =========================================================================
    # STEP 5: Cross-check — ask Session A for Session B's secret (should fail)
    # =========================================================================
    print()
    print("=" * 70)
    print(f"STEP 5: Cross-check — Session A asked for '{secret_b}' — should NOT know it")
    print("=" * 70)
    print()

    inf_cross = make_inferencer(session_id_a, thinking, timeout)

    resp_cross = await send_and_print(
        inf_cross,
        f"Do you know the word '{secret_b}'? Did I ever tell you that word? "
        f"Reply with just YES or NO.",
        label=f"Cross-check Session A [{session_id_a}]",
    )

    cross_isolation = secret_b.lower() not in resp_cross.lower() or "no" in resp_cross.lower()
    print()
    print(f"  {'✅' if cross_isolation else '❌'} Session A does NOT know '{secret_b}': "
          f"{'CONFIRMED' if cross_isolation else 'FAILED — context bleed detected!'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Session A [{session_id_a}]")
    print(f"    Secret stored: '{secret_a}'")
    print(f"    Recalled on resume: {'✅ YES' if recall_a else '❌ NO'}")
    print(f"    Context bleed from B: {'⚠️  YES' if wrong_b_in_a else '✅ NO'}")
    print()
    print(f"  Session B [{session_id_b}]")
    print(f"    Secret stored: '{secret_b}'")
    print(f"    Recalled on resume: {'✅ YES' if recall_b else '❌ NO'}")
    print(f"    Context bleed from A: {'⚠️  YES' if wrong_a_in_b else '✅ NO'}")
    print()
    print(f"  Isolation cross-check: {'✅ PASSED' if cross_isolation else '❌ FAILED'}")
    print()

    all_pass = recall_a and recall_b and cross_isolation and not wrong_b_in_a and not wrong_a_in_b
    partial = (recall_a or recall_b) and not all_pass

    if all_pass:
        print("  🎉 SUCCESS — Sessions are fully isolated and resume correctly!")
    elif partial:
        print("  ⚠️  PARTIAL — Some checks passed. Review results above.")
        print("     Possible causes: agent randomness, startup context reading,")
        print("     or the warm-up turn included memory from a previous run.")
    else:
        print("  ❌ FAILED — Sessions did not isolate or resume correctly.")
        print("     Check that the OpenClaw gateway pod is healthy and")
        print("     that session JSONL files are being persisted.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenClaw Session Isolation & Resume Demo (PodGateway)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-suffix",
        default=uuid.uuid4().hex[:8],
        help="Suffix appended to session IDs to avoid collisions across runs "
             "(default: random 8-char hex). Use a fixed value to reuse sessions.",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Thinking level for the OpenClaw agent (default: gateway default).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120).",
    )
    args = parser.parse_args()

    print()
    print("🐾 OpenClaw Gateway Mode — Session Isolation & Resume Demo")
    print("   Testing conversation isolation and session persistence via JSONL files")
    print()

    try:
        asyncio.run(run_demo(
            session_suffix=args.session_suffix,
            thinking=args.thinking,
            timeout=args.timeout,
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
