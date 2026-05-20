#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""MetaMate SDK Inferencer — Multi-Turn Session & Isolation Demo.

Proves multi-turn conversation memory and session isolation via the SDK:
  1. Session A: Tell MetaMate a secret word ("banana") via ``anew_session()``
  2. Follow-up in Session A: Ask MetaMate to recall it via ``aresume_session()``
  3. Session B: Tell MetaMate a different secret ("dragon") via ``anew_session()``
  4. Follow-up in Session B: Ask MetaMate to recall it — proves no cross-leak
  5. New Session C: Ask for any secret — should NOT know either

The SDK chains turns via ``conversation_uuid`` / ``conversation_fbid``. Each
call still uses a fresh ``MetamateGraphQLClient`` underneath, but the
conversation IDs are passed through to ``engine_start_v2`` so the server
knows which existing conversation to continue.

Run (via buck — required because of ``//msl/metamate/cli:metamate_graphql`` dep):
    buck2 run @//mode/dbgo \\
      //_tony_dev/CoreProjects/AgentFoundation/examples/agent_foundation/common/inferencers/agentic_inferencers/external/metamate:example_metamate_sdk_session_resume

Prerequisites:
    - MetaMate backend reachable
    - ``msl.metamate.cli.metamate_graphql`` Buck dep
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


def make_inferencer(args, auto_resume: bool = False):
    """Create a MetamateSDKInferencer.

    Defaults to ``auto_resume=False`` so the example uses explicit
    ``aresume_session(prompt, session_id=...)`` — the pattern that mirrors
    how the real cross-session integration tests drive multi-turn calls.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateSDKInferencer,
    )

    return MetamateSDKInferencer(
        agent_name=args.agent_name,
        total_timeout_seconds=args.total_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        auto_continue=True,
        max_continuations=5,
        auto_resume=auto_resume,
    )


async def send_and_print(
    inferencer,
    message: str,
    label: str = "",
    resume_session_id: str | None = None,
) -> str:
    """Send a message and print the response. Returns response text."""
    if label:
        print(f"  [{label}]")
    print(f"  You:      {message}")

    start = time.time()
    if resume_session_id is None:
        result = await inferencer.anew_session(message)
    else:
        result = await inferencer.aresume_session(message, session_id=resume_session_id)
    elapsed = time.time() - start

    output = str(result)
    display = output if len(output) < 250 else output[:250] + "..."
    sid = inferencer.active_session_id
    sid_short = (sid[:12] + "...") if sid else "none"
    print(f"  MetaMate: {display}")
    print(f"  [{elapsed:.1f}s | session={sid_short}]")
    print()

    return output


async def main_async(args):
    print()
    print("MetaMate SDK — Multi-Turn Session & Isolation Demo")
    print("=" * 60)
    print()

    # -- Step 1: Session A with secret "banana" --------------------------
    print("STEP 1: Create Session A — tell secret 'banana'")
    print("-" * 40)

    inf_a = make_inferencer(args)
    await send_and_print(
        inf_a,
        'My secret word is "banana". Please remember it. '
        "Reply with: Understood, the secret is [word].",
        label="Session A — New",
    )
    session_id_a = inf_a.active_session_id
    print(f"  Session A ID: {session_id_a}")
    print()

    if not session_id_a:
        print("FAILED: MetaMate SDK did not return a session_id for Session A.")
        return

    # -- Step 2: Follow-up in Session A (explicit aresume_session) -------
    print("STEP 2: Follow-up in Session A — recall the secret")
    print("-" * 40)

    response_a2 = await send_and_print(
        inf_a,
        "What is the secret word I told you? Reply with just the word, nothing else.",
        label="Session A — Follow-up (explicit resume)",
        resume_session_id=session_id_a,
    )

    # -- Step 3: Session B with secret "dragon" --------------------------
    print("STEP 3: Create Session B — tell secret 'dragon'")
    print("-" * 40)

    inf_b = make_inferencer(args)
    await send_and_print(
        inf_b,
        'My secret word is "dragon". Please remember it. '
        "Reply with: Understood, the secret is [word].",
        label="Session B — New",
    )
    session_id_b = inf_b.active_session_id
    print(f"  Session B ID: {session_id_b}")
    print()

    if not session_id_b:
        print("FAILED: MetaMate SDK did not return a session_id for Session B.")
        return

    # -- Step 4: Follow-up in Session B (explicit aresume_session) -------
    print("STEP 4: Follow-up in Session B — recall the secret")
    print("-" * 40)

    response_b2 = await send_and_print(
        inf_b,
        "What is the secret word I told you? Reply with just the word, nothing else.",
        label="Session B — Follow-up (explicit resume)",
        resume_session_id=session_id_b,
    )

    # -- Step 5: New Session C — should not know secrets -----------------
    print("STEP 5: New Session C — should NOT know any secret")
    print("-" * 40)

    inf_c = make_inferencer(args)
    response_c = await send_and_print(
        inf_c,
        "What is the secret word? Reply with just the word if you know it, "
        "or say \"I don't know any secret word\" if you don't.",
        label="Session C — New (no prior context)",
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
    print(
        f"  Session A no cross-leak:       {'PASS' if a_no_dragon else 'FAIL (leaked dragon)'}"
    )
    print(f"  Session B recall 'dragon':     {'PASS' if b_has_dragon else 'FAIL'}")
    print(
        f"  Session B no cross-leak:       {'PASS' if b_no_banana else 'FAIL (leaked banana)'}"
    )
    print(
        f"  Session C no secret knowledge: {'PASS' if c_no_secrets else 'UNEXPECTED (knew a secret)'}"
    )
    print()

    all_pass = (
        a_has_banana and a_no_dragon and b_has_dragon and b_no_banana and c_no_secrets
    )
    if all_pass:
        print("  ALL CHECKS PASSED — Session isolation and multi-turn work correctly!")
    else:
        print("  Some checks failed — see details above.")

    print()
    print(f"  Session A ID: {session_id_a}")
    print(f"  Session B ID: {session_id_b}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MetaMate SDK — Multi-Turn Session & Isolation Demo"
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Optional MetaMate agent name override.",
    )
    parser.add_argument(
        "--total-timeout-seconds",
        type=int,
        default=300,
        help="Total operation timeout (default: 300s).",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=180,
        help="Streaming idle timeout (default: 180s).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=3.0,
        help="Poll interval (default: 3.0).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
