#!/usr/bin/env python3
"""RovoChat Inferencer — Session Save & Resume Demo.

Demonstrates that RovoChat conversations are isolated and can be
independently resumed by their session ID. Proves that:

1. Different sessions maintain separate context (secrets)
2. A saved session_id can be used to resume a conversation later
3. Resumed sessions correctly recall their own context

The Test:
    - Session A: Tell RovoChat a secret word ("banana")
    - Session B: Tell RovoChat a different secret word ("dragon")
    - Resume Session A: Ask for the secret → should say "banana"
    - Resume Session B: Ask for the secret → should say "dragon"

Prerequisites:
    Set env vars: JIRA_URL + JIRA_EMAIL + JIRA_API_TOKEN
    Or: ROVOCHAT_EMAIL + ROVOCHAT_API_TOKEN + ROVOCHAT_BASE_URL

Usage:
    python example_rovochat_session_resume.py
    python example_rovochat_session_resume.py --env-file /path/to/.env
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
# RichPythonUtils (sibling project)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)


def load_env_file(path: str) -> None:
    """Load environment variables from a file."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                val = val.strip().strip("\"'")
                if val and key not in os.environ:
                    os.environ[key] = val


def create_inferencer(**kwargs):
    """Create a fresh RovoChatInferencer instance."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
        RovoChatInferencer,
    )

    return RovoChatInferencer(**kwargs)


async def send_and_print(
    inferencer, message: str, label: str = "", session_id: str = ""
) -> str:
    """Send a message via streaming and print the response.

    Args:
        inferencer: RovoChatInferencer instance.
        message: Message to send.
        label: Optional label to print above the exchange.
        session_id: If provided, resume this conversation instead of
            creating a new one.

    Returns:
        The response text.
    """
    if label:
        print(f"  [{label}]")
    print(f"  You: {message}")
    print(f"  Rovo: ", end="", flush=True)

    response_text = ""

    if session_id:
        # Use ainfer with session_id to resume, then stream
        resp = await inferencer.ainfer(
            message, session_id=session_id, return_sdk_response=True
        )
        response_text = resp.content
        print(response_text, end="")
    else:
        async for chunk in inferencer.ainfer_streaming(message):
            print(chunk, end="", flush=True)
            response_text += chunk

    print()
    return response_text


async def run_demo(inferencer_kwargs: dict) -> None:
    """Run the session save & resume demo."""

    secret_a = "banana"
    secret_b = "dragon"

    # =========================================================
    # STEP 1: Create Session A — tell it a secret
    # =========================================================
    print()
    print("=" * 70)
    print("STEP 1: Create Session A — Secret word is 'banana'")
    print("=" * 70)
    print()

    inferencer_a = create_inferencer(**inferencer_kwargs)

    await send_and_print(
        inferencer_a,
        f"I'm going to tell you a secret word. Remember it carefully. "
        f"The secret word is: {secret_a}. "
        f"Please confirm you remember the secret word.",
        label="Session A",
    )

    session_id_a = inferencer_a.active_session_id
    print()
    print(f"  📌 Session A saved: {session_id_a}")

    # =========================================================
    # STEP 2: Create Session B — tell it a DIFFERENT secret
    # =========================================================
    print()
    print("=" * 70)
    print("STEP 2: Create Session B — Secret word is 'dragon'")
    print("=" * 70)
    print()

    inferencer_b = create_inferencer(**inferencer_kwargs)

    await send_and_print(
        inferencer_b,
        f"I'm going to tell you a secret word. Remember it carefully. "
        f"The secret word is: {secret_b}. "
        f"Please confirm you remember the secret word.",
        label="Session B",
    )

    session_id_b = inferencer_b.active_session_id
    print()
    print(f"  📌 Session B saved: {session_id_b}")

    # Verify sessions are different
    assert session_id_a != session_id_b, "Sessions should have different IDs!"
    print()
    print(f"  ✓ Sessions are different: A={session_id_a[:12]}... B={session_id_b[:12]}...")

    # =========================================================
    # STEP 3: Resume Session A — ask for the secret
    # =========================================================
    print()
    print("=" * 70)
    print("STEP 3: Resume Session A — Should recall 'banana'")
    print("=" * 70)
    print()

    # Create a FRESH inferencer (simulating a new process)
    inferencer_resume_a = create_inferencer(**inferencer_kwargs)

    # Resume session A by passing the saved session_id
    response_a = await send_and_print(
        inferencer_resume_a,
        "What is the secret word I told you earlier? Reply with just the word.",
        label=f"Resuming Session A ({session_id_a[:12]}...)",
        session_id=session_id_a,
    )

    recall_a = secret_a.lower() in response_a.lower()
    print()
    print(f"  {'✅' if recall_a else '❌'} Expected '{secret_a}': {'FOUND' if recall_a else 'NOT FOUND'} in response")

    # =========================================================
    # STEP 4: Resume Session B — ask for the secret
    # =========================================================
    print()
    print("=" * 70)
    print("STEP 4: Resume Session B — Should recall 'dragon'")
    print("=" * 70)
    print()

    # Create another FRESH inferencer
    inferencer_resume_b = create_inferencer(**inferencer_kwargs)

    # Resume session B
    response_b = await send_and_print(
        inferencer_resume_b,
        "What is the secret word I told you earlier? Reply with just the word.",
        label=f"Resuming Session B ({session_id_b[:12]}...)",
        session_id=session_id_b,
    )

    recall_b = secret_b.lower() in response_b.lower()
    print()
    print(f"  {'✅' if recall_b else '❌'} Expected '{secret_b}': {'FOUND' if recall_b else 'NOT FOUND'} in response")

    # =========================================================
    # SUMMARY
    # =========================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Session A ({session_id_a[:12]}...): secret='{secret_a}' → {'✅ RECALLED' if recall_a else '❌ FAILED'}")
    print(f"  Session B ({session_id_b[:12]}...): secret='{secret_b}' → {'✅ RECALLED' if recall_b else '❌ FAILED'}")
    print()

    if recall_a and recall_b:
        print("  🎉 SUCCESS — Sessions are isolated and resume correctly!")
    elif recall_a or recall_b:
        print("  ⚠️  PARTIAL — One session recalled correctly, the other didn't.")
        print("     This may indicate context bleeding or session resume issues.")
    else:
        print("  ❌ FAILED — Neither session recalled its secret.")
        print("     Check if session resume is working correctly.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RovoChat Session Save & Resume Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", help="RovoChat API base URL")
    parser.add_argument("--email", help="Atlassian email for Basic Auth")
    parser.add_argument("--api-token", help="Atlassian API token")
    parser.add_argument("--cloud-id", help="Atlassian Cloud ID")
    parser.add_argument("--uct-token", help="Pre-generated UCT token")
    parser.add_argument("--env-file", help="Path to .env file to load")
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    # Build inferencer kwargs from CLI args
    kwargs = {}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.email:
        kwargs["email"] = args.email
    if args.api_token:
        kwargs["api_token"] = args.api_token
    if args.cloud_id:
        kwargs["cloud_id"] = args.cloud_id
    if args.uct_token:
        kwargs["uct_token"] = args.uct_token

    print()
    print("🔐 RovoChat Session Save & Resume Demo")
    print("   Testing conversation isolation and session persistence")
    print()

    try:
        asyncio.run(run_demo(kwargs))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
