#!/usr/bin/env python3
"""RovoChat Inferencer — Streaming vs Non-Streaming Demo.

Demonstrates both streaming and non-streaming (sync) modes of the
RovoChat inferencer with real queries against the RovoChat API.

Prerequisites:
    Set one of these env var combinations:
      - JIRA_URL + JIRA_EMAIL + JIRA_API_TOKEN  (simplest)
      - ROVOCHAT_BASE_URL + ROVOCHAT_EMAIL + ROVOCHAT_API_TOKEN
      - ROVOCHAT_UCT_TOKEN + ROVOCHAT_BASE_URL + ROVOCHAT_CLOUD_ID

    Or pass credentials directly via command-line arguments.

Usage:
    # Using env vars (zero-config):
    python example_rovochat_streaming_and_sync.py

    # With explicit credentials:
    python example_rovochat_streaming_and_sync.py \\
        --base-url https://hello.atlassian.net \\
        --email you@atlassian.com \\
        --api-token YOUR_TOKEN

    # Custom query:
    python example_rovochat_streaming_and_sync.py \\
        --query "How do I create a Jira issue via the REST API?"

    # Streaming only:
    python example_rovochat_streaming_and_sync.py --mode streaming

    # Non-streaming only:
    python example_rovochat_streaming_and_sync.py --mode sync
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
    """Load environment variables from a file (simple .env parser)."""
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


def create_inferencer(args: argparse.Namespace):
    """Create a RovoChatInferencer from CLI args or env vars."""
    # Lazy import to allow env var setup first
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
        RovoChatInferencer,
    )

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

    inferencer = RovoChatInferencer(**kwargs)
    return inferencer


def demo_sync(inferencer, query: str) -> None:
    """Demonstrate non-streaming (synchronous) inference.

    Sends the query and waits for the complete response before printing.
    Simple but no incremental output — you see nothing until it's done.
    """
    print("=" * 70)
    print("MODE: Non-Streaming (Synchronous)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print("Response:")
    print("-" * 70)
    print(result)
    print("-" * 70)
    print(f"\n⏱  {elapsed:.1f}s | {len(result)} chars")
    print()


async def demo_streaming(inferencer, query: str) -> None:
    """Demonstrate async streaming inference.

    Sends the query and prints text chunks as they arrive in real-time.
    Much better UX for long responses — you see output incrementally.
    """
    print("=" * 70)
    print("MODE: Streaming (Async)")
    print("=" * 70)
    print(f"Query: {query}")
    print()
    print("Response (streaming):")
    print("-" * 70)

    start = time.time()
    total_chars = 0
    chunk_count = 0

    async for chunk in inferencer.ainfer_streaming(query):
        print(chunk, end="", flush=True)
        total_chars += len(chunk)
        chunk_count += 1

    elapsed = time.time() - start

    print()
    print("-" * 70)
    print(f"\n⏱  {elapsed:.1f}s | {total_chars} chars | {chunk_count} chunks")
    print()


async def demo_streaming_with_session(inferencer, queries: list) -> None:
    """Demonstrate multi-turn streaming with conversation persistence.

    Sends multiple queries in the same conversation, showing how
    context is maintained across turns.
    """
    print("=" * 70)
    print("MODE: Multi-Turn Streaming")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Query: {query}")
        print()

        start = time.time()
        total_chars = 0

        async for chunk in inferencer.ainfer_streaming(query):
            print(chunk, end="", flush=True)
            total_chars += len(chunk)

        elapsed = time.time() - start
        print(f"\n  [⏱ {elapsed:.1f}s | {total_chars} chars]")

    print()
    print("=" * 70)
    print(f"✓ Completed {len(queries)} turns in same conversation")
    print(f"  Session ID: {inferencer.active_session_id}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RovoChat Inferencer — Streaming vs Non-Streaming Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "sync", "multi-turn", "all"],
        default="all",
        help="Demo mode (default: all)",
    )
    parser.add_argument(
        "--query",
        default="How to check recent Confluence pages of a person through CLI or API?",
        help="Query to send to RovoChat",
    )
    parser.add_argument("--base-url", help="RovoChat API base URL")
    parser.add_argument("--email", help="Atlassian email for Basic Auth")
    parser.add_argument("--api-token", help="Atlassian API token")
    parser.add_argument("--cloud-id", help="Atlassian Cloud ID")
    parser.add_argument("--uct-token", help="Pre-generated UCT token")
    parser.add_argument(
        "--env-file",
        help="Path to .env file to load",
    )
    args = parser.parse_args()

    # Load env file if specified
    if args.env_file:
        load_env_file(args.env_file)

    # Create inferencer
    try:
        inferencer = create_inferencer(args)
    except Exception as e:
        print(f"❌ Failed to create inferencer: {e}")
        print()
        print("Set credentials via env vars or CLI arguments. See --help.")
        sys.exit(1)

    print()
    print("🤖 RovoChat Inferencer Demo")
    print(f"   Base URL:  {inferencer.base_url}")
    print(f"   Cloud ID:  {inferencer.cloud_id or '(auto from gateway)'}")
    print(f"   Auth mode: {inferencer._create_auth().auth_mode}")
    print()

    # Run demos
    try:
        if args.mode in ("sync", "all"):
            demo_sync(inferencer, args.query)

        if args.mode in ("streaming", "all"):
            # Reset session so streaming gets a fresh conversation
            inferencer.reset_session()
            asyncio.run(demo_streaming(inferencer, args.query))

        if args.mode in ("multi-turn", "all"):
            # Reset session for multi-turn demo
            inferencer.reset_session()
            asyncio.run(
                demo_streaming_with_session(
                    inferencer,
                    [
                        args.query,
                        "Can you summarize the above in 3 bullet points?",
                    ],
                )
            )

        print("🎉 Demo complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
