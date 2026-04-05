#!/usr/bin/env python3
"""Real integration test for MetamateCliInferencer with deep research mode.

Tests deep research (SPACES_DEEP_RESEARCH_AGENT and METAMATE_MDR) with a
realistic long-form research prompt to verify the inferencer produces
meaningful, substantive research output.

Usage (via buck2):
    # Run deep research with default prompt (hardcoded constants investigation)
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode deep-research

    # Run with METAMATE_MDR agent instead of SPACES_DEEP_RESEARCH_AGENT
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode deep-research --agent metamate_mdr

    # Run streaming deep research
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode streaming

    # Compare agents: SPACES_DEEP_RESEARCH_AGENT vs METAMATE_MDR
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode compare-agents

    # Run with custom prompt from file
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --prompt-file /path/to/prompt.txt

    # Run with inline custom prompt
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --query "Research torch.compile graph breaks in recommendation models"
"""

import argparse
import asyncio
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Agent name constants
SPACES_DEEP_RESEARCH_AGENT = "SPACES_DEEP_RESEARCH_AGENT"
METAMATE_MDR = "METAMATE_MDR"

# Default prompt loaded from _inputs/ at runtime
_INPUTS_DIR = os.path.join(os.path.dirname(__file__), "_inputs")
_DEFAULT_PROMPT_FILE = os.path.join(
    _INPUTS_DIR, "deep_research_hardcoded_constants.txt"
)

DEFAULT_QUERY = (
    "What are best practices for managing hard-coded hyperparameters "
    "in large-scale deep learning recommendation models? Compare "
    "config-driven approaches (dataclass configs, YAML, Hydra) vs "
    "hard-coded constants. Include examples from DLRM, DCN, PLE, "
    "and MHTA architectures."
)


def _load_prompt(prompt_file: str | None, query: str | None) -> str:
    """Load prompt from file or use query string."""
    if query:
        return query
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file) as f:
            return f.read().strip()
    # Try default input file
    if os.path.isfile(_DEFAULT_PROMPT_FILE):
        with open(_DEFAULT_PROMPT_FILE) as f:
            return f.read().strip()
    return DEFAULT_QUERY


def test_deep_research_sync(
    query: str, agent_name: str = SPACES_DEEP_RESEARCH_AGENT
):
    """Test synchronous deep research call."""
    print("\n" + "=" * 70)
    print(f"TEST: Deep Research Sync (agent={agent_name})")
    print("=" * 70)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"FAIL: Cannot import MetamateCliInferencer: {e}")
        return False

    try:
        inferencer = MetamateCliInferencer(
            deep_research=True,
            agent_name=agent_name,
            timeout_seconds=900,  # 15 min for deep research
            idle_timeout_seconds=600,  # 10 min idle
        )
        print(f"  Created inferencer (deep_research=True, agent={agent_name})")
        print(f"  Query length: {len(query)} chars")
        print(f"  Query preview: {query[:200]}...")
        print("  (Deep research may take 5-10 minutes...)")

        start_time = time.time()
        result = inferencer.infer(query)
        elapsed = time.time() - start_time

        output = str(result)
        print(f"\n  Response received in {elapsed:.1f}s")
        print(f"  Output length: {len(output)} chars")
        print(f"  Output lines: {output.count(chr(10))}")
        print(f"  Has markdown headings: {'#' in output}")
        print("-" * 70)
        # Show more output for deep research
        preview_len = 5000
        print(output[:preview_len] + "..." if len(output) > preview_len else output)
        print("-" * 70)

        # Deep research should produce substantial output
        if len(output) >= 200:
            print(f"\n  PASSED (deep research, {len(output)} chars, {elapsed:.1f}s)")
            return True
        else:
            print(f"\n  FAILED: Output too short ({len(output)} chars)")
            return False

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_deep_research_streaming(
    query: str, agent_name: str = SPACES_DEEP_RESEARCH_AGENT
):
    """Test streaming deep research call."""
    print("\n" + "=" * 70)
    print(f"TEST: Deep Research Streaming (agent={agent_name})")
    print("=" * 70)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"FAIL: Cannot import MetamateCliInferencer: {e}")
        return False

    try:
        cache_folder = "/tmp/test_metamate_deep_research_streaming"
        os.makedirs(cache_folder, exist_ok=True)

        inferencer = MetamateCliInferencer(
            deep_research=True,
            agent_name=agent_name,
            timeout_seconds=900,
            idle_timeout_seconds=600,
            cache_folder=cache_folder,
        )
        print(f"  Created inferencer (deep_research=True, agent={agent_name})")
        print(f"  Cache folder: {cache_folder}")
        print(f"  Query length: {len(query)} chars")
        print("  (Deep research may take 5-10 minutes...)")
        print("-" * 70)
        print("STREAMING OUTPUT:")
        print("-" * 70)

        start_time = time.time()
        chunks = []
        chunk_count = 0

        async for chunk in inferencer.ainfer_streaming(query):
            chunks.append(chunk)
            chunk_count += 1
            print(chunk, end="", flush=True)

        elapsed = time.time() - start_time
        full_output = "".join(chunks)

        print("\n" + "-" * 70)
        print(f"\n  Streaming completed in {elapsed:.1f}s")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total chars: {len(full_output)}")

        # List cache files
        if os.path.isdir(cache_folder):
            cache_files = os.listdir(cache_folder)
            print(f"  Cache files: {cache_files}")

        if len(full_output) >= 200:
            print(f"\n  PASSED (streaming deep research, {len(full_output)} chars)")
            return True
        else:
            print(f"\n  FAILED: Output too short ({len(full_output)} chars)")
            return False

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_compare_agents(query: str):
    """Compare SPACES_DEEP_RESEARCH_AGENT vs METAMATE_MDR with the same query."""
    print("\n" + "#" * 70)
    print("COMPARISON: SPACES_DEEP_RESEARCH_AGENT vs METAMATE_MDR")
    print("#" * 70)
    print(f"Query: {query[:200]}...")
    print("#" * 70)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"FAIL: Cannot import: {e}")
        return []

    results = []

    for agent_name in [SPACES_DEEP_RESEARCH_AGENT, METAMATE_MDR]:
        print(f"\n{'=' * 70}")
        print(f"Agent: {agent_name}")
        print("=" * 70)

        inferencer = MetamateCliInferencer(
            deep_research=True,
            agent_name=agent_name,
            timeout_seconds=900,
            idle_timeout_seconds=600,
        )
        print(f"  Command: {inferencer.construct_command(query)[:200]}...")

        start_time = time.time()
        try:
            result = inferencer.infer(query)
            output = str(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            output = ""
        elapsed = time.time() - start_time

        print(f"\n  Response ({len(output)} chars, {elapsed:.1f}s):")
        print("-" * 70)
        preview_len = 3000
        print(output[:preview_len] + "..." if len(output) > preview_len else output)
        print("-" * 70)

        results.append((f"Deep Research ({agent_name})", bool(output) and len(output) >= 200))

    # Summary
    print(f"\n{'#' * 70}")
    print("COMPARISON SUMMARY")
    print("#" * 70)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    print("#" * 70)

    return results


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Deep research integration tests for MetamateCliInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deep research with default long prompt
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research

  # Compare SPACES_DEEP_RESEARCH_AGENT vs METAMATE_MDR
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode compare-agents

  # Streaming deep research
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --mode streaming

  # Custom prompt from file
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_deep_research -- --prompt-file /path/to/prompt.txt
        """,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Inline query (overrides --prompt-file and default input file)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a file containing the research prompt",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["deep-research", "streaming", "compare-agents"],
        default="deep-research",
        help="Test mode (default: deep-research)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["spaces", "metamate_mdr"],
        default="spaces",
        help="Agent to use: spaces=SPACES_DEEP_RESEARCH_AGENT, metamate_mdr=METAMATE_MDR (default: spaces)",
    )

    args = parser.parse_args()

    query = _load_prompt(args.prompt_file, args.query)
    agent_name = METAMATE_MDR if args.agent == "metamate_mdr" else SPACES_DEEP_RESEARCH_AGENT

    print("=" * 70)
    print("METAMATE CLI - DEEP RESEARCH INTEGRATION TESTS")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Agent: {agent_name}")
    print(f"Query length: {len(query)} chars")
    print(f"Query preview: {query[:200]}...")

    results = []

    if args.mode == "deep-research":
        results.append(
            (f"Deep Research Sync ({agent_name})", test_deep_research_sync(query, agent_name))
        )

    elif args.mode == "streaming":
        passed = asyncio.run(test_deep_research_streaming(query, agent_name))
        results.append((f"Deep Research Streaming ({agent_name})", passed))

    elif args.mode == "compare-agents":
        results.extend(test_compare_agents(query))

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
