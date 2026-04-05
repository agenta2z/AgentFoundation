#!/usr/bin/env python3
"""Quick test: run MetamateCliInferencer deep research with a prompt from file.

Usage:
    buck2 run @fbcode//mode/dbgo fbcode//rankevolve/test/agentic_foundation:test_deep_research_from_file
    buck2 run @fbcode//mode/dbgo fbcode//rankevolve/test/agentic_foundation:test_deep_research_from_file -- --prompt-file /path/to/prompt.txt
    buck2 run @fbcode//mode/dbgo fbcode//rankevolve/test/agentic_foundation:test_deep_research_from_file -- --agent METAMATE_MDR
"""

import argparse
import importlib.resources
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt file. Defaults to bundled _inputs/deep_research_hardcoded_constants.txt",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="SPACES_DEEP_RESEARCH_AGENT",
        help="Agent name (default: SPACES_DEEP_RESEARCH_AGENT)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Total timeout in seconds (default: 900)",
    )
    args = parser.parse_args()

    # Load prompt
    if args.prompt_file:
        prompt_path = args.prompt_file
    else:
        # Try relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(
            script_dir,
            "_inputs",
            "deep_research_hardcoded_constants.txt",
        )
        if not os.path.isfile(prompt_path):
            # Try importlib.resources for buck-bundled resources
            try:
                ref = importlib.resources.files(
                    "rankevolve.test.agentic_foundation.common.inferencers.external.metamate._inputs"
                ).joinpath("deep_research_hardcoded_constants.txt")
                prompt_path = str(ref)
            except Exception:
                pass

    if not os.path.isfile(prompt_path):
        print(f"ERROR: Cannot find prompt file at {prompt_path}")
        return 1

    with open(prompt_path) as f:
        query = f.read().strip()

    print("=" * 70)
    print("DEEP RESEARCH TEST (from file)")
    print("=" * 70)
    print(f"Prompt file: {prompt_path}")
    print(f"Agent: {args.agent}")
    print(f"Timeout: {args.timeout}s")
    print(f"Query length: {len(query)} chars")
    print(f"Query first 300 chars:\n{query[:300]}...")
    print("=" * 70)

    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateCliInferencer,
    )

    inferencer = MetamateCliInferencer(
        deep_research=True,
        agent_name=args.agent,
        timeout_seconds=args.timeout,
        idle_timeout_seconds=600,
    )

    print(f"\nCommand: {inferencer.construct_command(query)[:300]}...")
    print("\nSending deep research query (may take 5-10 minutes)...")
    print("-" * 70)

    start_time = time.time()
    result = inferencer.infer(query)
    elapsed = time.time() - start_time

    output = str(result)
    print(f"\nResponse received in {elapsed:.1f}s")
    print(f"Output length: {len(output)} chars")
    print(f"Output lines: {output.count(chr(10))}")
    print(f"Has markdown headings: {'#' in output}")
    print("-" * 70)
    print("FULL OUTPUT:")
    print("-" * 70)
    print(output)
    print("-" * 70)

    if len(output) >= 200:
        print(f"\nPASSED: Got {len(output)} chars in {elapsed:.1f}s")
        return 0
    else:
        print(f"\nFAILED: Output too short ({len(output)} chars)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
