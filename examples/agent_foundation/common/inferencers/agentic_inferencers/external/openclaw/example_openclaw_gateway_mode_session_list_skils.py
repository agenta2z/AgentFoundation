#!/usr/bin/env python3
"""OpenClaw Inferencer — List Available Skills Demo (PodGateway mode).

Asks the OpenClaw agent to list all skills available in its current session
and prints them in a structured, readable format.

Demonstrates:
  1. Connecting to the OpenClaw gateway (PodGateway mode)
  2. Querying the agent for its available skills
  3. Parsing and displaying the response clearly
  4. Verifying key skills (TWG, agentic-search) are present

The OpenClaw agent discovers skills by reading SKILL.md files from:
  - Bundled skills: /usr/lib/node_modules/openclaw/skills/<name>/SKILL.md
  - Personal skills: /sandbox/.agents/skills/<name>/SKILL.md (TWG, agentic-search)

Skills are injected into the agent's system prompt via a <available_skills> XML
block — the agent reads each SKILL.md on demand when the task matches.

Prerequisites:
  - Docker container ``openshell-cluster-openshell`` running (``./run.sh start``)
  - OpenClaw gateway pod healthy inside the container

Usage:
    python example_openclaw_gateway_mode_session_list_skils.py
    python example_openclaw_gateway_mode_session_list_skils.py --session-id my-session
    python example_openclaw_gateway_mode_session_list_skils.py --thinking low
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# parents[8] = CoreProjects  (script is 9 dirs deep from CoreProjects)
_core_projects = Path(__file__).parents[8]
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = _core_projects / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ─────────────────────────────────────────────────────────────────────────────

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    OpenClawError,
    OpenClawInferencer,
    OpenClawMode,
    OpenClawNotFoundError,
)

# Skills we expect to be present — used for verification
EXPECTED_SKILLS = {
    "twg": "Atlassian TeamWork Graph (Jira, Confluence, Loom, Bitbucket, Atlas)",
    "agentic-search": "Deep enterprise knowledge search across Atlassian + Google Drive + Slack",
    "coding-agent": "Delegate coding tasks to Codex / Claude Code / Pi agents",
    "github": "GitHub CLI operations",
    "skill-creator": "Create, edit, improve, or audit AgentSkills and SKILL.md files",
}


async def list_skills(session_id: str, thinking: str | None, timeout: int) -> None:
    """Connect to the OpenClaw gateway and ask the agent to list its skills."""

    print()
    print("=" * 70)
    print("OpenClaw — Available Skills Listing")
    print("=" * 70)
    print(f"  Mode:       PodGateway (ws://127.0.0.1:18789)")
    print(f"  Session ID: {session_id!r}")
    print(f"  Thinking:   {thinking or 'gateway default'}")
    print()

    inf = OpenClawInferencer(
        mode=OpenClawMode.PodGateway,
        session_id=session_id,
        thinking=thinking,
        timeout_seconds=timeout,
        enable_turn_separation=False,       # flat stream for easy parsing
        always_initialize_new_session=True, # auto warm-up on new sessions
        auto_resume=True,
    )

    # ── Query the agent ───────────────────────────────────────────────────────
    prompt = (
        "List ALL your available skills exactly as follows:\n"
        "For each skill, print one line: '- <skill-name>: <one-sentence description>'\n"
        "Do NOT use any tools. Do NOT search anything. "
        "Just list every skill you see in your available_skills list. "
        "Include ALL of them, including TWG and agentic-search."
    )

    print("Asking agent to list all available skills...")
    print("-" * 70)

    t0 = time.time()
    chunks: list[str] = []
    first_chunk_time: float | None = None

    async for chunk in inf.ainfer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    elapsed = time.time() - t0
    response = "".join(chunks)
    print()
    print("-" * 70)
    print(f"  ⏱  ttft={first_chunk_time - t0:.2f}s  total={elapsed:.1f}s")
    print()

    # ── Parse and verify ──────────────────────────────────────────────────────
    response_lower = response.lower()

    # Count skill lines (lines starting with "- ")
    skill_lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip().startswith("-") and ":" in line
    ]

    print("=" * 70)
    print("Verification")
    print("=" * 70)
    print(f"  Skills listed by agent: ~{len(skill_lines)}")
    print()

    all_found = True
    for skill_name, description in EXPECTED_SKILLS.items():
        found = skill_name.lower() in response_lower
        status = "✅" if found else "❌"
        print(f"  {status} {skill_name:<20} — {description}")
        if not found:
            all_found = False

    print()
    if all_found:
        print("  🎉 All expected skills are present!")
    else:
        print("  ⚠️  Some expected skills are missing.")
        print("     This may mean the skill's SKILL.md is not in the agent's path,")
        print("     or the agent chose not to list it. Try re-running.")

    print()
    print(f"  Session ID: {inf.active_session_id or session_id!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenClaw Gateway Mode — List Available Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-id",
        default="skills-list-demo",
        help="Session ID to use (default: 'skills-list-demo'). "
             "Reuse the same ID to skip the warm-up turn on repeat runs.",
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

    try:
        asyncio.run(list_skills(
            session_id=args.session_id,
            thinking=args.thinking,
            timeout=args.timeout,
        ))
    except OpenClawNotFoundError as e:
        print(f"\n❌ OpenClaw not reachable: {e}")
        print("   Ensure './run.sh start' is running and gateway is at ws://127.0.0.1:18789")
        sys.exit(1)
    except OpenClawError as e:
        print(f"\n❌ OpenClaw error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")


if __name__ == "__main__":
    main()
