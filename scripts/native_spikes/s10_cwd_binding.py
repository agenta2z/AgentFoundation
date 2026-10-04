"""S10 — session pinning, resume across a process restart, and cwd binding.

Claude Code stores a session's transcript under the project directory of the
cwd it ran in (``projects/<cwd slug>/<id>.jsonl``). The spike pins a session
id in cwd A and tells it a codeword, then:

  1. resumes it from cwd A in a new process (the control: pin + restart),
  2. resumes it from a different cwd B: does the vendor find the session, and
     if not, does it fail with the "No conversation found with session ID"
     marker the native backends classify as a lost session (``recap``), rather
     than silently starting an empty session under the same id?
  3. resumes it from cwd A again (the session is intact).

``claude -p`` (claude_cli) and the Agent SDK (claude_sdk).

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s10_cwd_binding.py [--kinds cli,sdk] [--model haiku]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

from _spike_common import (
    Checks,
    claude_argv,
    project_dir,
    run_claude,
    run_sdk,
    sdk_options,
    section,
    short,
)

MISSING = (
    "No conversation found with session ID"  # session/claude_*.py _MISSING_SESSION
)
ASK = "What codeword did I ask you to remember? Reply with only the codeword, or NONE."


def _entries(transcript: Path) -> list[dict]:
    if not transcript.exists():
        return []
    with open(transcript) as f:
        return [json.loads(line) for line in f if line.strip()]


def _request_tokens(entries: list[dict]) -> int:
    """Input + cache write + cache read of the last API request in ``entries``."""
    for entry in reversed(entries):
        usage = (entry.get("message") or {}).get("usage")
        if entry.get("type") == "assistant" and usage:
            return sum(
                usage.get(key) or 0
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
    return 0


async def _turn(
    kind: str, model: str, cwd: str, prompt: str, **session: str
) -> tuple[str, str, bool, str]:
    """(reply, stderr, failed, the cwd the vendor reports in its init event)"""
    if kind == "cli":
        run = await run_claude(claude_argv(prompt, model=model, **session), cwd=cwd)
        failed = not run.result or bool(run.result.get("is_error"))
        return run.text, run.stderr, failed, str(run.init.get("cwd", ""))
    run = await run_sdk(sdk_options(cwd=cwd, model=model, **session), prompt)
    result = run.result
    failed = bool(run.error) or result is None or bool(result.is_error)
    init = run.system("init")
    init_cwd = str(init[0].data.get("cwd", "")) if init else ""
    return run.text, "\n".join(run.stderr) + run.error, failed, init_cwd


async def spike(kind: str, model: str, c: Checks) -> None:
    k = f"S10[{kind}]"
    section(k)
    cwd_a = tempfile.mkdtemp(prefix=f"s10a_{kind}_")
    cwd_b = tempfile.mkdtemp(prefix=f"s10b_{kind}_")
    sid = str(uuid.uuid4())
    word = f"ORCHID-{uuid.uuid4().hex[:5].upper()}"
    reply, _, failed, _ = await _turn(
        kind,
        model,
        cwd_a,
        f"Remember the codeword {word}. Reply with exactly: NOTED",
        session_id=sid,
    )
    c.check(f"{k} pinned session ran in cwd A", not failed and "NOTED" in reply, reply)
    transcript = project_dir(cwd_a) / f"{sid}.jsonl"
    c.check(
        f"{k} transcript lives under cwd A's project dir",
        transcript.exists(),
        str(transcript.parent.name),
    )
    size_before = transcript.stat().st_size if transcript.exists() else 0

    reply, _, failed, _ = await _turn(kind, model, cwd_a, ASK, resume=sid)
    c.check(
        f"{k} restart + resume from cwd A remembers",
        not failed and word in reply,
        reply,
    )

    size_a = transcript.stat().st_size if transcript.exists() else 0
    before = _entries(transcript)
    reply, stderr, failed, init_cwd = await _turn(kind, model, cwd_b, ASK, resume=sid)
    grew = (transcript.stat().st_size if transcript.exists() else 0) > size_a
    in_b = (project_dir(cwd_b) / f"{sid}.jsonl").exists()
    added = _entries(transcript)[len(before) :]
    # Transport evidence that the cwd-B request carried the conversation: its
    # first entry chains onto the session's last entry, and the request is
    # larger than the previous (turn 2) request. Whether the model then recalls
    # the codeword is its judgment (haiku may read the transcript's
    # "environment" attachment — working directory changed — as a new session).
    last_uuid = next((e["uuid"] for e in reversed(before) if e.get("uuid")), None)
    first_new = next((e for e in added if e.get("uuid")), {})
    chained = bool(last_uuid) and first_new.get("parentUuid") == last_uuid
    larger = _request_tokens(added) > _request_tokens(before)
    if failed:
        outcome = "refused"
    elif word in reply or (grew and not in_b and chained and larger):
        outcome = "continued"
    else:
        outcome = "empty session"
    c.info(
        f"{k} resume from cwd B",
        f"{outcome}; reply={reply[:80]!r} (codeword recalled={word in reply}); "
        f"runs in cwd B={init_cwd == cwd_b}; appends to the transcript under "
        f"A={grew}; new transcript under B={in_b}; chained onto the last entry="
        f"{chained}; request tokens {_request_tokens(before)} -> "
        f"{_request_tokens(added)}",
    )
    c.check(
        f"{k} cwd B never yields a silent empty session",
        outcome != "empty session",
        outcome,
    )
    if outcome == "refused":
        c.check(
            f"{k} the refusal carries the backends' session-missing marker",
            MISSING in stderr,
            stderr[-200:],
        )

    reply, _, failed, _ = await _turn(kind, model, cwd_a, ASK, resume=sid)
    c.check(
        f"{k} session {short(sid)} still resumes from cwd A",
        not failed and word in reply and transcript.stat().st_size > size_before,
        reply,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--kinds", default="cli,sdk")
    args = parser.parse_args()
    c = Checks("S10")
    for kind in args.kinds.split(","):
        await spike(kind.strip(), args.model, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
