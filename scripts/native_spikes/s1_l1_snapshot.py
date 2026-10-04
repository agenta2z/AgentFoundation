"""S1 — is the L1 append frozen across resume and ``fork_session``, and is it
re-recorded after compaction?

Claude Code records the first request's system prompt (the preset plus the
``--append-system-prompt-file`` text) as a ``prompt_snapshot`` transcript
attachment and resends that record on later requests and resumes. The spike
pins a session with codeword A in the L1 file, rewrites the file to codeword B,
and asks for the codeword after:

  1. a resume of the session (new process),
  2. a resume of ``fork_session(session, up_to_message_id=<turn 1 reply>)``
     (the native rewind path),
  3. ``/compact`` on a second fork of turn 1, then a resume.

Turn 1 never mentions the codeword, so a reply can only come from the system
prompt. Evidence: the recorded ``prompt_snapshot`` texts (A vs B) in each
transcript, and the reply. Runs through ``claude -p`` (claude_cli shape) and
the Agent SDK (claude_sdk shape: preset + append file).

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s1_l1_snapshot.py [--kinds cli,sdk] [--model haiku]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from _spike_common import (
    Checks,
    claude_argv,
    run_claude,
    run_sdk,
    sdk_options,
    section,
    session_entries,
    short,
    snapshot_texts,
    snapshots_after_compaction,
)

ASK = "What is the session codeword? Reply with only the codeword."


def _l1(word: str) -> str:
    return (
        f"Session codeword: {word}. If the user asks for the session codeword, "
        "reply with exactly that codeword and nothing else.\n"
    )


class Turns:
    """One turn = one vendor process (CLI) or one fresh SDK client."""

    def __init__(self, kind: str, model: str, work: str, l1_path: str) -> None:
        self.kind, self.model, self.work, self.l1_path = kind, model, work, l1_path

    async def say(self, prompt: str, *, session_id: str = "", resume: str = "") -> Any:
        if self.kind == "cli":
            run = await run_claude(
                claude_argv(
                    prompt,
                    model=self.model,
                    session_id=session_id,
                    resume=resume,
                    append_file=self.l1_path,
                ),
                cwd=self.work,
            )
            compacted = bool(run.system("compact_boundary"))
            return run.text, compacted, run.rc != 0 and not run.text
        run = await run_sdk(
            sdk_options(
                cwd=self.work,
                model=self.model,
                session_id=session_id,
                resume=resume,
                append_file=self.l1_path,
            ),
            prompt,
        )
        compacted = bool(run.system("compact_boundary"))
        return run.text, compacted, bool(run.error)


def _last_assistant_uuid(entries: list[dict[str, Any]]) -> str:
    for entry in reversed(entries):
        if entry.get("type") == "assistant" and not entry.get("isSidechain"):
            return str(entry.get("uuid") or "")
    return ""


def _has(texts: list[str], word: str) -> bool:
    return any(word in t for t in texts)


async def spike(kind: str, model: str, c: Checks) -> None:
    from claude_agent_sdk import fork_session

    section(f"S1 [{kind}] L1 snapshot across resume / fork / compaction")
    work = tempfile.mkdtemp(prefix=f"s1snap_{kind}_")
    l1_path = str(Path(work) / "l1.txt")
    word_a = f"ALPHA-{uuid.uuid4().hex[:5].upper()}"
    word_b = f"BRAVO-{uuid.uuid4().hex[:5].upper()}"
    Path(l1_path).write_text(_l1(word_a), encoding="utf-8")
    turns = Turns(kind, model, work, l1_path)
    sid = str(uuid.uuid4())

    reply, _, failed = await turns.say(
        "Reply with exactly the word READY.", session_id=sid
    )
    entries = session_entries(sid)
    c.check(f"S1[{kind}] turn 1 ran", not failed and "READY" in reply.upper(), reply)
    c.check(
        f"S1[{kind}] turn 1 recorded a prompt_snapshot carrying L1=A",
        _has(snapshot_texts(entries), word_a),
        f"{len(snapshot_texts(entries))} snapshot(s) in {short(sid)}",
    )
    boundary = _last_assistant_uuid(entries)
    c.check(f"S1[{kind}] turn 1 boundary found", bool(boundary))

    Path(l1_path).write_text(_l1(word_b), encoding="utf-8")
    reply, _, _ = await turns.say(ASK, resume=sid)
    entries = session_entries(sid)
    c.check(
        f"S1[{kind}] resume with L1 file=B answers A (frozen across resume)",
        word_a in reply and word_b not in reply,
        reply,
    )
    c.check(
        f"S1[{kind}] resume recorded no snapshot with B",
        _has(snapshot_texts(entries), word_a)
        and not _has(snapshot_texts(entries), word_b),
        f"{len(snapshot_texts(entries))} snapshot(s)",
    )

    forked = (await asyncio.to_thread(fork_session, sid, None, boundary)).session_id
    fork_entries = session_entries(forked)
    c.check(
        f"S1[{kind}] fork_session copied the A snapshot",
        forked != sid and _has(snapshot_texts(fork_entries), word_a),
        f"{short(sid)} -> {short(forked)}",
    )
    reply, _, _ = await turns.say(ASK, resume=forked)
    fork_entries = session_entries(forked)
    c.check(
        f"S1[{kind}] resumed fork with L1 file=B answers A (frozen across fork)",
        word_a in reply and word_b not in reply,
        reply,
    )
    c.check(
        f"S1[{kind}] resumed fork recorded no snapshot with B",
        not _has(snapshot_texts(fork_entries), word_b),
    )

    second = (await asyncio.to_thread(fork_session, sid, None, boundary)).session_id
    _, compacted, failed = await turns.say("/compact", resume=second)
    c.check(
        f"S1[{kind}] /compact on a fork emitted compact_boundary",
        compacted and not failed,
    )
    reply, _, _ = await turns.say(ASK, resume=second)
    after = snapshots_after_compaction(session_entries(second))
    c.check(
        f"S1[{kind}] after /compact a new snapshot records L1=B (re-recorded)",
        _has(after, word_b) and not _has(after, word_a),
        f"{len(after)} snapshot(s) after the boundary",
    )
    c.check(
        f"S1[{kind}] after /compact the model answers B",
        word_b in reply and word_a not in reply,
        reply,
    )


def _quiet(_line: str) -> None:
    pass


async def _drain(client: Any, prompt: str) -> tuple[str, bool]:
    from claude_agent_sdk import ResultMessage, SystemMessage

    await client.query(prompt)
    text, compacted = "", False
    async for message in client.receive_response():
        if isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
            compacted = True
        if isinstance(message, ResultMessage):
            text = message.result or ""
    return text, compacted


async def persistent_sdk(model: str, c: Checks) -> None:
    """The ``claude_sdk`` backend keeps one process per session: does an L1
    file rewritten after that process started reach the snapshot recorded at
    the next ``/compact`` (the D8 re-arm), or does the process keep the text it
    read at launch?"""
    from claude_agent_sdk import ClaudeSDKClient

    k = "S1[sdk, one process]"
    section(f"{k} L1 file rewritten mid-process, then /compact")
    work = tempfile.mkdtemp(prefix="s1snap_persist_")
    l1_path = str(Path(work) / "l1.txt")
    word_a = f"ALPHA-{uuid.uuid4().hex[:5].upper()}"
    word_b = f"BRAVO-{uuid.uuid4().hex[:5].upper()}"
    Path(l1_path).write_text(_l1(word_a), encoding="utf-8")
    sid = str(uuid.uuid4())
    options = sdk_options(
        cwd=work, model=model, session_id=sid, append_file=l1_path, stderr=_quiet
    )
    async with ClaudeSDKClient(options=options) as client:
        reply, _ = await _drain(client, "Reply with exactly the word READY.")
        c.check(f"{k} turn 1 ran", "READY" in reply.upper(), reply)
        Path(l1_path).write_text(_l1(word_b), encoding="utf-8")
        _, compacted = await _drain(client, "/compact")
        c.check(f"{k} /compact emitted compact_boundary", compacted)
        reply, _ = await _drain(client, ASK)
    after = snapshots_after_compaction(session_entries(sid))
    c.check(
        f"{k} the post-compaction snapshot keeps the launch-time L1 (A): the file is read once per process",
        _has(after, word_a) and not _has(after, word_b),
        f"{len(after)} snapshot(s) after the boundary; A={_has(after, word_a)} B={_has(after, word_b)}",
    )
    c.check(f"{k} the model answers A", word_a in reply and word_b not in reply, reply)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--kinds", default="cli,sdk")
    args = parser.parse_args()
    c = Checks("S1-snapshot")
    for kind in args.kinds.split(","):
        await spike(kind.strip(), args.model, c)
    if "sdk" in args.kinds.split(","):
        await persistent_sdk(args.model, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
