"""S13 — exact rewind of a Claude Code session (SDK and CLI backends).

Question: does ``fork_session(source, up_to_message_id=<turn boundary>)`` give a
new session that holds exactly the earlier turns, which the backend can then
resume — and do the remapped turn boundaries support a second rewind of the
forked session?

Drives ``NativeConversationalInferencer`` (the real ``claude`` binary through
the claude_sdk or claude_cli backend) through three remembered codewords,
rewinds to turn 3 and then to turn 2, and asks which codewords the agent
still knows.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s13_claude_sdk_fork.py --model sonnet [--kind claude_cli]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile

from _spike_common import project_dir
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
)

_WORDS = ("ALPHA-1", "BRAVO-2", "CHARLIE-3")
_LIST = "List every codeword I asked you to remember, comma-separated, nothing else."
# A codeword the agent saves to Claude Code's auto-memory (the project's
# `memory/` dir, shared by every session of the cwd) outlives a rewind.
_IN_CONVERSATION_ONLY = (
    "Keep it in this conversation only: do not save it to memory or any file."
)


async def _say(native: NativeConversationalInferencer, text: str, turn: int) -> str:
    result = await native.run_agentic_loop(text, turn_number=turn)
    print(f"[turn {turn}] {text!r} -> {result.text!r}", flush=True)
    return result.text


def _transcript_ids(session_id: str) -> set[str]:
    from claude_agent_sdk import get_session_messages

    return {m.uuid for m in get_session_messages(session_id)}


async def main(model: str, kind: str) -> int:
    work = tempfile.mkdtemp(prefix="s13_fork_")
    native = NativeConversationalInferencer(
        backend={"kind": kind, "cwd": work, "model": model},
        record_store=InMemoryRecordStore(),
        conversation_key="s13",
        native_session_dir=work,
        rewind_on_repeat_turn=True,
    )
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
        if not ok:
            failures.append(name)

    async with native:
        for turn, word in enumerate(_WORDS, start=1):
            await _say(
                native,
                f"Remember codeword {word}. {_IN_CONVERSATION_ONLY} "
                "Acknowledge in three words.",
                turn,
            )
        source = native._load_record().vendor_session_id

        await native.rewind_to(3)
        record = native._load_record()
        forked = record.vendor_session_id
        check(
            "fork adopted a new session id",
            forked and forked != source,
            f"{source} -> {forked}",
        )
        check("generation advanced", record.generation == 1, str(record.generation))
        check(
            "boundaries kept for turns 1-2",
            set(record.turn_boundaries) == {"1", "2"},
            str(record.turn_boundaries),
        )
        ids = _transcript_ids(forked)
        check(
            "boundaries point into the forked transcript",
            all(b in ids for b in record.turn_boundaries.values()),
        )
        text = await _say(native, _LIST, 3)
        check(
            "forked session forgot turn 3",
            "BRAVO-2" in text and "CHARLIE-3" not in text,
            text,
        )

        await native.rewind_to(2)
        text = await _say(native, _LIST, 2)
        check(
            "second rewind of the fork forgot turn 2",
            "ALPHA-1" in text and "BRAVO-2" not in text,
            text,
        )
    memory = project_dir(work) / "memory"
    saved = [
        f.name
        for f in (memory.glob("*") if memory.is_dir() else [])
        if f.is_file() and any(word in f.read_text() for word in _WORDS)
    ]
    check("no codeword saved to the vendor's auto-memory", not saved, str(saved))
    print("S13 PASS" if not failures else f"S13 FAIL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument(
        "--kind", default="claude_sdk", choices=("claude_sdk", "claude_cli")
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model, args.kind)))
