"""S16c — a native conversation on the Metamate backend (real Metamate).

Runs ``NativeConversationalInferencer`` with ``kind: metamate`` (tool-less):
chat with a remembered codeword, an SOP entered by slash command, and a turn
that must use the active SOP state from the per-turn envelope.

    buck2 run @//mode/dbgo //_tony_dev/CoreProjects/AgentFoundation/scripts/native_spikes:s16c_metamate_native
"""

from __future__ import annotations

import asyncio
import tempfile

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
)


async def amain() -> int:
    work = tempfile.mkdtemp(prefix="s16c_")
    native = NativeConversationalInferencer(
        backend={"kind": "metamate", "cwd": work, "l2_envelope_allowed": True},
        record_store=InMemoryRecordStore(),
        conversation_key="s16c",
        native_session_dir=work,
        allowed_sops=["model_optimization"],
    )
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail[:300]!r}", flush=True)
        if not ok:
            failures.append(name)

    async with native:
        r = await native.run_agentic_loop(
            "Remember the codeword PELICAN-42. Which SOPs can you run here?",
            turn_number=1,
        )
        check(
            "catalog from the session instructions",
            "Model Optimization" in r.text,
            r.text,
        )
        sid = native._load_record().vendor_session_id
        check("conversation recorded", bool(sid), sid)
        r = await native.run_agentic_loop("/sop model_optimization", turn_number=2)
        check("SOP entered by slash command", native.sop_state is not None, r.text)
        r = await native.run_agentic_loop(
            "Which SOP is active, and what do you need from me next?", turn_number=3
        )
        check("uses the active SOP state", "target path" in r.text.lower(), r.text)
        r = await native.run_agentic_loop(
            "What codeword did I give you? Reply with just the codeword.", turn_number=4
        )
        check("same conversation resumed", "PELICAN-42" in r.text, r.text)
        check("vendor session kept", native._load_record().vendor_session_id == sid)
    print("S16c PASS" if not failures else f"S16c FAIL: {failures}")
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
