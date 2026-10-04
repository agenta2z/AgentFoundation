"""S16 — Metamate as a tool-less native backend: does ``internal_prompt`` work
as the per-request carrier of session instructions + turn context?

Questions:
  1. Does the model follow ``internal_prompt`` (visibility)?
  2. Is it persisted in the conversation (does turn 2, sent without it, still
     follow it)?
  3. Is it exposed as a visible message in the conversation history?
  4. Does a changed ``internal_prompt`` on a later request take effect?
  5. Can one turn's reply be told apart from earlier turns by message uuid
     (the poll returns the whole conversation)?

    buck2 run @//mode/dbgo //_tony_dev/CoreProjects/AgentFoundation/scripts/native_spikes:s16_metamate_internal_prompt
"""

from __future__ import annotations

import time
import uuid

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    DEFAULT_API_KEY,
    DEFAULT_MODE,
    DEFAULT_STREAM_TYPE,
    DEFAULT_SURFACE,
    resolve_metamate_client_cls,
)

_TERMINAL = {"COMPLETED", "ERROR", "FAILED", "CANCELLED", "ABORTED"}


def _text_of(outputs, message_uuids: set[str]) -> str:
    parts = []
    for out in outputs:
        block = getattr(out, "block", None)
        if block is None or getattr(block, "message_uuid", None) not in message_uuids:
            continue
        content = getattr(block, "content", None)
        for attr, field in (
            ("markdown", "value"),
            ("agent_message", "markdown"),
            ("text_string", "value"),
        ):
            value = getattr(getattr(content, attr, None), field, None)
            if value:
                parts.append(value)
    return "\n".join(parts)


def _messages(outputs) -> list:
    return [m for m in (getattr(o, "message", None) for o in outputs) if m is not None]


def turn(client, prompt: str, internal: str | None, conv: tuple | None) -> tuple:
    before = set()
    if conv:
        before = {
            m.uuid for m in _messages(client.get_conversation_for_stream(conv[0]))
        }
    result = client.engine_start_v2(
        prompt=prompt,
        request_id=str(uuid.uuid4()),
        api_key=DEFAULT_API_KEY,
        surface=DEFAULT_SURFACE,
        mode=DEFAULT_MODE,
        stream_type=DEFAULT_STREAM_TYPE,
        internal_prompt=internal,
        conversation_uuid=conv[0] if conv else None,
        conversation_fbid=conv[1] if conv else None,
        timeout_seconds=600,
    )
    conv = (result.conversation.uuid, result.conversation.fbid)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        outputs = client.get_conversation_for_stream(conv[0])
        new = [m for m in _messages(outputs) if m.uuid not in before]
        assistant = [
            m for m in new if str(getattr(m, "role", "")).upper() == "ASSISTANT"
        ]
        status = str(assistant[-1].status).split(".")[-1].upper() if assistant else ""
        if status in _TERMINAL:
            roles = [
                (str(getattr(m, "role", "")), str(m.status).split(".")[-1]) for m in new
            ]
            text = _text_of(outputs, {m.uuid for m in assistant})
            visible_internal = bool(internal) and internal[:40] in _text_of(
                outputs, {m.uuid for m in _messages(outputs)}
            )
            return conv, text, roles, visible_internal
        time.sleep(3)
    raise TimeoutError("no terminal status")


_HOST_CONTEXT = """## How this session works
You are working inside the AgentFoundation host application, which runs
Standard Operating Procedures (SOPs) with the user. The user starts or resumes
an SOP by typing `/sop <name>`; you cannot start one yourself.

## Available SOPs
- **Model Optimization** (`model_optimization`): optimize ML models through
  research-propose-experiment-analyze cycles.
- **Code Optimization** (`code_optimization`): investigate a codebase and
  propose ranked refactors.

<af_context generation="1" origin="user">
Active SOP: Model Optimization — current phase 0a "Setup workflow target path".
Next step: ask the user for the workflow target path (a directory under the
session root /home/user/project). Phases that need host tools must run on a
tool-capable backend.
</af_context>"""


def context_check(client) -> None:
    """Host context (not a formatting override) carried by internal_prompt."""
    conv, text, _roles, visible = turn(
        client,
        "Which SOPs can you run here? List their names only.",
        _HOST_CONTEXT,
        None,
    )
    print("C1:", text[-400:])
    print(
        "C1 uses the SOP catalog:",
        "Model Optimization" in text and "Code Optimization" in text,
    )
    print("C1 internal visible in history:", visible)
    conv, text, _roles, _ = turn(
        client, "What SOP is active, and what is the next step?", _HOST_CONTEXT, conv
    )
    print("C2:", text[-400:])
    print("C2 uses the active SOP state:", "target path" in text.lower())
    conv, text, _roles, _ = turn(
        client, "Remind me which SOP is active? One line.", None, conv
    )
    print("C3 (no internal_prompt):", text[-300:])
    print("C3 context persisted:", "model optimization" in text.lower())


def main() -> None:
    client = resolve_metamate_client_cls(None)(cat=None)
    context_check(client)
    rule = (
        "Host instructions for this session: you are the assistant 'Zeta'. "
        "End EVERY reply with the exact token [ZETA-END]."
    )
    conv, text, roles, visible = turn(
        client, "Hi! In one sentence, who are you?", rule, None
    )
    print("T1 roles:", roles, "| internal visible in history:", visible)
    print("T1:", text[-300:])
    print("Q1 follows internal_prompt:", "[ZETA-END]" in text)

    conv, text, roles, _ = turn(client, "What is 2+2? Answer briefly.", None, conv)
    print("T2 roles:", roles)
    print("T2:", text[-300:])
    print("Q2 persisted without resending:", "[ZETA-END]" in text)

    changed = (
        "Host instructions: end EVERY reply with the exact token [OMEGA-END] instead."
    )
    conv, text, roles, _ = turn(client, "What is 3+3? Answer briefly.", changed, conv)
    print("T3:", text[-300:])
    print("Q4 changed internal_prompt takes effect:", "[OMEGA-END]" in text)
    print(
        "Q5 per-turn text isolated:",
        "2+2" not in text and "4" not in text.split("[")[0][:3],
    )
    print("conversation:", conv)


if __name__ == "__main__":
    main()
