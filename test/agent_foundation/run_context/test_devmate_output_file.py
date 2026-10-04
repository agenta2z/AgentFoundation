"""devmate's dump file is per invocation (plan v8 §13, P10; B28 family, inventory §23).

``construct_command`` creates the call's ``--dump-final-structs-to-file`` path and
``parse_output`` reads and deletes it. Inside an invocation the path is a component
of that invocation, so overlapping calls on one leaf never read or delete each
other's dump; a direct hook call outside one keeps it on the instance.
"""

from __future__ import annotations

import asyncio
import os

from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
    devmate_cli_inferencer as devmate_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.run_context import aopen_invocation


def _devmate(tmp_path, monkeypatch):
    monkeypatch.setattr(devmate_module, "sync_config_to_target", lambda *a, **k: None)
    (tmp_path / ".sl").mkdir(exist_ok=True)
    return DevmateCliInferencer(
        target_path=str(tmp_path), dump_output=True, cli_mode="run"
    )


def test_overlapping_invocations_each_keep_their_own_dump_file(tmp_path, monkeypatch):
    devmate = _devmate(tmp_path, monkeypatch)
    seen = {}

    async def call(label, delay):
        async with aopen_invocation(devmate):
            devmate.construct_command({"prompt": label})
            created = devmate._output_file
            await asyncio.sleep(delay)
            seen[label] = (created, devmate._output_file)
            devmate._cleanup_output_file()

    async def main():
        await asyncio.gather(call("a", 0.05), call("b", 0))

    asyncio.run(main())
    (a_created, a_seen), (b_created, b_seen) = seen["a"], seen["b"]
    assert a_created != b_created
    assert (a_seen, b_seen) == (a_created, b_created)
    assert devmate.__dict__.get("_output_file") is None
    assert not os.path.exists(a_created) and not os.path.exists(b_created)


def test_a_direct_hook_call_keeps_the_dump_file_on_the_instance(tmp_path, monkeypatch):
    devmate = _devmate(tmp_path, monkeypatch)
    devmate.construct_command({"prompt": "q"})
    path = devmate.__dict__.get("_output_file")
    assert path is not None and path == devmate._output_file
    devmate._cleanup_output_file()
    assert devmate._output_file is None and not os.path.exists(path)
