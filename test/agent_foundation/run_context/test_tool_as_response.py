"""A tool call's structured response lives in the invocation (plan v8 §13, P10; B28).

``ToolAsInferencer`` builds its ``ToolInferencerResponse`` when the subprocess stream
ends and ``_ainfer`` of the same invocation returns it; it is never stored on the
instance, so overlapping calls on one tool each return their own. ``cancel()``
terminates every subprocess the instance is running.
"""

from __future__ import annotations

import asyncio
import sys

from agent_foundation.common.inferencers.agentic_inferencers.tool_inferencers.tool_as_inferencer import (
    ToolAsInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext

# Prints its argument and exits with the given code; "slow" lingers first.
SCRIPT = (
    "import sys, time\n"
    "name, code = sys.argv[1], int(sys.argv[2])\n"
    "time.sleep(0.4 if name == 'slow' else 0)\n"
    "print(name)\n"
    "sys.exit(code)\n"
)


def _tool(tmp_path):
    script = tmp_path / "tool.py"
    script.write_text(SCRIPT, encoding="utf-8")
    return ToolAsInferencer(
        tool_name="echoer",
        command=["python3", str(script)],
        args_template=["${name}", "${code}"],
        env={"PATH": sys.exec_prefix + "/bin:/usr/bin:/bin"},
        success_check=lambda return_code, stdout: return_code == 0,
    )


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_overlapping_host_calls_on_one_tool_each_return_their_own_response(tmp_path):
    tool = _tool(tmp_path)

    async def main():
        return await asyncio.gather(
            tool.ainfer({"name": "slow", "code": 4}, run_context=_host(tmp_path / "a")),
            tool.ainfer({"name": "fast", "code": 0}, run_context=_host(tmp_path / "b")),
        )

    slow, fast = asyncio.run(main())
    assert (slow.stdout.strip(), slow.return_code, slow.success) == ("slow", 4, False)
    assert (fast.stdout.strip(), fast.return_code, fast.success) == ("fast", 0, True)
    assert "_last_response" not in vars(tool) and tool._procs == set()


def test_cancel_terminates_the_running_subprocess(tmp_path):
    tool = _tool(tmp_path)

    async def started():
        while not tool._procs:
            await asyncio.sleep(0.01)

    async def main():
        call = asyncio.ensure_future(tool.ainfer({"name": "slow", "code": 0}))
        await asyncio.wait_for(started(), 5)
        await tool.cancel()
        return await asyncio.wait_for(call, 5)

    response = asyncio.run(main())
    assert response.return_code != 0 and tool._procs == set()
