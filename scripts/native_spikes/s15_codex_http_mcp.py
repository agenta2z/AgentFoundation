"""S15 — Codex CLI as a native backend: AF tools over localhost HTTP MCP, L1 as
``developer_instructions``, the ``--json`` event schema, resume, and hermetic
auth.

Every run uses the argv shape of the ``codex_cli`` backend
(``codex exec [resume <id>] --json --skip-git-repo-check [--ignore-user-config
--ignore-rules] -c mcp_servers.af.url=… -c mcp_servers.af.bearer_token_env_var=…
-c developer_instructions=… --dangerously-bypass-approvals-and-sandbox <prompt>``)
in a temporary ``CODEX_HOME`` (bootstrapped by the Meta launcher) where the spike
plants a user-config canary: a stdio MCP server in ``config.toml`` that records
when it is started, and a codeword in the user ``AGENTS.md``.

  inherit   (control) the canary server starts; AF tool + L1 work
  hermetic  ``--ignore-user-config --ignore-rules``: authenticates, the canary
            server does not start, AF tool + L1 still work (they ride ``-c``)
  resume    ``exec resume <thread>`` with the hermetic flags: authenticates and
            remembers the thread

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s15_codex_http_mcp.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from _spike_common import (
    canary_events,
    Checks,
    echo_tool,
    HttpMcpServer,
    run_claude as run_jsonl,  # any CLI that prints JSON lines
    section,
    short,
    SYSTEM_PYTHON,
    write_stdio_canary_server,
)

TOKEN_ENV = "AF_MCP_TOKEN"  # session/codex_cli.py _TOKEN_ENV


def _argv(
    server: HttpMcpServer, l1: str, prompt: str, *, hermetic: bool, resume: str = ""
) -> list[str]:
    argv = [shutil.which("codex") or "codex", "exec"]
    if resume:
        argv += ["resume", resume]
    argv += ["--json", "--skip-git-repo-check"]
    if hermetic:
        argv += ["--ignore-user-config", "--ignore-rules"]
    argv += [
        "-c",
        f"mcp_servers.af.url={json.dumps(server.url)}",
        "-c",
        f"mcp_servers.af.bearer_token_env_var={json.dumps(TOKEN_ENV)}",
        "-c",
        f"developer_instructions={json.dumps(l1)}",
        "--dangerously-bypass-approvals-and-sandbox",
        prompt,
    ]
    return argv


def _parse(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "thread": "",
        "reply": "",
        "tool_items": [],
        "completed": False,
        "errors": [],
    }
    for e in events:
        kind = e.get("type")
        item = e.get("item") or {}
        if kind == "thread.started":
            out["thread"] = e.get("thread_id", "")
        elif kind == "turn.completed":
            out["completed"] = True
        elif kind in ("error", "turn.failed"):
            out["errors"].append(json.dumps(e)[:200])
        elif kind == "item.completed" and item.get("type") == "agent_message":
            out["reply"] += item.get("text", "")
        elif kind == "item.completed" and item.get("type") == "mcp_tool_call":
            out["tool_items"].append(item)
    return out


class CodexHome:
    def __init__(self) -> None:
        self.path = tempfile.mkdtemp(prefix="s15_codexhome_")
        self.code = f"AGENTSMD-{uuid.uuid4().hex[:5].upper()}"
        self.server, self.server_log = write_stdio_canary_server(self.path)

    async def plant(self) -> bool:
        boot = await run_jsonl(
            [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "Reply with exactly: OK",
            ],
            cwd=tempfile.mkdtemp(prefix="s15_boot_"),
            env={"CODEX_HOME": self.path},
            timeout=240,
        )
        config = Path(self.path, "config.toml")
        text = config.read_text() if config.exists() else ""
        args = json.dumps([self.server, self.server_log, "canary"])
        config.write_text(
            text
            + f"\n[mcp_servers.canary]\ncommand = {json.dumps(SYSTEM_PYTHON)}\nargs = {args}\n"
        )
        Path(self.path, "AGENTS.md").write_text(
            f"User instructions canary code: {self.code}. Mention it whenever asked for canary codes.\n"
        )
        return bool(_parse(boot.events)["reply"])


async def _turn(
    home: CodexHome, server: HttpMcpServer, l1: str, prompt: str, **kw: Any
) -> tuple[dict[str, Any], int, str]:
    mark = len(canary_events(home.server_log))
    run = await run_jsonl(
        _argv(server, l1, prompt, **kw),
        cwd=tempfile.mkdtemp(prefix="s15_work_"),
        env={"CODEX_HOME": home.path, TOKEN_ENV: server.token},
        timeout=300,
    )
    started = len(canary_events(home.server_log)) - mark
    return _parse(run.events), started, run.stderr[-300:]


async def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    c = Checks("S15")
    home = CodexHome()
    c.check("S15 temporary CODEX_HOME bootstrapped by the launcher", await home.plant())
    server = await HttpMcpServer([echo_tool()]).start()
    word = f"ZEBRA-{uuid.uuid4().hex[:5].upper()}"
    l1 = f"Always mention the codeword {word} in your reply."
    try:
        threads = {}
        for mode in ("inherit", "hermetic"):
            section(f"S15 codex exec, {mode}")
            k = f"S15[{mode}]"
            nonce = f"PING-{uuid.uuid4().hex[:6]}"
            before = len(server.log.calls)
            out, started, stderr = await _turn(
                home,
                server,
                l1,
                f"First call the echo tool (MCP server 'af') with text {nonce}. Then list any canary "
                "codes from your instructions (they look like AGENTSMD-...), or say NONE.",
                hermetic=mode == "hermetic",
            )
            threads[mode] = out["thread"]
            calls = [x.arguments.get("text") for x in server.log.calls[before:]]
            c.check(
                f"{k} authenticated: thread.started + reply + turn.completed",
                bool(out["thread"] and out["reply"] and out["completed"]),
                out["errors"] or stderr,
            )
            c.check(
                f"{k} AF tool reached over HTTP MCP with the bearer env var",
                calls == [nonce],
                calls,
            )
            c.check(
                f"{k} mcp_tool_call item in --json output",
                bool(out["tool_items"]),
                [sorted(i) for i in out["tool_items"]][:1],
            )
            c.check(
                f"{k} developer_instructions (L1) followed",
                word in out["reply"],
                out["reply"][:160],
            )
            c.check(
                f"{k} user config.toml MCP server started: {started > 0}",
                (started > 0) == (mode == "inherit"),
                started,
            )
            c.info(f"{k} user AGENTS.md canary in reply", home.code in out["reply"])

        section("S15 codex exec resume, hermetic")
        thread = threads.get("hermetic", "")
        out, started, stderr = await _turn(
            home,
            server,
            l1,
            "Which text did you send to the echo tool in your previous turn? Reply with only that text.",
            hermetic=True,
            resume=thread,
        )
        c.check(
            "S15[resume] authenticated and continued the thread",
            out["completed"] and bool(out["reply"]),
            out["errors"] or stderr,
        )
        c.check(
            "S15[resume] same thread",
            out["thread"] in ("", thread),
            f"{short(thread)} -> {short(out['thread'])}",
        )
        c.check(
            "S15[resume] remembers the previous turn",
            "PING-" in out["reply"],
            out["reply"][:120],
        )
        c.check(
            "S15[resume] user config.toml MCP server not started", started == 0, started
        )
    finally:
        await server.stop()
    return c.exit_code()


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())
    sys.exit(asyncio.run(main()))
