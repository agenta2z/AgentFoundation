"""S7 spike: claude-agent-sdk + the pinned system Claude Code CLI, configured
the way the native ``claude_sdk`` backend configures it — preset system prompt
plus an append file through ``extra_args``, the in-process ``af`` MCP server,
hook callbacks, ``get_mcp_status`` — in one real SDK session.

The pinned ``cli_path`` is a wrapper that records the argv the SDK starts it
with and then execs the system ``claude``, so the checks see exactly what the
SDK launched:

* the SDK version is the tested one; the wrapper (the pinned path) ran, and
  the CLI that answered reports the system binary's version, not the SDK's
  bundled CLI version;
* argv: ``--append-system-prompt-file <file>``, no ``--system-prompt`` (the
  preset is kept), ``--session-id`` pinned, the SDK ``af`` server in
  ``--mcp-config``;
* ``get_mcp_status`` lists ``af`` connected with its tool before the turn; the
  turn's ``init`` lists it connected with ``mcp__af__echo``;
* hooks: ``UserPromptSubmit`` saw the prompt; ``PreToolUse`` and
  ``PostToolUse`` saw ``mcp__af__echo``;
* the in-process handler ran with the nonce and its result reached the reply;
* the transcript's recorded system prompt holds the preset and the append.

Run (from the AgentFoundation root):
    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s7_sdk_cli_pin.py [--model haiku]
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from _spike_common import (
    CallLog,
    Checks,
    echo_tool,
    hook_matcher,
    sdk_mcp_server,
    sdk_options,
    section,
    session_entries,
    short,
    snapshot_texts,
)

TESTED_SDK = "0.1.58"


def _write_wrapper(directory: str, real: str) -> tuple[str, Path]:
    """A ``cli_path`` that records its argv (NUL-separated) and execs ``real``."""
    record = Path(directory) / "argv.bin"
    wrapper = Path(directory) / "claude_pinned"
    wrapper.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\0' \"$@\" > {shlex.quote(str(record))}\n"
        f'exec {shlex.quote(real)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return str(wrapper), record


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else ""


async def _session(options: Any, prompt: str) -> dict[str, Any]:
    """Connect, read ``get_mcp_status`` until ``af`` settles, run one turn."""
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage, SystemMessage

    out: dict[str, Any] = {"status": None, "init": {}, "result": None}
    async with ClaudeSDKClient(options=options) as client:
        for _ in range(75):
            response = await client.get_mcp_status()
            servers = {s.get("name"): s for s in (response or {}).get("mcpServers", [])}
            out["status"] = servers.get("af")
            if out["status"] and out["status"].get("status") != "pending":
                break
            await asyncio.sleep(0.2)
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                out["init"] = dict(message.data)
            elif isinstance(message, ResultMessage):
                out["result"] = message
    return out


async def main() -> int:
    import claude_agent_sdk
    from claude_agent_sdk._cli_version import __cli_version__ as bundled_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    args = parser.parse_args()
    c = Checks("S7")
    work = tempfile.mkdtemp(prefix="s7_pin_")
    real = shutil.which("claude") or "claude"
    system_version = (
        subprocess.run([real, "--version"], capture_output=True, text=True)
        .stdout.strip()
        .split()[0]
    )
    sdk_dir = Path(claude_agent_sdk.__file__).parent
    c.check(
        "SDK is the tested version",
        claude_agent_sdk.__version__ == TESTED_SDK,
        claude_agent_sdk.__version__,
    )
    c.info(
        "SDK bundled CLI",
        f"version {bundled_version}; binary installed={(sdk_dir / '_bundled').exists()}",
    )
    c.info("system CLI", f"{real} {system_version}")

    sentinel = f"S7-APPEND-{uuid.uuid4().hex[:8]}"
    append = Path(work) / "l1.md"
    append.write_text(
        f"Session instructions sentinel: {sentinel}. Follow the user.\n",
        encoding="utf-8",
    )
    nonce = f"S7ECHO-{uuid.uuid4().hex[:6]}"
    hooks_seen: dict[str, list[dict[str, Any]]] = {}

    def recorder(event: str) -> Any:
        async def _hook(input_data: dict, _tool_use_id: Any, _ctx: Any) -> dict:
            hooks_seen.setdefault(event, []).append(dict(input_data))
            return {}

        return _hook

    calls = CallLog()
    wrapper, record = _write_wrapper(work, real)
    pinned = str(uuid.uuid4())
    options = sdk_options(
        cwd=work,
        model=args.model,
        session_id=pinned,
        append_file=str(append),
        mcp_servers={"af": sdk_mcp_server([echo_tool()], calls)},
        hooks={
            event: [hook_matcher(recorder(event))]
            for event in ("UserPromptSubmit", "PreToolUse", "PostToolUse")
        },
        stderr=lambda _line: None,
    )
    options.cli_path = wrapper
    prompt = (
        f"Call the echo tool with text {nonce}, then reply with exactly the "
        "tool's result and nothing else."
    )
    section("S7 one SDK session through the pinned CLI")
    out = await _session(options, prompt)

    argv = record.read_bytes().decode().split("\0")[:-1] if record.exists() else []
    init, result = out["init"], out["result"]
    text = str(getattr(result, "result", "") or "")
    c.check(
        "the pinned cli_path is what the SDK started", bool(argv), f"{len(argv)} args"
    )
    c.check(
        "the CLI that answered is the system binary, not the SDK's bundled CLI",
        init.get("claude_code_version") == system_version,
        f"init claude_code_version={init.get('claude_code_version')!r} "
        f"system={system_version} bundled={bundled_version}",
    )
    c.check(
        "argv: append file via extra_args, no --system-prompt (preset kept)",
        _flag_value(argv, "--append-system-prompt-file") == str(append)
        and "--system-prompt" not in argv
        and "--system-prompt-file" not in argv,
        [a for a in argv if a.startswith("--")],
    )
    c.check(
        "argv: pinned --session-id; the SDK af server in --mcp-config",
        _flag_value(argv, "--session-id") == pinned
        and '"af"' in _flag_value(argv, "--mcp-config")
        and '"sdk"' in _flag_value(argv, "--mcp-config"),
        f"session={short(_flag_value(argv, '--session-id'))} "
        f"mcp-config={_flag_value(argv, '--mcp-config')[:120]!r}",
    )
    status = out["status"] or {}
    c.check(
        "get_mcp_status: af connected with its tool, before the turn",
        status.get("status") == "connected"
        and any("echo" in str(t.get("name", "")) for t in status.get("tools") or []),
        {k: v for k, v in status.items() if k != "tools"},
    )
    af_init = next(
        (s for s in init.get("mcp_servers") or [] if s.get("name") == "af"), {}
    )
    c.check(
        "init: af connected, mcp__af__echo listed",
        af_init.get("status") == "connected"
        and "mcp__af__echo" in (init.get("tools") or []),
        af_init,
    )
    prompts = [h.get("prompt", "") for h in hooks_seen.get("UserPromptSubmit", [])]
    c.check("hook: UserPromptSubmit saw the prompt", prompt in prompts, prompts[:1])
    for event in ("PreToolUse", "PostToolUse"):
        names = [h.get("tool_name") for h in hooks_seen.get(event, [])]
        c.check(f"hook: {event} saw mcp__af__echo", "mcp__af__echo" in names, names)
    echo_calls = [x.arguments for x in calls.named("echo")]
    c.check(
        "in-process MCP: the handler ran with the nonce; its result is the reply",
        echo_calls == [{"text": nonce}] and f"ECHO::{nonce}" in text,
        f"calls={echo_calls} reply={text[:80]!r}",
    )
    snapshots = snapshot_texts(
        session_entries(getattr(result, "session_id", "") or pinned)
    )
    c.check(
        "transcript: the recorded system prompt holds the preset and the append",
        any(sentinel in s and "Claude Code" in s for s in snapshots),
        f"{len(snapshots)} prompt_snapshot(s)",
    )
    shutil.rmtree(work, ignore_errors=True)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
