"""S6 — AF tools under Claude Code's tool search, and the managed policy for a
custom MCP server ``af``.

A. Managed settings (static read of ``/etc/claude-code/*.json``): which
   profiles set ``allowManagedMcpServersOnly`` (a custom ``af`` server is then
   refused unless allowlisted), ``allowManagedHooksOnly`` (the native backends'
   hooks — L2, turn stop, subagent deny — are then ignored) and
   ``disableBypassPermissionsMode``. Only the default profile can be exercised
   live on a devserver; the sensitive profiles are reported, not run.

B. Tool search, live, with the real AF surface (4 action tools, 5 widgets,
   5 SOP-control tools = 14) and with 26 more realistic action tools (40):
   are the ``mcp__af__*`` tools deferred (transcript ``deferred_tools_delta``
   attachment, ``ToolSearch`` in the init tool list) or sent up front, is ``af``
   connected, and does the model still find and call the right AF tool from a
   task description (``enter_sop`` for "start the model_optimization SOP",
   ``single_choice`` for a one-of-three question)? Agent SDK (in-process
   server) and ``claude -p`` (HTTP server).

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s6_tool_search_managed.py [--kinds sdk,cli] [--model sonnet]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from _spike_common import (
    api_usage,
    attachments,
    CallLog,
    Checks,
    claude_argv,
    HttpMcpServer,
    run_claude,
    run_sdk,
    sdk_mcp_server,
    sdk_options,
    section,
    session_entries,
    SpikeTool,
)

_POLICY_KEYS = (
    "allowManagedMcpServersOnly",
    "allowManagedHooksOnly",
    "allowManagedPermissionRulesOnly",
    "disableBypassPermissionsMode",
)

_WIDGETS = {
    "clarification": "Ask the user one free-text question (conversation widget). The answer arrives as the next message.",
    "single_choice": "Ask the user to pick exactly one option from a list (conversation widget).",
    "multiple_choice": "Ask the user to pick one or more options from a list (conversation widget).",
    "confirmation": "Ask the user to confirm or reject a proposed action (conversation widget).",
    "proposal_selection": "Let the user select research proposals from a generated proposals file (conversation widget).",
}
_SOP = {
    "enter_sop": "Enter (start) a Standard Operating Procedure (SOP) by name, optionally with the user's request.",
    "resume_sop": "Resume a paused or in-progress SOP.",
    "pause_sop": "Pause the active SOP.",
    "exit_sop": "Exit the active SOP.",
    "sop_status": "Show the active SOP's phase and next step.",
}
_EXTRA = (
    (
        "dataset_profile",
        "Profile a Hive dataset: row counts, null rates and value distributions per column.",
    ),
    (
        "experiment_launch",
        "Launch a training experiment from a config file on the cluster.",
    ),
    (
        "experiment_status",
        "Report the status, metrics and logs of a running training experiment.",
    ),
    ("experiment_compare", "Compare evaluation metrics of two finished experiments."),
    ("feature_lookup", "Look up a ranking feature's definition, owner and coverage."),
    (
        "model_registry_search",
        "Search the model registry for published model snapshots.",
    ),
    (
        "eval_run",
        "Run an offline evaluation of a model snapshot on a held-out dataset.",
    ),
    ("metric_definition", "Explain how a product metric is defined and computed."),
    ("dashboard_snapshot", "Render a dashboard chart to an image for a report."),
    ("query_presto", "Run a read-only Presto SQL query and return the first rows."),
    ("code_search", "Search the monorepo for symbols, files or text."),
    ("diff_summary", "Summarize a code review diff: files touched and intent."),
    ("task_create", "Create a work-tracking task with a title and description."),
    ("task_comment", "Add a comment to an existing work-tracking task."),
    ("oncall_lookup", "Find who is on call for a service or team."),
    ("doc_search", "Search internal wiki and documentation pages."),
    ("doc_write", "Create or update a document from markdown."),
    ("capacity_check", "Check GPU capacity and quota for a team."),
    ("job_logs", "Fetch the logs of a cluster job by id."),
    ("job_cancel", "Cancel a running cluster job by id."),
    ("notebook_create", "Create an analysis notebook pre-filled with a query."),
    (
        "ab_test_readout",
        "Summarize an A/B test's readout: metrics, significance, decision.",
    ),
    ("alert_status", "List firing alerts for a service."),
    ("config_get", "Read a configuration value from the config store."),
    ("schema_describe", "Describe a table's schema, partitions and retention."),
    ("lineage_upstream", "List the upstream tables and pipelines of a dataset."),
)


def _schema(kind: str) -> dict[str, Any]:
    if kind in _WIDGETS:
        props: dict[str, Any] = {
            "prompt": {"type": "string"},
            "output": {"type": "array", "items": {"type": "string"}},
        }
        if kind in ("single_choice", "multiple_choice"):
            props["choices"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            }
        return props
    if kind == "enter_sop":
        return {
            "name": {"type": "string"},
            "request": {"type": "string"},
            "yolo": {"type": "boolean"},
            "fresh": {"type": "boolean"},
        }
    return {"request": {"type": "string"}}


def af_tools(total: int) -> list[SpikeTool]:
    from agent_foundation.resources.tools.registry import load_all_tools

    async def _ok(args: dict[str, Any]) -> str:
        return "AF_END_TURN — done; the result reaches the user."

    registry = load_all_tools()
    tools = []
    for name, tool in sorted(registry.items()):
        if tool.tool_type == "Action" and getattr(tool, "agent_enabled", True):
            props = {
                p.name.lstrip("-").replace("-", "_"): {"type": "string"}
                for p in tool.parameters or []
            }
            tools.append(SpikeTool(name, tool.description or name, props, _ok))
    for name, desc in {**_WIDGETS, **_SOP}.items():
        tools.append(SpikeTool(name, desc, _schema(name), _ok))
    for name, desc in _EXTRA[: max(0, total - len(tools))]:
        tools.append(SpikeTool(name, desc, _schema(name), _ok))
    return tools


def managed_policy(c: Checks) -> None:
    section("S6-A managed Claude Code settings (/etc/claude-code)")
    files = sorted(glob.glob("/etc/claude-code/*.json"))
    c.check("S6 managed settings files readable", bool(files), files)
    default = None
    for path in files:
        data = json.loads(Path(path).read_text())
        values = {
            k: data.get(k, data.get("permissions", {}).get(k)) for k in _POLICY_KEYS
        }
        allowed = [
            " ".join(s.get("serverCommand", [])[-1:])
            for s in data.get("allowedMcpServers", [])
        ]
        c.info(Path(path).name, f"{values} allowedMcpServers={allowed}")
        if Path(path).name == "managed-settings.json":
            default = values
    c.check(
        "S6 default profile allows a custom MCP server (no allowManagedMcpServersOnly)",
        default is not None and not default["allowManagedMcpServersOnly"],
        default,
    )
    c.check(
        "S6 default profile allows session hooks (no allowManagedHooksOnly)",
        default is not None and not default["allowManagedHooksOnly"],
        default,
    )


async def _turns(
    kind: str, model: str, tools: list[SpikeTool], calls: CallLog, prompts: list[str]
) -> tuple[str, list, list]:
    work = tempfile.mkdtemp(prefix=f"s6_{kind}_{len(tools)}_")
    sid = str(uuid.uuid4())
    inits: list[dict[str, Any]] = []
    texts: list[str] = []
    if kind == "sdk":
        server = sdk_mcp_server(tools, calls)
        for i, prompt in enumerate(prompts):
            run = await run_sdk(
                sdk_options(
                    cwd=work,
                    model=model,
                    session_id="" if i else sid,
                    resume=sid if i else "",
                    mcp_servers={"af": server},
                ),
                prompt,
            )
            init = run.system("init")
            inits.append(dict(init[0].data) if init else {})
            texts.append(run.text or run.error)
        return sid, inits, texts
    server = await HttpMcpServer(tools, calls).start()
    try:
        config = server.write_config(work)
        for i, prompt in enumerate(prompts):
            run = await run_claude(
                claude_argv(
                    prompt,
                    model=model,
                    session_id="" if i else sid,
                    resume=sid if i else "",
                    mcp_config=config,
                    allowed_tools=["mcp__af"],
                ),
                cwd=work,
            )
            inits.append(run.init)
            texts.append(run.text or run.stderr[-200:])
    finally:
        await server.stop()
    return sid, inits, texts


async def tool_search(kind: str, model: str, total: int, c: Checks) -> None:
    tools = af_tools(total)
    k = f"S6[{kind}, {len(tools)} AF tools]"
    section(k)
    calls = CallLog()
    sid, inits, texts = await _turns(
        kind,
        model,
        tools,
        calls,
        [
            "Start the model_optimization SOP for me.",
            "Ask me which color I prefer among red, green and blue, as a question where I pick exactly one.",
        ],
    )
    init = inits[0]
    af_status = next(
        (
            s.get("status")
            for s in init.get("mcp_servers") or []
            if s.get("name") == "af"
        ),
        None,
    )
    c.check(
        f"{k} af MCP server connected",
        af_status == "connected",
        init.get("mcp_servers"),
    )
    listed = [t for t in init.get("tools") or [] if str(t).startswith("mcp__af__")]
    entries = session_entries(sid)
    deferred = set()
    for e in attachments(entries, "deferred_tools_delta"):
        deferred |= {
            n
            for n in e["attachment"].get("addedNames") or []
            if n.startswith("mcp__af__")
        }
    c.info(
        f"{k} init lists {len(listed)} af tools; ToolSearch in init tools",
        "ToolSearch" in (init.get("tools") or []),
    )
    c.check(
        f"{k} every AF tool is deferred behind ToolSearch",
        len(deferred) == len(tools),
        f"{len(deferred)}/{len(tools)} in deferred_tools_delta",
    )
    usage = api_usage(entries)
    if usage:
        first = usage[0]
        c.info(
            f"{k} first request tokens",
            {key: first[key] for key in ("input", "cache_write", "cache_read")},
        )
    sop = [x.arguments for x in calls.named("enter_sop")]
    c.check(
        f"{k} 'start the model_optimization SOP' -> enter_sop(name=model_optimization)",
        any("model_optimization" in str(a.get("name", "")) for a in sop),
        f"enter_sop calls {sop}; all calls {[x.name for x in calls.calls]}",
    )
    c.check(
        f"{k} 'pick exactly one' -> single_choice",
        bool(calls.named("single_choice")),
        [x.name for x in calls.calls],
    )


async def sdk_health(model: str, c: Checks) -> None:
    """What the ``claude_sdk`` backend's health check sees before the first turn."""
    from claude_agent_sdk import ClaudeSDKClient

    section("S6 SDK get_mcp_status before any turn")
    options = sdk_options(
        cwd=tempfile.mkdtemp(prefix="s6_health_"),
        model=model,
        mcp_servers={"af": sdk_mcp_server(af_tools(14), CallLog())},
        stderr=lambda _line: None,
    )
    status: Any = None
    async with ClaudeSDKClient(options=options) as client:
        for _ in range(75):
            response = await client.get_mcp_status()
            servers = {s.get("name"): s for s in (response or {}).get("mcpServers", [])}
            status = servers.get("af")
            if status is not None and status.get("status") != "pending":
                break
            await asyncio.sleep(0.2)
    summary = {k: v for k, v in (status or {}).items() if k != "tools"}
    c.check(
        "S6[sdk] get_mcp_status lists the in-process af server as connected",
        bool(status) and status.get("status") == "connected",
        summary,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--kinds", default="sdk,cli")
    parser.add_argument("--sizes", default="14,40")
    args = parser.parse_args()
    c = Checks("S6")
    managed_policy(c)
    if "sdk" in args.kinds.split(","):
        await sdk_health(args.model, c)
    for kind in args.kinds.split(","):
        for total in (int(s) for s in args.sizes.split(",")):
            await tool_search(kind.strip(), args.model, total, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
