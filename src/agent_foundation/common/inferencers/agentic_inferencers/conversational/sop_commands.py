"""SOPCommandsMixin — the slash commands shared by the conversational
orchestrators.

The SOP commands parse their argument line and delegate to ``SOPController``
(``enter`` / ``resume`` / ``cmd_*``), passing the host's ``_enter_sop`` /
``_reload_sop_definition`` as the state builder / reloader so instance
monkey-patches of those keep working. Discovery is by ``CommandRegistry``
scanning the host's MRO for ``@command`` methods; the declaration order here is
the order the classic prompt lists them in. A host changes a command's behavior
by overriding its method, which keeps the command's position.
"""

from __future__ import annotations

from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    command,
)


class SOPCommandsMixin:
    """``/help``, ``/status``, ``/clear``, ``/sop``, ``/pause_sop``,
    ``/exit_sop``, ``/resume_sop``, ``/model``, ``/root`` and ``/target``."""

    @command("help", description="List available commands", aliases=("?",))
    async def _cmd_help(self) -> str:
        lines = ["Available commands:"]
        for meta in self._commands.list_commands():
            aliases = (
                f" (aliases: {', '.join('/' + a for a in meta.aliases)})"
                if meta.aliases
                else ""
            )
            lines.append(f"  /{meta.name}{aliases} — {meta.description}")
        return "\n".join(lines)

    @command("status", description="Show SOP state and session info", aliases=("s",))
    async def _cmd_status(self) -> str:
        # Phase K5: thin wrapper. SOP-portion of /status lives on the controller.
        return self.sop_controller.cmd_status_summary(len(self._messages))

    @command("clear", description="Clear conversation history")
    async def _cmd_clear(self) -> str:
        self._messages = []
        return "Conversation history cleared."

    @command(
        "sop",
        aliases=("enter_sop",),
        description=(
            "Enter an SOP, optionally with an initial request to start on. "
            "Usage: /enter_sop <name> [--yolo] [--fresh] [request...]"
        ),
        requires_args=True,
    )
    async def _cmd_sop(self, args: str = "") -> str:
        return self.sop_controller.cmd_sop(
            args,
            build_state=self._enter_sop,
            yolo_mode_setter=self._set_yolo_mode,
        )

    @command(
        "pause_sop",
        description="Pause the active SOP for a short ad-hoc diversion",
        requires_active_sop=True,
    )
    async def _cmd_pause_sop(self) -> str:
        return self.sop_controller.cmd_pause_sop()

    @command(
        "exit_sop",
        description="Exit the active SOP (resumable later)",
        requires_active_sop=True,
    )
    async def _cmd_exit_sop(self) -> str:
        return self.sop_controller.cmd_exit_sop()

    @command(
        "resume_sop",
        description=(
            "Resume a paused or exited SOP (optionally by name), optionally "
            "with a request to continue on. Usage: /resume_sop [name] [request...]"
        ),
        requires_args=True,
    )
    async def _cmd_resume_sop(self, args: str = "") -> str:
        return self.sop_controller.cmd_resume_sop(
            args, reload=self._reload_sop_definition
        )

    @command(
        "model",
        description="Change the LLM model",
        aliases=("set_model",),
        requires_args=True,
    )
    async def _cmd_set_model(self, model_name: str = "") -> str:
        if not model_name:
            current = self.prior_context.get("model_name", "default")
            return f"Current model: {current}. Usage: /model <name>"
        self.prior_context["model_name"] = model_name
        return f"Model set to {model_name}."

    @command(
        "root",
        description="Set the session root directory",
        aliases=("set_session_root",),
        requires_args=True,
    )
    async def _cmd_set_session_root(self, path: str = "") -> str:
        if not path:
            current = self.prior_context.get("session_root_path", "not set")
            return f"Current session root: {current}. Usage: /root <path>"
        self.prior_context["session_root_path"] = path
        return f"Session root set to {path}."

    @command(
        "target",
        description="Set the workflow target path",
        aliases=("set_workflow_target_path",),
        requires_args=True,
    )
    async def _cmd_set_target(self, path: str = "") -> str:
        if not path:
            current = self.prior_context.get("workflow_target_path", "not set")
            return f"Current target path: {current}. Usage: /target <path>"
        self.prior_context["workflow_target_path"] = path
        return f"Target path set to {path}."

    def _set_yolo_mode(self, value: bool) -> None:
        self.yolo_mode = value
