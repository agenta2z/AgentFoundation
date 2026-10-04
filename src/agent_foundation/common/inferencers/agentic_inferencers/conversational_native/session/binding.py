"""SessionBinding — which inferencer currently drives a live vendor session.

A ``SessionActor`` outlives the inferencer that started it: hosts evict and
rebuild their per-conversation inferencer (backend switch, restart of the
inferencer cache) while the runtime manager keeps the vendor session alive for
the rebuilt one. The vendor's tool handlers and hooks were registered once, at
session open, so they must not close over the inferencer that opened the
session. They resolve the current host through this binding instead, and the
turn driver rebinds it to itself before every vendor turn.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BridgeToolSpec,
)


class SessionBinding:
    """Implements ``SessionHooks`` and the tool handlers by forwarding to the
    bound host (a ``NativeConversationalInferencer``)."""

    def __init__(self, host: Any) -> None:
        self._host = host

    @property
    def host(self) -> Any:
        return self._host

    def bind(self, host: Any) -> None:
        self._host = host

    # -- tools ---------------------------------------------------------------

    def tools(self, manifest: list[BridgeToolSpec]) -> list[BridgeToolSpec]:
        """The vendor-facing tool specs: same names/schemas as ``manifest``,
        handlers routed through the binding by tool name."""
        return [
            BridgeToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
                handler=self._handler(spec.name),
            )
            for spec in manifest
        ]

    def _handler(self, name: str):
        async def handler(args: dict[str, Any]) -> Any:
            return await self._host.bridge.call(name, args)

        return handler

    # -- SessionHooks --------------------------------------------------------

    def l2_for_turn(self) -> str:
        return self._host.l2_for_turn()

    async def before_af_tool(
        self, tool_name: str, tool_use_id: str, agent_id: Optional[str]
    ) -> Optional[str]:
        return await self._host.before_af_tool(tool_name, tool_use_id, agent_id)

    async def after_af_tool(self, tool_name: str, tool_use_id: str) -> bool:
        return await self._host.after_af_tool(tool_name, tool_use_id)

    def on_compaction(self) -> None:
        self._host.on_compaction()
