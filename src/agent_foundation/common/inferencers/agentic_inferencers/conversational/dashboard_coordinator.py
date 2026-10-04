# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""DashboardCoordinator — hub/dashboard business logic, extracted from the CI.

Owns three responsibilities previously inline on ConversationalInferencer
(~230 LOC total):

- ``normalize_directives(tools)`` — pre-fork: sets ``metadata.open_dashboard``
  on each tool that carries a dashboard flag, gated by the dashboard's
  ``embeds`` listing this tool's widget_type.
- ``build_seed(carrier, collected, next_dashboard_directives)`` — assembles
  the opener seed (selected ids + proposals payload + auto_implement).
- ``maybe_open(tools, collected, next_dashboard_directives)`` — post-fork
  opener (yolo + interactive): narrows the executor via Protocol, calls
  either ``HubAwareToolExecutor.create_experiment_hub`` or
  ``DashboardAwareToolExecutor.open_dashboard``.

Narrow context (Design Principle #4): receives ``tool_registry``,
``tool_dispatcher``, and ``prior_context_reader`` — never a full CI back-ref.
The framework-tier inferencer no longer imports ``HubAwareToolExecutor`` /
``DashboardAwareToolExecutor`` by name; those imports live here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    DashboardAwareToolExecutor,
    HubAwareToolExecutor,
)
from agent_foundation.resources.tools.models import ToolDefinition
from attr import attrib, attrs

logger: logging.Logger = logging.getLogger(__name__)


@attrs(kw_only=True, slots=False)
class DashboardCoordinator:
    tool_registry: Optional[dict[str, ToolDefinition]] = attrib(default=None)
    # Executor Protocol impl — may be a HubAwareToolExecutor / DashboardAwareToolExecutor.
    # None for headless/test paths — maybe_open() no-ops in that case.
    tool_dispatcher: Any = attrib(default=None)
    # Live-view reader of the CI's prior_context (dict). Called on each build.
    prior_context_reader: Callable[[], Mapping[str, Any]] = attrib()

    # -----------------------------------------------------------------
    # normalize_directives (pre-fork)
    # -----------------------------------------------------------------

    def normalize_directives(self, tools: list[ConversationTool]) -> None:
        """Generic ``--<dashboard>`` convention (pre-fork, any tool type).

        For each conversation tool carrying a dashboard flag — either the sugar
        flag named after a registered Dashboard tool (e.g. ``experiment_hub:true``)
        or the explicit ``host_dashboard:"<id>"`` — set
        ``tool.metadata["open_dashboard"] = "<dashboard_id>"`` and a default
        ``submit_label`` ("📊 Go To <label>"), gated by the dashboard's ``embeds``
        listing this tool's widget type. This single generic location feeds the
        widget relabel and the post-fork opener.

        The ``tool_registry`` is the single source of truth for Dashboard
        definitions. If the host's registry does not include Dashboard tools,
        this method correctly no-ops: showing "Go To Experiment Hub" when the
        hub isn't loadable (no executor to open it) would be a worse UX than
        the widget's generic "Advance N proposals" fallback label.
        """
        registry = self.tool_registry or {}
        dashboard_ids = {
            name
            for name, td in registry.items()
            if getattr(td, "tool_type", "") == "Dashboard"
        }
        if not dashboard_ids:
            return
        _falsey = (None, False, "", "false", "False", "0", 0)
        for tool in tools:
            meta = tool.metadata if isinstance(tool.metadata, dict) else {}
            target = meta.get("open_dashboard") or meta.get("host_dashboard")
            if not target:
                # Sugar: truthy metadata key whose (underscored) name is a
                # registered Dashboard tool id (e.g. ``experiment_hub: true``).
                for key, val in meta.items():
                    cand = str(key).replace("-", "_")
                    if cand in dashboard_ids and val not in _falsey:
                        target = cand
                        break
            if not target:
                continue
            target = str(target).replace("-", "_")
            if target not in dashboard_ids:
                logger.info(
                    "[dashboard] tool requested unknown dashboard %r; ignoring",
                    target,
                )
                continue
            dash = registry.get(target)
            dcfg = getattr(dash, "dashboard_config", None) or {}
            embeds = dcfg.get("embeds") or []
            widget_type = tool.tool_type
            if embeds and widget_type and widget_type not in embeds:
                logger.info(
                    "[dashboard] %s does not embed %r; not routing this tool",
                    target,
                    widget_type,
                )
                continue
            if not isinstance(tool.metadata, dict):
                tool.metadata = {}
            tool.metadata["open_dashboard"] = target
            label = dcfg.get("label") or target
            tool.metadata.setdefault("submit_label", f"📊 Go To {label}")

    # -----------------------------------------------------------------
    # build_seed (opener input)
    # -----------------------------------------------------------------

    def build_seed(
        self,
        carrier: ConversationTool,
        collected: Any,
        *,
        next_dashboard_directives: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the dashboard seed (selected ids + proposals_data) for the opener.

        ``selected_proposal_ids`` are AF P-ids (canonical end-to-end). Reads the
        just-collected selection from ``collected`` (interactive → str of joined
        ids; yolo → dict keyed by the tool's output var), and the proposals
        payload from the carrier's enriched ``metadata['proposals']``, falling
        back to ``prior_context['research_proposals_data']``.
        """
        meta = carrier.metadata if isinstance(carrier.metadata, dict) else {}
        output_vars = carrier.output_vars or ["selected_proposal_ids"]
        ids_raw: Any = ""
        if isinstance(collected, dict):
            for var in output_vars:
                if collected.get(var):
                    ids_raw = collected[var]
                    break
            if not ids_raw:
                ids_raw = (
                    collected.get("selected_proposal_ids")
                    or collected.get("selected_proposals")
                    or ""
                )
        elif isinstance(collected, str):
            ids_raw = collected
        elif isinstance(collected, (list, tuple)):
            ids_raw = collected
        if isinstance(ids_raw, (list, tuple)):
            selected_ids = [str(x).strip() for x in ids_raw if str(x).strip()]
        else:
            selected_ids = [s.strip() for s in str(ids_raw).split(",") if s.strip()]
        proposals_data: dict[str, Any] = {}
        raw_proposals = meta.get("proposals")
        if isinstance(raw_proposals, dict):
            proposals_data = raw_proposals
        elif isinstance(raw_proposals, list):
            proposals_data = {"proposals": raw_proposals}
        else:
            ctx = self.prior_context_reader() or {}
            ctx_raw = ctx.get("research_proposals_data")
            if isinstance(ctx_raw, str) and ctx_raw.strip():
                try:
                    parsed = json.loads(ctx_raw)
                    if isinstance(parsed, dict):
                        proposals_data = parsed
                except (json.JSONDecodeError, ValueError):
                    proposals_data = {}
            elif isinstance(ctx_raw, dict):
                proposals_data = ctx_raw
        directives = next_dashboard_directives or {}
        auto_implement = bool(directives.get("auto_implement", False))
        return {
            "dashboard_id": str(meta.get("open_dashboard") or ""),
            "selected_proposal_ids": selected_ids,
            "selected_details": [{"id": pid} for pid in selected_ids],
            "proposals_data": proposals_data,
            "custom_queries": meta.get("custom_queries"),
            "group_by": meta.get("group_by", "batch"),
            "initial_view": meta.get("initial_view", "selection"),
            "proposals_path": meta.get("proposals_path"),
            "auto_implement": auto_implement,
        }

    # -----------------------------------------------------------------
    # maybe_open (post-fork opener)
    # -----------------------------------------------------------------

    async def maybe_open(
        self,
        tools: list[ConversationTool],
        collected: Any,
        *,
        next_dashboard_directives: Optional[dict[str, Any]] = None,
    ) -> None:
        """Post-fork dashboard opener (yolo + interactive paths).

        Scans ``tools`` for the one carrying ``metadata.open_dashboard`` (single
        per group-rule (a), but scan — don't assume ``[0]``). Narrows the
        executor via Protocol and awaits the open so the subtab is active +
        persisted before the turn advances. ``selected_proposal_ids`` is already
        published by the ProposalSelectionHandler, so Phase 3 unblocks even if
        this no-ops (no hub-aware executor → graceful degrade).
        """
        carrier = next(
            (
                t
                for t in tools
                if isinstance(t.metadata, dict) and t.metadata.get("open_dashboard")
            ),
            None,
        )
        if carrier is None:
            return
        dashboard_id = str(carrier.metadata["open_dashboard"])
        if self.tool_dispatcher is None:
            logger.info(
                "[dashboard] no executor available; cannot open %s", dashboard_id
            )
            return
        try:
            seed = self.build_seed(
                carrier,
                collected,
                next_dashboard_directives=next_dashboard_directives,
            )
            if dashboard_id == "experiment_hub" and isinstance(
                self.tool_dispatcher, HubAwareToolExecutor
            ):
                await self.tool_dispatcher.create_experiment_hub(
                    selected_details=seed.get("selected_details", []),
                    proposals_data=seed.get("proposals_data", {}),
                    custom_queries=seed.get("custom_queries"),
                    group_by=seed.get("group_by", "batch"),
                    auto_implement=bool(seed.get("auto_implement", False)),
                )
            elif isinstance(self.tool_dispatcher, DashboardAwareToolExecutor):
                await self.tool_dispatcher.open_dashboard(dashboard_id, seed)
            else:
                logger.info(
                    "[dashboard] executor is not dashboard-aware; skipping "
                    "open of %s (selection already published)",
                    dashboard_id,
                )
        except Exception as exc:  # noqa: BLE001 — open is best-effort
            logger.warning(
                "[dashboard] open of %s failed (selection already published): %s",
                dashboard_id,
                exc,
            )
