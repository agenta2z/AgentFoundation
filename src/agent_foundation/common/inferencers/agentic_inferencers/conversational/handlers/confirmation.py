# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ConfirmationHandler — confirm widget with action-tool config panel.

Owns the largest chunk of legacy enrichment logic: builds tool_params from
paired action_tools + tool_registry + resolve_tool_name; resolves view path
+ view_label via 3-step fallback chain (phase_outputs viewable_artifact_path,
workflow_target_path docs, tool registry viewable_output_label).

Mutation contract: writes to `tool.metadata["tool_params"]`, `tool.metadata["view"]`,
`tool.metadata["view_label"]` only. Does NOT add dynamic attributes.

handle_response decodes the widget's `{"choice", "param_overrides", "variables"}`
shape into typed effects: OverrideNextActionToolArgs, SetTurnVariables, and —
on confirm-yes — ApplyContextUpdates({"_confirmation_gate_passed": True}).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    ApplyContextUpdates,
    OverrideNextActionToolArgs,
    SetTurnVariables,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
    HandlerContext,
    HandlerResult,
    InferencerEffect,
)
from agent_foundation.common.ui.input_modes import (
    InputMode,
    InputModeConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


class ConfirmationHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.CONFIRMATION

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        metadata: dict[str, Any] = {
            "widget_type": "confirmation",
            "note_variable": "additional_instructions",
        }
        if tool.metadata:
            metadata.update(tool.metadata)
        # `tool_params` is now baked into tool.metadata by enrich_before_send;
        # the .update() above already pulls it in. No separate overlay needed.
        return InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=tool.prompt,
            metadata=metadata,
        )

    async def enrich_before_send(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> None:
        if not tool.metadata:
            tool.metadata = {}

        # 1. Build tool_params from paired action_tools (config-panel feature).
        # Skipped on standalone path where ctx.action_tools is None — preserves
        # today's behavior (the config panel only appears in the bundled flow).
        if (
            ctx.action_tools is not None
            and ctx.tool_registry is not None
            and ctx.resolve_tool_name is not None
        ):
            tool_params: list[dict[str, Any]] = []
            for at in ctx.action_tools:
                tool_name = at.get("name", "")
                canonical = ctx.resolve_tool_name(tool_name)
                tool_def = ctx.tool_registry.get(canonical)
                if tool_def:
                    tool_params.extend(
                        p.to_dict() for p in tool_def.parameters if not p.positional
                    )
            if tool_params:
                tool.metadata["tool_params"] = tool_params

        # 2. View path resolution — validate LLM-provided view, else fallback.
        # Snapshot whether the LLM supplied `view` BEFORE tier-1 deletion so we
        # can distinguish "LLM omitted view" (no fallback should fire) from
        # "LLM provided invalid view" (rescue chain should attempt to recover).
        # Without this gate, fallbacks fire on every confirmation widget — even
        # pre-action gates where the LLM intentionally omitted view because no
        # artifact exists yet — producing a phantom "View" button that links
        # nowhere or to a stale prior-phase artifact (cross-phase pollution).
        llm_supplied_view = "view" in tool.metadata
        llm_view = tool.metadata.get("view")
        if llm_view and not Path(llm_view).is_file():
            logger.info(
                "CONFIRMATION: LLM view path invalid (not a file): %s — will override",
                str(llm_view)[:100],
            )
            del tool.metadata["view"]

        # Fallback chain only runs when the LLM tried to supply a view but it
        # was invalid (tier-1 deleted it). If the LLM never supplied view at
        # all, leave metadata empty — the React widget will render without a
        # View button, matching the LLM's intent.
        run_view_fallback = llm_supplied_view and "view" not in tool.metadata

        # 2a. phase_outputs viewable_artifact_path
        if run_view_fallback:
            phase_outputs = ctx.prior_context.get("phase_outputs", {})
            if isinstance(phase_outputs, dict):
                viewable = phase_outputs.get("viewable_artifact_path")
                if viewable and Path(viewable).exists():
                    tool.metadata["view"] = viewable
                    label = phase_outputs.get("viewable_artifact_label")
                    if label:
                        tool.metadata["view_label"] = label
                    logger.info(
                        "CONFIRMATION: view set from viewable_artifact_path: %s (label=%s)",
                        viewable,
                        label,
                    )

        # 2b. workflow_target_path docs index
        if run_view_fallback and "view" not in tool.metadata:
            workflow_target_path = ctx.prior_context.get("workflow_target_path", "")
            if workflow_target_path and workflow_target_path != "not set":
                target_dir = Path(workflow_target_path)
                if target_dir.is_file():
                    target_dir = target_dir.parent
                docs_index = target_dir / "docs" / "_build" / "html" / "index.html"
                if docs_index.exists():
                    tool.metadata["view"] = str(docs_index)
                    logger.info(
                        "CONFIRMATION: view set from fallback: %s",
                        str(docs_index),
                    )

        # 3. view_label resolution from tool registry via tool_phase_map.
        if "view" in tool.metadata and "view_label" not in tool.metadata:
            tool_name = ""
            try:
                tool_map = ctx.prior_context.get("tool_phase_map", {})
                completed = ctx.prior_context.get("completed_phases", [])
                if completed:
                    last_phase = completed[-1]
                    last_phase_id = (
                        last_phase.get("phase", "")
                        if isinstance(last_phase, dict)
                        else getattr(last_phase, "phase", "")
                    )
                    for tname, pid in tool_map.items():
                        if str(pid) == str(last_phase_id):
                            tool_name = tname
                            break
                if not tool_name:
                    cur_phase = ctx.prior_context.get("current_phase", "")
                    for tname, pid in tool_map.items():
                        if str(pid) == str(cur_phase):
                            tool_name = tname
                            break
            except Exception as e:
                logger.info("CONFIRMATION step3 reverse-lookup error: %s", e)
            if tool_name:
                try:
                    from agent_foundation.resources.tools.registry import (
                        load_tool as _load_tool,
                    )

                    source_tool = _load_tool(tool_name)
                    if source_tool and source_tool.viewable_output_label:
                        tool.metadata["view_label"] = source_tool.viewable_output_label
                except ImportError:
                    logger.info(
                        "CONFIRMATION: load_tool not available "
                        "(agent_foundation.resources.tools.registry not installed)"
                    )
                except Exception as e:
                    logger.info(
                        "CONFIRMATION: load_tool('%s') failed: %s", tool_name, e
                    )

        # J2 (LAYER 1 final guard): ensure view_label is set whenever view is.
        # Tier-3's tool_phase_map reverse-lookup is brittle — if the map is
        # empty (e.g., SOP didn't load) or no entry matches the current/completed
        # phase, view_label stays unset and the React widget button shows
        # generic text. Default to a sensible label so the button is never blank.
        if "view" in tool.metadata and "view_label" not in tool.metadata:
            tool.metadata["view_label"] = "View"
            logger.info(
                "CONFIRMATION: J2 default view_label='View' (tool_phase_map "
                "lookup found no matching tool)"
            )

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        # Confirmation widget returns {"choice": "yes"|"no"|..., "param_overrides", "variables"}
        # Standalone (clarification-style) returns {"content": str}
        choice_value = ""
        effects: list[InferencerEffect] = []

        if isinstance(response, dict):
            if "choice" in response:
                choice_value = str(response["choice"])
                param_overrides = response.get("param_overrides")
                if param_overrides:
                    effects.append(OverrideNextActionToolArgs(param_overrides))
                variables = response.get("variables")
                if variables and isinstance(variables, dict):
                    effects.append(SetTurnVariables(variables))
            else:
                choice_value = response.get("content") or response.get("custom_text") or ""

        # Set gate flag on confirm-yes (string-keyed bridge to SOP rendering).
        if choice_value.lower() in ("yes", "proceed"):
            effects.append(ApplyContextUpdates({"_confirmation_gate_passed": True}))

        return HandlerResult(text=choice_value, effects=effects)
