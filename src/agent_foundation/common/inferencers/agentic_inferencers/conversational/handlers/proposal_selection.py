# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ProposalSelectionHandler — decode + enrich for the proposal-selection widget.

AF-specific behavior (differs from RE's port):
- Accepts THREE response shapes (in preference order): ``selected_proposals``,
  ``selected``, or ``choice_indices``. RE only handles the first.
- Publishes the comma-joined selected-ids string via
  ``PublishSessionVariablesEffect`` under each of ``tool.output_vars``, using
  ``tool_type=tool.tool_type`` so the CI's ``set_session_variables`` also
  publishes the ``<tool_type>__<var>`` namespaced alias (per A1.b convention).
- Captures ``response.get("auto_implement")`` — a bool from the "Implement
  Selected Proposals Now" checkbox in the React widget — via
  ``DashboardDirectiveEffect({"auto_implement": bool})``. The
  ``DashboardCoordinator.maybe_open()`` (loop-frame, Phase I) reads the
  aggregate directive to pass to ``HubAwareToolExecutor.create_experiment_hub``.

Hub creation is LOOP-FRAME (per Effect-boundary rule §A1.1) — this handler
does NOT call ``create_experiment_hub``. The RE port did it in-handler; AF
keeps it loop-frame so the AGGREGATE seed across all tools in a compound widget
is used (not just this handler's per-tool view).

``enrich_before_send`` mutates ``tool.metadata`` to populate the widget's
``proposals`` payload from ``phase_outputs["research_proposals_data"]``, with
a fallback to parsing from workspace via
``agent_foundation.common.data_models.proposal.parser.parse_proposals``.
Also injects ``view`` (unified plan path) + ``view_label`` for the
"View Full Research" button.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    DashboardDirectiveEffect,
    PublishSessionVariablesEffect,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
    HandlerContext,
    HandlerResult,
    InferencerEffect,
)
from agent_foundation.common.ui.input_modes import (
    ChoiceOption,
    InputMode,
    InputModeConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


class ProposalSelectionHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.PROPOSAL_SELECTION

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        # Selection flows through the multiple-choice machinery (one option per
        # proposal id), while the full proposal payload rides in metadata for
        # the rich React ProposalSelectionWidget. Hosts without that widget
        # registered degrade to a plain multi-select over the same options.
        options = [
            ChoiceOption(
                label=c.label,
                value=c.value,
                description=getattr(c, "description", "") or "",
            )
            for c in tool.choices
        ]
        metadata: dict[str, Any] = {"widget_type": "proposal_selection"}
        if tool.metadata:
            metadata.update(tool.metadata)
        return InputModeConfig(
            mode=InputMode.MULTIPLE_CHOICE,
            prompt=tool.prompt,
            options=options,
            allow_custom=False,
            metadata=metadata,
            show_select_all=tool.show_select_all,
            select_all_text=tool.select_all_text or "All proposals",
        )

    async def enrich_before_send(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> None:
        """Populate proposals + view metadata for the selection widget.

        Primary: read pre-parsed ``phase_outputs["research_proposals_data"]``.
        Fallback: parse via ``parse_proposals(workspace)``.
        Also injects ``view`` (unified plan path) + ``view_label`` for the
        "View Full Research" button.
        """
        if not tool.metadata:
            tool.metadata = {}

        phase_outputs = ctx.prior_context.get("phase_outputs", {})
        if not isinstance(phase_outputs, dict):
            return

        # 1. Primary: pre-parsed proposals. Stored as a JSON string by the
        # research_propose action tool; the React widget expects a dict.
        proposals_data = phase_outputs.get("research_proposals_data")
        if proposals_data:
            if isinstance(proposals_data, str):
                try:
                    proposals_data = _json.loads(proposals_data)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "PROPOSAL_SELECTION: research_proposals_data is a "
                        "non-JSON string; widget may render empty: %s",
                        e,
                    )
            tool.metadata["proposals"] = proposals_data
        else:
            # 2. Fallback: parse from workspace path.
            workspace = phase_outputs.get("research_proposals")
            if workspace:
                try:
                    from agent_foundation.common.data_models.proposal.parser import (
                        parse_proposals,
                    )

                    data = parse_proposals(workspace)
                    if data:
                        tool.metadata["proposals"] = data.to_dict()
                except ImportError:
                    logger.info("PROPOSAL_SELECTION: parse_proposals not available")
                except (OSError, ValueError) as e:
                    logger.info(
                        "PROPOSAL_SELECTION: parse_proposals('%s') failed: %s",
                        workspace,
                        e,
                    )

        # 3. View injection: unified plan path → "View Full Research" button
        unified_plan = phase_outputs.get("unified_plan_path")
        if unified_plan:
            tool.metadata["view"] = unified_plan
            tool.metadata.setdefault("view_label", "View Full Research")

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        """Decode selected proposal ids into a comma-joined string; emit effects.

        Response-shape acceptance order:
          1. ``response["selected_proposals"]`` — list of id strings
          2. ``response["selected"]`` — list of id strings
          3. ``response["choice_indices"]`` — list of ints indexing tool.choices

        Also captures optional ``response["auto_implement"]`` (bool) into a
        ``DashboardDirectiveEffect``.
        """
        if not isinstance(response, dict):
            return HandlerResult(text=str(response))

        selected_list = _extract_selected_ids(response, tool)
        if selected_list is None:
            return HandlerResult(text="")

        joined = ",".join(str(s) for s in selected_list)
        effects: list[InferencerEffect] = []

        if tool.output_vars:
            effects.append(
                PublishSessionVariablesEffect(
                    {var: joined for var in tool.output_vars},
                    tool_type=tool.tool_type,
                )
            )

        # G12: "Implement Selected Proposals Now" checkbox. Absent → default
        # False behavior downstream; present → captured via directive effect.
        if isinstance(response.get("auto_implement"), bool):
            effects.append(
                DashboardDirectiveEffect(
                    {"auto_implement": bool(response["auto_implement"])}
                )
            )

        return HandlerResult(text=joined, effects=effects)


def _extract_selected_ids(
    response: dict[str, Any],
    tool: ConversationTool,
) -> list[Any] | None:
    """Return the selected-ids list from one of 3 response shapes, or None."""
    if isinstance(response.get("selected_proposals"), list):
        return response["selected_proposals"]
    if isinstance(response.get("selected"), list):
        return response["selected"]
    choice_indices = response.get("choice_indices")
    if isinstance(choice_indices, list) and tool.choices:
        return [
            tool.choices[i].value
            for i in choice_indices
            if isinstance(i, int) and 0 <= i < len(tool.choices)
        ]
    return None
