# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Group selected hypotheses by their research batch.

Ported from RankEvolve's ``_group_selected_by_batch`` (in the conversational
``handlers.proposal_selection`` handler) so the hub controller can batch
selected proposals without importing the host's proposal-selection handler.
"""

from __future__ import annotations

from typing import Any


def group_selected_by_batch(
    selected: list[dict[str, Any]],
    proposals_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Group selected hypotheses by their research batch, preserving order.

    Args:
        selected: list of hypothesis dicts (full proposal dicts, not just IDs).
        proposals_data: the original proposals metadata, which has the full
            phases/batches structure.

    Returns:
        List of batch group dicts, each with 'batch_id', 'batch_label',
        'hypotheses'.
    """
    selected_ids = {h["id"] for h in selected}
    selected_map = {h["id"]: h for h in selected}
    groups: list[dict[str, Any]] = []
    for phase in proposals_data.get("phases", []):
        for batch in phase.get("batches", []):
            batch_hyps = [
                selected_map[hid]
                for hid in batch.get("hypothesis_ids", [])
                if hid in selected_ids
            ]
            if batch_hyps:
                groups.append(
                    {
                        "batch_id": batch.get("id", ""),
                        "batch_label": batch.get("label", ""),
                        "hypotheses": batch_hyps,
                    }
                )
    # Hypotheses not in any batch (orphans) get an "Ungrouped" group.
    assigned = {h["id"] for g in groups for h in g["hypotheses"]}
    orphans = [h for h in selected if h["id"] not in assigned]
    if orphans:
        groups.append(
            {
                "batch_id": "misc",
                "batch_label": "Ungrouped",
                "hypotheses": orphans,
            }
        )
    return groups
