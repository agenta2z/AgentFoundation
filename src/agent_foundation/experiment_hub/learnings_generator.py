# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Hybrid generator for the per-session Accumulated Learnings doc.

Pipeline:
  1. ``precompute_actions(submissions, proposals, overrides)`` — DETERMINISTIC
     Python-only computation of structured aggregates (verdict matrix,
     hypothesis_rerank scores, slot-disjoint combo candidates, deprioritize
     list). Returns a dict matching the ``learnings_actions`` JSON schema
     with EMPTY rationale/reason text fields.
  2. ``regenerate_accumulated_learnings(session_dir)`` — orchestrates the full
     pipeline: load inputs, call precompute_actions, write the precomputed
     JSON to disk for the subagent to consume. The actual narrative markdown
     is written by a Claude Code subagent (out-of-process — synth orchestrator
     spawns it). For Phase 1 / Phase 2 demo this function ALSO supports a
     deterministic fallback that writes a templated markdown so the GET
     endpoint always has something to serve.

Single-writer invariant: all writes (precomputed JSON + markdown body)
go to ``<session_dir>/_learnings/`` — a NEW directory NOT touched by the
agent server. session_state.json is never mutated by this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Slot vocabulary — mirrors HYPOTHESIS_SLOTS in synthesize_session.py.
# Kept in sync manually; future work could share via a single source.
# ────────────────────────────────────────────────────────────────────────────

HYPOTHESIS_SLOTS: dict[str, list[str]] = {
    "H1": ["sequence_length"],
    "H11": ["sequence_length", "backbone"],
    "H17": ["sequence_length", "input_transform"],
    "H25": ["sequence_length", "backbone"],
    "H26": ["sequence_length", "backbone"],
    "H27": ["sequence_length", "backbone"],
    "H54": ["sequence_length", "input_transform"],
    "H56": ["sequence_length"],
    "H3": ["backbone"],
    "H14": ["backbone"],
    "H16": ["backbone"],
    "H20": ["backbone"],
    "H22": ["backbone"],
    "H23": ["backbone"],
    "H24": ["backbone"],
    "H28": ["backbone"],
    "H53": ["backbone"],
    "H2": ["embedding_conditioning"],
    "H5": ["embedding_conditioning"],
    "H6": ["embedding_conditioning"],
    "H7": ["embedding_conditioning"],
    "H8": ["embedding_conditioning"],
    "H4": ["scoring_head"],
    "H9": ["temporal_bias"],
    "H10": ["temporal_bias"],
    "H37": ["temporal_bias"],
    "H35": ["negatives_sampling"],
    "H41": ["negatives_sampling"],
    "H42": ["negatives_sampling"],
    "H43": ["negatives_sampling"],
    "H19": ["cross_domain_init"],
    "H38": ["cross_domain_init"],
    "H34": ["aux_target"],
    "H40": ["aux_target"],
    "H13": ["compressor_strategy"],
    "H15": ["backbone", "embedding_conditioning"],
    "H18": ["sequence_length", "input_transform", "embedding_conditioning"],
    "H21": ["embedding_conditioning", "scoring_head"],
    "H55": ["sequence_length", "input_transform", "embedding_conditioning"],
}

# Family map — mirrors family_for() in synthesize_session.py.
_L400_FAMILY = {"H1", "H17", "H17_Z200", "H17,H8", "H54,H8"}
_PRISM_FAMILY = {"H2_LAYERSCALE", "H5_LAYERSCALE", "H6", "H8"}
_HYBRID_FUSION_FAMILY = {"H14", "H14,H8", "H22", "H23", "H53", "H20"}
_MULTI_TOKEN_FAMILY = {"H4_BROKEN", "H4_DSPACE", "H4_FIXED", "H4_GATED_RES"}
_AUDIT_FAMILY = {"H53", "H56_BASELINE", "H56_RERUN"}
_BROKEN_ABLATION = {"H2_BROKEN", "H5_BROKEN", "H4_BROKEN"}


def family_for(combo_key: str) -> str:
    if combo_key in _BROKEN_ABLATION:
        return "broken_ablation_family"
    if combo_key in _L400_FAMILY:
        return "L=400_family"
    if combo_key in _PRISM_FAMILY:
        return "PRISM_family"
    if combo_key in _HYBRID_FUSION_FAMILY:
        return "hybrid_fusion_family"
    if combo_key in _MULTI_TOKEN_FAMILY:
        return "multi_token_family"
    if combo_key in _AUDIT_FAMILY:
        return "audit_family"
    return "default"


# ────────────────────────────────────────────────────────────────────────────
# Algorithmic precomputation
# ────────────────────────────────────────────────────────────────────────────


def _h_aggregates(submissions: list[dict], h_id: str) -> dict[str, Any]:
    """Per-H aggregates: best deltaPct, validated count, early_kill count,
    consistency, exp_count, best NDCG@10 in any combo, list of submission IDs."""
    h_subs = [s for s in submissions if h_id in (s.get("selectedItems") or [])]
    deltas = [s["deltaPct"] for s in h_subs if s.get("deltaPct") is not None]
    best_dpct = max(deltas) if deltas else 0.0
    worst_dpct = min(deltas) if deltas else 0.0
    validated = sum(
        1 for s in h_subs if s.get("verdict") in ("win", "strong_win", "possible_win")
    )
    early_kill = sum(1 for s in h_subs if s.get("status") == "early_killed")
    completed = sum(1 for s in h_subs if s.get("status") == "completed")
    failed = sum(1 for s in h_subs if s.get("status") == "error")
    if len(deltas) >= 2:
        # Consistency = 1 - normalized spread; clamped to [0, 1].
        consistency = max(0.0, 1.0 - (max(deltas) - min(deltas)) / 50.0)
    else:
        consistency = 0.5
    best_ndcg = 0.0
    best_combo = ""
    for s in h_subs:
        n = (s.get("finalMetrics") or {}).get("ndcg_10") or 0.0
        if n > best_ndcg:
            best_ndcg = n
            best_combo = s.get("comboKey", "")
    return {
        "h_id": h_id,
        "exp_count": len(h_subs),
        "completed": completed,
        "early_kill": early_kill,
        "failed": failed,
        "best_dpct": max(-50.0, min(50.0, best_dpct)),
        "worst_dpct": worst_dpct,
        "validated_in_winning_combos": validated,
        "consistency": consistency,
        "best_ndcg_10": best_ndcg,
        "best_combo": best_combo,
        "evidence_submissions": [s["id"] for s in h_subs],
    }


def _verdict_matrix_row(
    h_id: str, h_proposal: dict, agg: dict, baseline_ndcg: float
) -> dict:
    """One row of the cross-hypothesis verdict matrix (§2 of the doc)."""
    delta = (
        (agg["best_ndcg_10"] - baseline_ndcg) / baseline_ndcg * 100.0
        if baseline_ndcg and agg["best_ndcg_10"]
        else 0.0
    )
    if agg["validated_in_winning_combos"] > 0:
        confirmed = "✅ confirmed win"
    elif agg["early_kill"] > 0 and agg["completed"] == 0:
        confirmed = "⚠️ not yet validated (all early-killed)"
    elif agg["failed"] > 0 and agg["completed"] == 0:
        confirmed = "❌ regressed"
    elif agg["best_dpct"] < -2.0:
        confirmed = "❌ refuted"
    elif agg["completed"] > 0:
        confirmed = "≈ neutral"
    else:
        confirmed = "— untested"
    return {
        "id": h_id,
        "title": h_proposal.get("title", "")[:60] if h_proposal else "",
        "best_variant": agg["best_combo"] or "—",
        "best_ndcg10": round(agg["best_ndcg_10"], 4) if agg["best_ndcg_10"] else None,
        "delta_pct": round(delta, 2) if agg["best_ndcg_10"] else None,
        "confirmed": confirmed,
        "exp_count": agg["exp_count"],
    }


def _score_h(agg: dict) -> float:
    """Plan A1's scoring formula:
    score = 0.5 * best_deltaPct + 0.3 * validated - 0.5 * early_kill + 0.2 * consistency"""
    return (
        0.5 * agg["best_dpct"]
        + 0.3 * agg["validated_in_winning_combos"]
        - 0.5 * agg["early_kill"]
        + 0.2 * agg["consistency"]
    )


def _generate_combos(
    submissions: list[dict],
    h_aggregates: dict[str, dict],
    top_h_ids: list[str],
    baseline_ndcg: float,
    max_combos: int = 5,
) -> list[dict]:
    """Enumerate slot-disjoint pairs/triples among top H's; dedup against
    existing submissions; sort by predicted lift; emit top N."""
    existing_combo_keys = {s.get("comboKey", "") for s in submissions}
    candidates: list[dict] = []

    for size in (2, 3):
        for combo in combinations(top_h_ids, size):
            slots: list[str] = []
            for h in combo:
                slots.extend(HYPOTHESIS_SLOTS.get(h, []))
            if len(slots) != len(set(slots)):
                continue  # slot conflict — skip

            combo_key = ",".join(sorted(combo))
            if combo_key in existing_combo_keys:
                continue  # already tested

            # Predicted lift = sum of components' standalone deltaPct, with
            # diminishing returns factor (Plan A1).
            raw_lift = sum(h_aggregates[h]["best_dpct"] for h in combo)
            lift = max(0.0, min(5.0, raw_lift * 0.5))

            candidates.append(
                {
                    "comboKey": combo_key,
                    "selectedItems": sorted(combo),
                    "expectedNdcg10Lift": round(lift, 2),
                    "expectedNdcg10Absolute": round(
                        baseline_ndcg * (1.0 + lift / 100.0), 4
                    ),
                    "_raw_score": lift,
                }
            )

    candidates.sort(key=lambda c: c["_raw_score"], reverse=True)

    out: list[dict] = []
    for i, c in enumerate(candidates[:max_combos]):
        c.pop("_raw_score", None)
        comboId = f"REC-{i + 1}"
        c["comboId"] = comboId
        c["futureComboId"] = comboId
        c["title"] = ""  # subagent fills
        c["rationale"] = ""  # subagent fills
        c["confidence"] = "medium-high" if c["expectedNdcg10Lift"] > 2 else "medium"
        c["risk"] = ""  # subagent fills
        c["estimatedComputeHours"] = 12 if "H1" in c["selectedItems"] else 6
        c["configStatus"] = "needs_generation"  # default; subagent may flip
        c["configPathProposed"] = (
            "hstu-" + "-".join(h.lower() for h in c["selectedItems"]) + ".gin (NEW)"
        )
        c["preconditions"] = []
        c["constraintCheck"] = "PASSED"
        out.append(c)
    return out


def _flatten_proposals(proposals_data: dict) -> list[dict]:
    """Flatten ProposalSelectionData.phases[*].proposals[*] into a single list."""
    out: list[dict] = []
    for phase in proposals_data.get("phases", []):
        for p in phase.get("proposals", []):
            out.append(p)
    return out


def precompute_actions(
    submissions: list[dict],
    proposals_data: dict | None,
    overrides: dict | None = None,
    *,
    top_promote_count: int = 15,
    top_combo_seed: int = 8,
    max_combos: int = 5,
    baseline_submission_id: str | None = None,
) -> dict[str, Any]:
    """Deterministic Python pre-computation of the learnings_actions dict.

    Returns a fully-shaped dict matching the schema documented in the plan,
    with empty rationale/reason text fields (to be filled by the subagent
    narrative pass). Safe to call repeatedly; pure function of inputs.

    ``baseline_submission_id`` (optional): when provided, that submission is
    used as the active baseline for the verdict matrix and active_baseline
    summary block. When None, falls back to the documented resolver:
    ``isBaseline=True`` rows ranked by highest ``finalMetrics.ndcg_10``,
    tie-break by latest ``runFinishedAt``. This keeps Hub chips (which use
    the same resolver) and the drawer's matrix aligned for the same baseline.
    """
    # Local import keeps the lazy-import discipline used throughout this module.
    from agent_foundation.experiment_hub.verdict_computer import resolve_baseline

    proposals = _flatten_proposals(proposals_data) if proposals_data else []
    proposals_by_id: dict[str, dict] = {p["id"]: p for p in proposals if p.get("id")}

    # Default rank if proposal absent — large number to sort to bottom.
    def old_rank(h_id: str) -> int:
        return (proposals_by_id.get(h_id) or {}).get("rank", 999)

    # Per-H aggregates
    all_h_ids = sorted(
        proposals_by_id.keys()
        or set(h for s in submissions for h in (s.get("selectedItems") or [])),
        key=lambda x: int(re.sub(r"[^\d]", "", x) or "0"),
    )
    h_agg: dict[str, dict] = {h: _h_aggregates(submissions, h) for h in all_h_ids}

    # Baseline + best-non-baseline. Resolver picks the active baseline per
    # the documented order: explicit id (caller-supplied) → isBaseline-flagged
    # rows by highest NDCG → first-match. Falls back to the historical
    # 0.1865 sentinel only when no baseline can be resolved at all.
    baseline, _baseline_source = resolve_baseline(
        submissions, choice_baseline_id=baseline_submission_id
    )
    baseline_ndcg = (baseline or {}).get("finalMetrics", {}).get("ndcg_10") or 0.1865
    non_baseline = [s for s in submissions if not s.get("isBaseline")]
    best_non_baseline = max(
        (s for s in non_baseline if (s.get("finalMetrics") or {}).get("ndcg_10")),
        key=lambda s: s["finalMetrics"]["ndcg_10"],
        default=None,
    )

    # Score + sort H's; assign new ranks ONLY to top movers.
    scored = sorted(
        ((-_score_h(agg), h) for h, agg in h_agg.items() if agg["exp_count"] > 0),
    )
    new_ranks: dict[str, int] = {}
    for i, (_, h) in enumerate(scored[:top_promote_count]):
        new_ranks[h] = i + 1

    rerank: list[dict] = []
    for h_id, new_rank in new_ranks.items():
        old = old_rank(h_id)
        if new_rank == old:
            continue
        agg = h_agg[h_id]
        rerank.append(
            {
                "id": h_id,
                "oldRank": old,
                "newRank": new_rank,
                "deltaRank": new_rank - old,
                "rationale": "",  # subagent fills
                "evidenceSubmissions": agg["evidence_submissions"],
                "confidence": "high" if agg["exp_count"] >= 2 else "medium",
            }
        )
    rerank.sort(key=lambda r: r["newRank"])

    # Combo recommendations from top promoted H's.
    top_h_ids = [h for _, h in scored[:top_combo_seed] if h_agg[h]["best_dpct"] > -10.0]
    new_combos = _generate_combos(
        submissions, h_agg, top_h_ids, baseline_ndcg, max_combos
    )

    # Deprioritize: H's with consistent regression
    deprioritize: list[dict] = []
    for h_id, agg in h_agg.items():
        if agg["exp_count"] >= 2 and agg["best_dpct"] < -10.0:
            deprioritize.append(
                {
                    "id": h_id,
                    "reason": (
                        f"All {agg['exp_count']} variants regressed "
                        f"(best Δ {agg['best_dpct']:+.1f}%; family {family_for(agg['best_combo'])})"
                    ),
                }
            )

    # Verdict matrix (top 25 H's by exp_count, descending)
    matrix_rows: list[dict] = []
    for h_id, agg in sorted(
        h_agg.items(),
        key=lambda x: (-x[1]["exp_count"], int(re.sub(r"[^\d]", "", x[0]) or "0")),
    ):
        if agg["exp_count"] == 0:
            continue
        matrix_rows.append(
            _verdict_matrix_row(h_id, proposals_by_id.get(h_id), agg, baseline_ndcg)
        )

    # Family aggregates
    family_aggs: dict[str, dict] = {}
    for s in submissions:
        f = family_for(s.get("comboKey", ""))
        if f not in family_aggs:
            family_aggs[f] = {
                "count": 0,
                "best_ndcg10": 0.0,
                "best_combo": "",
                "worst_dpct": 0.0,
            }
        family_aggs[f]["count"] += 1
        n = (s.get("finalMetrics") or {}).get("ndcg_10") or 0.0
        if n > family_aggs[f]["best_ndcg10"]:
            family_aggs[f]["best_ndcg10"] = n
            family_aggs[f]["best_combo"] = s.get("comboKey", "")
        d = s.get("deltaPct") or 0.0
        if d < family_aggs[f]["worst_dpct"]:
            family_aggs[f]["worst_dpct"] = d

    strongest_family = max(
        (f for f, a in family_aggs.items() if f != "default"),
        key=lambda f: family_aggs[f]["best_ndcg10"],
        default="L=400_family",
    )
    weakest_family = min(
        (f for f, a in family_aggs.items() if f != "default" and a["count"] > 0),
        key=lambda f: family_aggs[f]["worst_dpct"],
        default="multi_token_family",
    )

    # Summary block
    completed = [s for s in submissions if s.get("status") == "completed"]
    early_killed = [s for s in submissions if s.get("status") == "early_killed"]
    failed = [s for s in submissions if s.get("status") == "error"]

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "based_on_experiments": [s["id"] for s in submissions if s.get("id")],
        "summary": {
            "totalExperiments": len(submissions),
            "convergedCount": len(completed),
            "earlyKilledCount": len(early_killed),
            "failedCount": len(failed),
            "bestNonBaseline": (
                {
                    "submissionId": best_non_baseline["id"],
                    "comboKey": best_non_baseline.get("comboKey"),
                    "ndcg10": best_non_baseline["finalMetrics"]["ndcg_10"],
                    "deltaPct": best_non_baseline.get("deltaPct"),
                }
                if best_non_baseline
                else None
            ),
            "strongestBaseline": (
                {
                    "submissionId": baseline["id"],
                    "comboKey": baseline.get("comboKey"),
                    "ndcg10": baseline.get("finalMetrics", {}).get("ndcg_10"),
                }
                if baseline
                else None
            ),
            "strongestFamily": strongest_family,
            "weakestFamily": weakest_family,
            "familyAggregates": family_aggs,
        },
        "active_baseline": (
            {
                "submissionId": baseline["id"],
                "config": (baseline.get("config") or {}).get("gin_config", ""),
                "ndcg10": baseline.get("finalMetrics", {}).get("ndcg_10"),
            }
            if baseline
            else None
        ),
        "verdict_matrix": matrix_rows,
        "hypothesisRerank": rerank,
        "newCombos": new_combos,
        "deprioritize": deprioritize,
        "openQuestions": [],  # subagent fills
        "applied_changes_log": [],
    }


# ────────────────────────────────────────────────────────────────────────────
# Loaders + atomic write
# ────────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + "_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if isinstance(data, str):
                fh.write(data)
            else:
                json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_session_inputs(
    session_dir: Path,
) -> tuple[list[dict], dict | None, dict | None]:
    """Load (submissions, proposals_data, overrides) for the session.

    submissions = first hub_*_submissions.json found in session_dir.
    proposals_data = phase_outputs.research_proposals_data parsed (JSON-encoded
                     STRING per Schema fact #5).
    overrides = proposal_overrides.json sidecar (None if absent).
    """
    # Submissions: first hub_*_submissions.json
    submissions: list[dict] = []
    for sub_path in sorted(session_dir.glob("hub_*_submissions.json")):
        data = _load_json(sub_path)
        if isinstance(data, dict):
            submissions = data.get("submissions") or []
            break

    # Proposals from session_state.json
    proposals_data: dict | None = None
    state = _load_json(session_dir / "session_state.json")
    if isinstance(state, dict):
        wc = state.get("workflow_context") or {}
        po = wc.get("phase_outputs") or {}
        raw = po.get("research_proposals_data")
        if isinstance(raw, str):
            try:
                proposals_data = json.loads(raw)
            except Exception:
                proposals_data = None
        elif isinstance(raw, dict):
            proposals_data = raw

    # Overrides sidecar (Phase 3)
    overrides = _load_json(session_dir / "proposal_overrides.json")

    return submissions, proposals_data, overrides


# ────────────────────────────────────────────────────────────────────────────
# Markdown body — deterministic fallback (used when no subagent is available)
# ────────────────────────────────────────────────────────────────────────────


def _render_template_body(actions: dict) -> str:
    """Deterministic markdown body. Used when there is no subagent available
    (e.g. server is offline). The subagent-written richer body OVERWRITES
    this; if the subagent file is present on disk, the GET endpoint serves
    that instead.
    """
    s = actions.get("summary") or {}
    best = s.get("bestNonBaseline") or {}
    base = s.get("strongestBaseline") or {}
    lines: list[str] = []
    lines.append("# Accumulated Learnings — ML-20M HSTU Round")
    lines.append("")
    lines.append(
        f"> Generated: {actions.get('generated_at', '?')} · "
        f"{s.get('totalExperiments', 0)} experiments · "
        f"{s.get('convergedCount', 0)} converged · "
        f"{s.get('earlyKilledCount', 0)} early-killed · "
        f"{s.get('failedCount', 0)} failed"
    )
    lines.append(
        f"> Active baseline: `{(actions.get('active_baseline') or {}).get('config', '?')}` "
        f"NDCG@10 = {(actions.get('active_baseline') or {}).get('ndcg10', '?')}"
    )
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    if best:
        lines.append(
            f"- **Best non-baseline**: `{best.get('comboKey')}` — "
            f"NDCG@10 = **{best.get('ndcg10')}** (Δ {best.get('deltaPct'):+.2f}%)"
        )
    if base:
        lines.append(
            f"- **Strongest baseline**: `{base.get('comboKey')}` — "
            f"NDCG@10 = {base.get('ndcg10')}"
        )
    lines.append(f"- **Strongest family**: {s.get('strongestFamily')}")
    lines.append(f"- **Weakest family**: {s.get('weakestFamily')}")
    lines.append("")
    lines.append("## 2. Cross-hypothesis verdict matrix")
    lines.append("")
    lines.append(
        "| H | Best variant | Best NDCG@10 | Δ vs baseline | Confirmed | # exp |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in actions.get("verdict_matrix", []):
        ndcg_cell = r["best_ndcg10"] if r["best_ndcg10"] is not None else "—"
        delta_cell = f"{r['delta_pct']:+.2f}%" if r["delta_pct"] is not None else "—"
        lines.append(
            f"| {r['id']} | `{r['best_variant']}` | "
            f"{ndcg_cell} | {delta_cell} | "
            f"{r['confirmed']} | {r['exp_count']} |"
        )
    lines.append("")
    lines.append("## 3. Family-level findings")
    lines.append("")
    lines.append("(Awaiting subagent narrative pass.)")
    lines.append("")
    lines.append("## 4. Cross-family insights")
    lines.append("")
    lines.append("(Awaiting subagent narrative pass.)")
    lines.append("")
    lines.append("## 5. Negative results & lessons")
    lines.append("")
    lines.append("(Awaiting subagent narrative pass.)")
    lines.append("")
    lines.append("## 6. Re-ranked priorities (narrative)")
    lines.append("")
    for r in actions.get("hypothesisRerank", []):
        arrow = "↑" if r["deltaRank"] < 0 else ("↓" if r["deltaRank"] > 0 else "→")
        lines.append(
            f"- **{r['id']}**: rank {r['oldRank']} {arrow} **{r['newRank']}** "
            f"({r.get('confidence', '?')} confidence)"
        )
    lines.append("")
    lines.append("## 7. Future combo recommendations (narrative)")
    lines.append("")
    for c in actions.get("newCombos", []):
        lines.append(
            f"- **{c['comboId']}** `{c['comboKey']}` — predicted Δ "
            f"+{c['expectedNdcg10Lift']}% (→ NDCG@10 ≈ {c['expectedNdcg10Absolute']}); "
            f"config: {c['configStatus']} (`{c['configPathProposed']}`)"
        )
    lines.append("")
    lines.append("## 8. Open research questions")
    lines.append("")
    lines.append("(Awaiting subagent narrative pass.)")
    lines.append("")
    lines.append("## 9. Architectural recommendations")
    lines.append("")
    lines.append("(Awaiting subagent narrative pass.)")
    lines.append("")
    lines.append("## 10. Machine-parsable summary")
    lines.append("")
    lines.append("```json learnings_actions")
    lines.append(json.dumps(actions, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Public API: the regenerate function
# ────────────────────────────────────────────────────────────────────────────


def _load_baseline_choice_id(session_dir: Path) -> str | None:
    """Return the ``baseline_submission_id`` from ``_learnings/baseline_choice.json``,
    or ``None`` if the file is absent / unreadable / lacks the field.

    The file is the canonical user-override channel. When absent, the
    resolver in ``verdict_computer.resolve_baseline`` falls back to
    ``isBaseline=True`` rows.
    """
    path = session_dir / "_learnings" / "baseline_choice.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    sid = data.get("baseline_submission_id")
    return sid if isinstance(sid, str) and sid else None


def regenerate_accumulated_learnings(
    session_dir: Path,
    *,
    baseline_submission_id: str | None = None,
    override_md: str | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute the Accumulated Learnings doc for a session.

    Always writes the precomputed JSON to <session_dir>/_learnings/
    learnings_actions.precomputed.json (so a subagent can consume it).

    Then EITHER:
      - If <session_dir>/_learnings/accumulated_learnings.md already exists
        AND was modified after the precomputed JSON, leaves it as-is
        (a subagent has already enriched it). We replace its trailing
        ``learnings_actions`` JSON fence with the freshly-precomputed one
        to avoid drift.
      - Else writes a deterministic templated body (the fallback).

    ``baseline_submission_id`` (optional): override the resolver. When None,
    reads ``_learnings/baseline_choice.json`` and uses its
    ``baseline_submission_id`` if present; otherwise falls back to the
    ``isBaseline=True`` resolver. This keeps the regenerate path explicit —
    nothing here auto-fires on baseline change; only the user's explicit
    "Regenerate" click invokes this function.

    Returns ``{markdownBody, actions, exists, lastModified, absPath}`` —
    the same shape the GET endpoint serves.
    """
    submissions, proposals_data, overrides = _load_session_inputs(session_dir)
    if baseline_submission_id is None:
        baseline_submission_id = _load_baseline_choice_id(session_dir)
    actions = precompute_actions(
        submissions,
        proposals_data,
        overrides,
        baseline_submission_id=baseline_submission_id,
    )

    learnings_dir = session_dir / "_learnings"
    learnings_dir.mkdir(parents=True, exist_ok=True)
    precomputed_path = learnings_dir / "learnings_actions.precomputed.json"
    md_path = learnings_dir / "accumulated_learnings.md"

    # When `target_path` is supplied, the caller is staging — write only to
    # the staged paths, not the LIVE paths. The caller is responsible for
    # the archive + atomic swap. The staged sidecar precomputed.json lands
    # alongside the staged markdown.
    is_staging = target_path is not None
    if is_staging:
        out_md_path = Path(target_path)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_precomputed_path = out_md_path.parent / precomputed_path.name
        _atomic_write(out_precomputed_path, actions)
    else:
        out_md_path = md_path
        out_precomputed_path = precomputed_path
        _atomic_write(precomputed_path, actions)

    # If the caller supplied an LLM-rendered narrative body (e.g. from the
    # /experiment-hypothesis-combos --aggregate-only refresh path), strip
    # the LLM's body from any trailing learnings_actions fence it emitted,
    # then re-attach a fence built by merging:
    #   - deterministic structural fields (id, oldRank, newRank, deltaRank,
    #     comboId, selectedItems, expectedNdcg10Lift, configStatus, …)
    #   - LLM-authored prose (rationale, title, risk, openQuestions[])
    # Deterministic owns structure; LLM owns prose. Robust against the
    # LLM accidentally paraphrasing a pass-through field.
    if override_md is not None:
        body_only = _strip_fence(override_md)
        # Detect any LLM-emitted fence; merge its prose into deterministic.
        _, llm_actions = _split_fence_local(override_md)
        final_actions = _merge_llm_prose_into_actions(actions, llm_actions)
        new_fence = (
            "```json learnings_actions\n"
            + json.dumps(final_actions, indent=2)
            + "\n```"
        )
        merged = body_only.rstrip() + "\n\n" + new_fence + "\n"
        _atomic_write(out_md_path, merged)
        return {
            "markdownBody": _strip_fence(merged),
            "actions": final_actions,
            "exists": True,
            "lastModified": out_md_path.stat().st_mtime,
            "absPath": str(out_md_path),
            "isStaged": is_staging,
        }

    # Strategy: if the markdown file already exists (subagent-enriched),
    # check whether its fence has hand-edited rationale text (subagent
    # narrative). If yes — preserve the existing fence and body verbatim
    # so we don't trample subagent text. If the existing fence has empty
    # rationale fields (deterministic stub), refresh it from precompute.
    # Body content is always preserved.
    if md_path.is_file():
        existing = md_path.read_text(encoding="utf-8")
        existing_body, existing_actions = _split_fence_local(existing)
        # If the existing actions have at least one non-empty rationale OR
        # at least one openQuestion, treat it as subagent-enriched and DO NOT
        # overwrite the fence (we'd lose the narrative). The precompute is
        # already saved at precomputed_path for any client wanting fresh data.
        subagent_filled = bool(
            existing_actions
            and (
                any(
                    r.get("rationale")
                    for r in existing_actions.get("hypothesisRerank", [])
                )
                or any(
                    c.get("rationale") for c in existing_actions.get("newCombos", [])
                )
                or existing_actions.get("openQuestions")
            )
        )
        if subagent_filled:
            # Honour the subagent's text; serve as-is.
            body = _strip_fence(existing)
            # Return the subagent's actions (which are the canonical wire payload)
            actions_to_serve = existing_actions
        else:
            # Stub fence — replace it with the freshly computed one.
            new_fence = (
                "```json learnings_actions\n" + json.dumps(actions, indent=2) + "\n```"
            )
            fence_re = re.compile(
                r"```json\s+learnings_actions\s*\n(.*?)\n```",
                re.DOTALL,
            )
            # Use a lambda to bypass re.sub's backslash-as-backref interpretation
            # (JSON contains \uXXXX sequences which would otherwise blow up).
            if fence_re.search(existing):
                updated = fence_re.sub(lambda _m: new_fence, existing, count=1)
            else:
                updated = existing.rstrip() + "\n\n" + new_fence + "\n"
            _atomic_write(md_path, updated)
            body = _strip_fence(updated)
            actions_to_serve = actions
    else:
        body_with_fence = _render_template_body(actions)
        _atomic_write(md_path, body_with_fence)
        body = _strip_fence(body_with_fence)
        actions_to_serve = actions

    return {
        "markdownBody": body,
        "actions": actions_to_serve,
        "exists": True,
        "lastModified": md_path.stat().st_mtime,
        "absPath": str(md_path),
    }


def _split_fence_local(md: str) -> tuple[str, dict | None]:
    """Lightweight local fence-split; mirrors learnings_parser.split_fence so
    learnings_generator stays parser-agnostic and avoids a circular import."""
    fence_re = re.compile(
        r"```json\s+learnings_actions\s*\n(.*?)\n```",
        re.DOTALL,
    )
    m = fence_re.search(md)
    if not m:
        return md, None
    try:
        return md[: m.start()].rstrip() + "\n", json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return md, None


def _merge_llm_prose_into_actions(
    deterministic: dict[str, Any],
    llm_actions: dict | None,
) -> dict[str, Any]:
    """Merge LLM-authored prose into the deterministic actions structure.

    Deterministic owns structural fields (id, oldRank, newRank, deltaRank,
    comboId, selectedItems, expectedNdcg10Lift, configStatus, evidence*,
    confidence, …). LLM owns prose fields:

        hypothesisRerank[*].rationale     (LLM-authored)
        newCombos[*].title                (LLM-authored)
        newCombos[*].rationale            (LLM-authored)
        newCombos[*].risk                 (LLM-authored)
        openQuestions                     (LLM-authored — deterministic leaves empty)

    Rows are matched by their stable identifiers: ``id`` for hypothesisRerank,
    ``comboId`` for newCombos. Rows that exist in deterministic but not in
    the LLM's fence keep their (empty) prose fields. Rows in the LLM's fence
    that don't appear in deterministic are dropped (deterministic is the
    canonical structure — the LLM is told not to invent rows).

    If ``llm_actions`` is None or has no enriched content, returns the
    deterministic dict unchanged.
    """
    if not isinstance(llm_actions, dict):
        return deterministic

    # Index LLM's enriched rows by their stable identifier for O(1) merge.
    llm_rerank_by_id: dict[str, dict] = {}
    for row in llm_actions.get("hypothesisRerank") or []:
        if isinstance(row, dict) and row.get("id"):
            llm_rerank_by_id[row["id"]] = row

    llm_combos_by_id: dict[str, dict] = {}
    for c in llm_actions.get("newCombos") or []:
        if isinstance(c, dict) and c.get("comboId"):
            llm_combos_by_id[c["comboId"]] = c

    llm_open_qs = llm_actions.get("openQuestions") or []

    enriched_anywhere = (
        any(r.get("rationale") for r in llm_rerank_by_id.values())
        or any(
            c.get("rationale") or c.get("title") or c.get("risk")
            for c in llm_combos_by_id.values()
        )
        or bool(llm_open_qs)
    )
    if not enriched_anywhere:
        # LLM emitted a fence but didn't fill any prose — nothing to merge.
        return deterministic

    merged = dict(deterministic)  # shallow copy; replace specific keys below

    merged_rerank: list[dict] = []
    for row in deterministic.get("hypothesisRerank") or []:
        out_row = dict(row)
        llm_row = llm_rerank_by_id.get(row.get("id"))
        if llm_row and llm_row.get("rationale"):
            out_row["rationale"] = llm_row["rationale"]
        merged_rerank.append(out_row)
    merged["hypothesisRerank"] = merged_rerank

    merged_combos: list[dict] = []
    for c in deterministic.get("newCombos") or []:
        out_c = dict(c)
        llm_c = llm_combos_by_id.get(c.get("comboId"))
        if llm_c:
            for prose_field in ("title", "rationale", "risk"):
                if llm_c.get(prose_field):
                    out_c[prose_field] = llm_c[prose_field]
        merged_combos.append(out_c)
    merged["newCombos"] = merged_combos

    if llm_open_qs:
        merged["openQuestions"] = llm_open_qs

    return merged


def _strip_fence(md: str) -> str:
    """Remove the ``` ```json learnings_actions ``` ``` block from md.
    Used so the GET endpoint serves the body separately from the actions JSON.
    """
    fence_re = re.compile(
        r"\n+## 10\..*?```json\s+learnings_actions\s*\n.*?\n```",
        re.DOTALL,
    )
    stripped = fence_re.sub("", md)
    if stripped == md:
        # Try the loose pattern in case section heading differs
        loose = re.compile(r"\n*```json\s+learnings_actions\s*\n.*?\n```", re.DOTALL)
        stripped = loose.sub("", md)
    return stripped.rstrip() + "\n"
