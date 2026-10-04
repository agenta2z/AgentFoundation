# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure-function verdict computation for Implementation Hub submissions.

Lifts the synth's verdict math (originally at
``_runtime/.../sessions/<sid>/_synth/synthesize_session.py:3503-3662``) into
a reusable, importable module. Two callers:

  1. ``submissions_service.get_submissions`` overlays per-row
     ``verdict``/``verdictLabel``/``deltaPct``/``comparisonEpoch``/``stability``/
     ``stabilityCov``/``baselineSubmissionId`` against a chosen baseline at
     read time. Persisted values become advisory cache.
  2. ``learnings_generator._verdict_matrix_row`` delegates here so the
     drawer's matrix and the Hub chips can never disagree.

Math is deterministic, LLM-free, and pure. The round-trip parity test
(``test_verdict_computer.py``) guards against threshold drift.
"""

from __future__ import annotations

from typing import Any


# ────────────────────────────────────────────────────────────────────────────
# Lifted verbatim from _synth/synthesize_session.py:3503-3543
# ────────────────────────────────────────────────────────────────────────────


def ndcg_at_or_below(rows: list[dict], target_ep: int) -> float | None:
    """Look up NDCG@10 at the largest epoch ≤ target_ep (handles missing epochs).

    Lifted from synthesize_session.py:_ndcg_at_or_below.
    """
    best_n = None
    for r in rows:
        ep = r.get("epoch", -1)
        if isinstance(ep, int) and ep <= target_ep:
            best_n = r.get("ndcg10")
    return best_n


def stability_classify(rows: list[dict]) -> tuple[str, float | None]:
    """Return ``(stability, cov)`` per the 4-state taxonomy:
    ``diverged | insufficient | stable | unstable``.

    'stable' requires ``CoV < 0.005`` AND ``|slope| < 1e-4`` across last K epochs.
    Lifted from synthesize_session.py:_stability_classify.

    Stability is INTRINSIC to a submission's own trajectory — independent of
    the baseline. Callers can cache the result per-submission.
    """
    n = len(rows)
    if n < 10:
        return "insufficient", None
    K = max(3, min(10, n // 2))
    last_window = [r["ndcg10"] for r in rows[-K:]]
    peak = max(r["ndcg10"] for r in rows)
    last3 = [r["ndcg10"] for r in rows[-3:]]
    if peak > 0 and all(x < 0.5 * peak for x in last3):
        return "diverged", None
    mean_n = sum(last_window) / len(last_window)
    if mean_n == 0:
        return "unstable", None
    variance = sum((x - mean_n) ** 2 for x in last_window) / len(last_window)
    stddev = variance**0.5
    cov = stddev / mean_n
    nW = len(last_window)
    x_mean = sum(range(nW)) / nW
    y_mean = mean_n
    num = sum((i - x_mean) * (last_window[i] - y_mean) for i in range(nW))
    denom = sum((i - x_mean) ** 2 for i in range(nW))
    slope = num / denom if denom else 0.0
    if cov < 0.005 and abs(slope) < 1e-4:
        return "stable", round(cov, 5)
    if slope < -2e-4:
        return "unstable", round(cov, 5)
    return ("stable" if cov < 0.005 else "unstable"), round(cov, 5)


def compute_verdict_raw(
    *,
    is_baseline: bool,
    exp_status: str,
    exp_trajectory: list[dict],
    baseline_id: str | None,
    baseline_trajectory: list[dict] | None,
) -> dict[str, Any]:
    """Compute win/loss verdict per the synth's decision tree.

    Returns dict with keys: ``verdict``, ``verdictLabel``, ``deltaPct``,
    ``comparisonEpoch``, ``stability``, ``stabilityCov``, ``baselineSubmissionId``.

    Lifted from synthesize_session.py:compute_verdict (lines 3546-3662).
    """
    if is_baseline:
        return {
            "verdict": "baseline",
            "verdictLabel": "baseline",
            "deltaPct": None,
            "comparisonEpoch": None,
            "stability": None,
            "stabilityCov": None,
            "baselineSubmissionId": None,
        }
    if not exp_trajectory or not baseline_trajectory:
        return {
            "verdict": "incomparable",
            "verdictLabel": "incomparable (no trajectory)",
            "deltaPct": None,
            "comparisonEpoch": None,
            "stability": "insufficient",
            "stabilityCov": None,
            "baselineSubmissionId": baseline_id,
        }

    exp_final_ep = max(r["epoch"] for r in exp_trajectory)
    base_final_ep = max(r["epoch"] for r in baseline_trajectory)
    fair_ep = min(exp_final_ep, base_final_ep)

    stability, stability_cov = stability_classify(exp_trajectory)

    if exp_final_ep < 10:
        return {
            "verdict": "incomparable",
            "verdictLabel": f"incomparable (only {exp_final_ep + 1}ep)",
            "deltaPct": None,
            "comparisonEpoch": fair_ep,
            "stability": "insufficient",
            "stabilityCov": stability_cov,
            "baselineSubmissionId": baseline_id,
        }

    exp_n = ndcg_at_or_below(exp_trajectory, fair_ep)
    base_n = ndcg_at_or_below(baseline_trajectory, fair_ep)
    if exp_n is None or base_n is None or base_n == 0:
        return {
            "verdict": "incomparable",
            "verdictLabel": "incomparable (missing data at fair epoch)",
            "deltaPct": None,
            "comparisonEpoch": fair_ep,
            "stability": stability,
            "stabilityCov": stability_cov,
            "baselineSubmissionId": baseline_id,
        }

    delta_pct = round((exp_n - base_n) / base_n * 100, 2)
    abs_pct = abs(delta_pct)

    if exp_status == "early_killed":
        if abs_pct < 0.5:
            verdict = "early_kill"
            label = f"early_kill (only {exp_final_ep + 1}ep)"
        elif delta_pct > 0:
            verdict = "possible_win"
            label = f"possible win {abs_pct:.1f}% (early-killed @{exp_final_ep + 1}ep)"
        else:
            verdict = "possible_loss"
            label = f"possible loss {abs_pct:.1f}% (early-killed @{exp_final_ep + 1}ep)"
    elif stability == "diverged":
        verdict = "diverged"
        label = f"diverged ({delta_pct:+.1f}%)"
    elif abs_pct < 0.5:
        verdict = "neutral"
        label = f"tied ({delta_pct:+.1f}%)"
    elif abs_pct >= 25:
        if delta_pct < 0:
            verdict = "catastrophic_loss"
            label = f"catastrophic loss {abs_pct:.1f}%"
        else:
            verdict = "strong_win"
            label = f"strong win {abs_pct:.1f}%"
    elif abs_pct >= 5:
        if delta_pct > 0:
            verdict = "strong_win"
            label = f"strong win {abs_pct:.1f}% {stability}"
        else:
            verdict = "strong_loss"
            label = f"strong loss {abs_pct:.1f}% {stability}"
    else:
        direction = "win" if delta_pct > 0 else "loss"
        if stability == "stable":
            verdict = direction
        else:
            verdict = f"possible_{direction}"
        label = f"{direction} {abs_pct:.1f}% {stability}"

    return {
        "verdict": verdict,
        "verdictLabel": label,
        "deltaPct": delta_pct,
        "comparisonEpoch": fair_ep,
        "stability": stability,
        "stabilityCov": stability_cov,
        "baselineSubmissionId": baseline_id,
    }


# ────────────────────────────────────────────────────────────────────────────
# Submission-shape adapter + resolver
# ────────────────────────────────────────────────────────────────────────────


_VERDICT_FIELDS: tuple[str, ...] = (
    "verdict",
    "verdictLabel",
    "deltaPct",
    "comparisonEpoch",
    "stability",
    "stabilityCov",
    "baselineSubmissionId",
)


def compute_verdict_for_row(
    submission: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute verdict fields for a single submission row vs a baseline row.

    Both arguments are submission dicts in the on-disk shape (camelCase
    ``epochTrajectory``/``status``/``id`` etc.). When ``baseline is None``
    the row is treated as having no baseline available.

    Returns the 7-field verdict dict (same shape ``compute_verdict_raw``
    returns). The caller is responsible for merging it onto the row.
    """
    if baseline is not None and submission.get("id") == baseline.get("id"):
        return compute_verdict_raw(
            is_baseline=True,
            exp_status=submission.get("status", ""),
            exp_trajectory=[],
            baseline_id=None,
            baseline_trajectory=None,
        )
    return compute_verdict_raw(
        is_baseline=bool(submission.get("isBaseline")) and baseline is None,
        exp_status=submission.get("status", "") or "",
        exp_trajectory=submission.get("epochTrajectory") or [],
        baseline_id=baseline.get("id") if baseline else None,
        baseline_trajectory=(baseline.get("epochTrajectory") if baseline else None),
    )


def resolve_baseline(
    submissions: list[dict[str, Any]],
    *,
    explicit_baseline_id: str | None = None,
    choice_baseline_id: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Pick the active baseline submission per the documented resolution order.

    Order:
      1. ``explicit_baseline_id`` (e.g. ``?baseline_id=`` query param).
      2. ``choice_baseline_id`` (e.g. ``baseline_choice.json`` sidecar).
      3. Among rows with ``isBaseline=True``: highest ``finalMetrics.ndcg_10``;
         tie-break by latest ``runFinishedAt``; final tie-break by ``id``
         lexicographic.
      4. The first row with ``isBaseline=True`` (today's `learnings_generator:280`
         behavior — kept as last-resort fallback for full back-compat).
      5. ``None``.

    Returns ``(submission_or_none, source)`` where ``source`` is one of:
      ``"query" | "choice_file" | "is_baseline_resolver" | "is_baseline_first" | "none"``.
    """
    by_id = {s.get("id"): s for s in submissions if s.get("id")}

    if explicit_baseline_id and explicit_baseline_id in by_id:
        return by_id[explicit_baseline_id], "query"

    if choice_baseline_id and choice_baseline_id in by_id:
        return by_id[choice_baseline_id], "choice_file"

    flagged = [s for s in submissions if s.get("isBaseline")]
    if not flagged:
        return None, "none"

    if len(flagged) == 1:
        return flagged[0], "is_baseline_first"

    def _sort_key(s: dict[str, Any]) -> tuple:
        ndcg = (s.get("finalMetrics") or {}).get("ndcg_10") or 0.0
        finished = s.get("runFinishedAt") or 0
        sid = s.get("id") or ""
        # Higher ndcg first, then later finished, then id ascending. We sort
        # ascending on the negated numerics so the first element wins.
        return (-ndcg, -finished, sid)

    return sorted(flagged, key=_sort_key)[0], "is_baseline_resolver"


def overlay_verdicts(
    submissions: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return a NEW list where each row has its verdict fields overlaid from
    ``compute_verdict_for_row(row, baseline)``. Persisted values for the seven
    overlay fields are replaced; all other fields pass through.

    Does not mutate the input. If ``baseline`` is None, each row is
    overlaid as if it had no baseline (every row gets verdict="unknown" via
    the baseline-absent branch — actually we leave the persisted values
    alone in that case so today's UI continues to work).
    """
    if baseline is None:
        # No baseline picked — leave persisted advisory values intact.
        # Cheap copy so callers can safely mutate the result.
        return [dict(s) for s in submissions]

    out: list[dict[str, Any]] = []
    for row in submissions:
        new_row = dict(row)
        verdict = compute_verdict_for_row(row, baseline)
        for k in _VERDICT_FIELDS:
            new_row[k] = verdict[k]
        out.append(new_row)
    return out
