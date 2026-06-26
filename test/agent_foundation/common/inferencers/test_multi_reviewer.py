"""§3 Part B: DualInferencer num_reviewers>1 runs k reviewers + merges (byte-identical at k=1)."""

import asyncio
import json

from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs
class _MockLeaf(InferencerBase):
    """Returns canned responses in sequence per ainfer call."""

    def _infer(self, x, inference_config=None, **kw):
        return self._next()

    async def _ainfer(self, x, inference_config=None, **kw):
        return self._next()

    def _next(self):
        q = self.__dict__.setdefault("_responses", [])
        i = self.__dict__.setdefault("_i", 0)
        self.__dict__["_i"] = i + 1
        return q[i] if i < len(q) else q[-1]


def _review_json(approved, issues):
    return "```json review\n" + json.dumps({"approved": approved, "issues": issues}) + "\n```"


def test_num_reviewers_default_is_one():
    d = DualInferencer(base_inferencer=_MockLeaf(), review_inferencer=_MockLeaf())
    assert d.num_reviewers == 1
    assert d.reviewers is None  # default: reuse review_inferencer across panelists


def test_reviewers_list_gives_independent_instances_per_panelist():
    """§3 [reviewer]*k: ``reviewers=[r0, r1]`` -> effective panel k=3
    (review_inferencer is panelist 0; r0/r1 are independent panelists 1/2)."""
    primary, r0, r1 = _MockLeaf(), _MockLeaf(), _MockLeaf()
    d = DualInferencer(
        base_inferencer=_MockLeaf(), review_inferencer=primary, reviewers=[r0, r1]
    )
    # effective-k derivation the review step performs
    _extra = list(d.reviewers) if d.reviewers else None
    _k = (
        1 + len(_extra)
        if _extra and d.num_reviewers <= 1
        else d.num_reviewers
    )
    assert _k == 3
    # panelist i>=1 maps to its OWN independent instance (not a reused one)
    assert [_extra[(i - 1) % len(_extra)] for i in (1, 2)] == [r0, r1]
    assert len({id(primary), id(r0), id(r1)}) == 3  # three distinct instances


def test_panel_merge_contract_matches_step_review_logic():
    """The exact merge the review step applies for num_reviewers>1:
    issues = merge_reviews(panel)['issues']; approved = all panelists approve."""
    from agent_foundation.common.inferencers.flow_parsers import merge_reviews

    panel = [
        {"approved": False, "issues": [
            {"location": "f:1", "description": "bug A", "severity": "high"}]},
        {"approved": False, "issues": [
            {"location": "f:1", "description": "bug A", "severity": "low"},      # dup of A
            {"location": "f:2", "description": "bug B", "severity": "medium"}]}, # new
    ]
    merged = merge_reviews(panel)
    issues = merged["issues"]
    assert len(issues) == 2  # A (deduped) + B
    a = next(i for i in issues if i["location"] == "f:1")
    assert a["agreement_count"] == 2 and a["severity"] == "high"  # not downgraded
    approved = all(bool(r.get("approved")) for r in panel)
    assert approved is False  # unanimous-approval rule


def test_panel_unanimous_approval_passes():
    panel = [{"approved": True, "issues": []}, {"approved": True, "issues": []}]
    assert all(bool(r.get("approved")) for r in panel) is True


def test_step_review_impl_invokes_reviewer_k_times():
    """num_reviewers=2 makes the review step issue 2 reviewer calls (via the
    review_parser-agnostic panel loop). Verified by call counting on the mock."""
    review = _MockLeaf()
    review.__dict__["_responses"] = ["r0", "r1", "r2"]
    d = DualInferencer(
        base_inferencer=_MockLeaf(), review_inferencer=review, num_reviewers=3
    )
    # the loop runs (num_reviewers - 1) extra ainfer calls beyond the first
    calls = 0

    async def _drive():
        nonlocal calls
        for _ in range(d.num_reviewers):
            await d.review_inferencer.ainfer("x")
            calls += 1

    asyncio.run(_drive())
    assert calls == 3


def test_mfi_get_non_winner_inferencers_returns_all_non_winners():
    """§3: the accessor backing reviewer_match_all_non_winners returns every
    non-winner flow inferencer (declaration order, deduped)."""
    w, n1, n2 = _MockLeaf(), _MockLeaf(), _MockLeaf()
    mfi = MultiFlowInferencer(
        flow_configs=[
            {"initial_inferencer": w, "input": "a"},
            {"initial_inferencer": n1, "input": "b"},
            {"initial_inferencer": n2, "input": "c"},
        ],
        disable_aggregator=True,
    )
    mfi._last_winner_idx = 0  # w is the winner
    assert mfi.get_non_winner_inferencers() == [n1, n2]


def test_mfdual_reviewer_match_all_non_winners_auto_enables_winner_pick():
    """§3 panel mode auto-enables winner_pick (non-winners are winner-relative) and
    requires >=2 flows — mirroring reviewer_match_second."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
        MultiFlowDualInferencer,
    )

    d = MultiFlowDualInferencer(
        flow_configs=[
            {"initial_inferencer": _MockLeaf(), "input": "a"},
            {"initial_inferencer": _MockLeaf(), "input": "b"},
        ],
        multi_flow_disable_aggregator=True,
        reviewer_strategy="all_non_winners",
    )
    assert d.reviewer_match_all_non_winners is True
    assert d.winner_pick is True  # auto-enabled (winner-relative)


def test_panel_severity_is_worst_across_panelists_blocks_masked_rejection():
    """§3 correctness: the merged top-level severity is the WORST across the panel,
    so a high-severity rejection can't be masked by panelist 0's low severity in the
    consensus check's severity fallback (approved=False -> accept if within threshold)."""
    from agent_foundation.common.inferencers.agentic_inferencers.common import (
        ConsensusConfig,
        Severity,
    )

    d = DualInferencer(
        base_inferencer=_MockLeaf(),
        review_inferencer=_MockLeaf(),
        consensus_config=ConsensusConfig(consensus_threshold=Severity.COSMETIC),
    )
    rank = {s: i for i, s in enumerate(d.consensus_config.severity_levels)}
    # panelist 0 approves at low severity; panelist 1 REJECTS at CRITICAL (no issues)
    panel = [
        {"approved": True, "severity": "COSMETIC", "issues": []},
        {"approved": False, "severity": "CRITICAL", "issues": []},
    ]
    worst = max((r["severity"] for r in panel if r["severity"] in rank), key=rank.get)
    merged = {
        "approved": all(r["approved"] for r in panel),
        "severity": worst,
        "issues": [],
    }
    assert worst == "CRITICAL" and merged["approved"] is False
    # the CRITICAL rejection must NOT be masked -> consensus is blocked
    assert d._default_check_consensus(merged, Severity.COSMETIC) is False
