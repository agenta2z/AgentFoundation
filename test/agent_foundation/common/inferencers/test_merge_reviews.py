"""§3 Part B — merge_reviews deterministic union / dedup / never-downgrade."""

from agent_foundation.common.inferencers.flow_parsers import merge_reviews


def test_union_dedup_by_location_and_normalized_description():
    r1 = {"issues": [{"location": "a.py:1", "description": "Null  deref", "severity": "high"}]}
    r2 = {"issues": [{"location": "a.py:1", "description": "null deref", "severity": "low"}]}  # dup (norm)
    r3 = {"issues": [{"location": "b.py:2", "description": "Race", "severity": "medium"}]}
    out = merge_reviews([r1, r2, r3])["issues"]
    assert len(out) == 2  # the two a.py:1 entries dedup to one
    a = next(i for i in out if i["location"] == "a.py:1")
    assert a["agreement_count"] == 2
    assert a["severity"] == "high"  # never downgraded to low


def test_higher_severity_wins_regardless_of_order():
    low_first = merge_reviews([
        {"issues": [{"location": "x", "description": "d", "severity": "low"}]},
        {"issues": [{"location": "x", "description": "d", "severity": "critical"}]},
    ])["issues"][0]
    assert low_first["severity"] == "critical" and low_first["agreement_count"] == 2


def test_accepts_reviews_key_and_skips_non_dicts():
    out = merge_reviews([
        {"reviews": [{"location": "x", "description": "d", "severity": "high"}]},
        None,
        "garbage",
        {"issues": ["not a dict"]},
    ])["issues"]
    assert len(out) == 1 and out[0]["agreement_count"] == 1


def test_deterministic_first_seen_order():
    out = merge_reviews([
        {"issues": [
            {"location": "c", "description": "3", "severity": "low"},
            {"location": "a", "description": "1", "severity": "low"},
            {"location": "b", "description": "2", "severity": "low"},
        ]},
    ])["issues"]
    assert [i["location"] for i in out] == ["c", "a", "b"]


def test_empty_input():
    assert merge_reviews([]) == {"issues": []}
    assert merge_reviews(None) == {"issues": []}
