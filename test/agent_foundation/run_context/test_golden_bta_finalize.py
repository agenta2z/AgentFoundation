"""Characterization goldens for BTA output finalization after the retry chain gives up.

When every BTA attempt raises, ``InferencerBase`` hands the call to an external
``fallback_inferencer`` or returns a non-exception ``default_return_or_raise`` value,
and then still runs ``BTA._finalize_output`` on that substitute answer (B36). This
module pins, for sync and async, what a caller observes:

  * the returned value, the stub calls, and each fallback transition (exception type,
    message, attempt count) seen by ``on_fallback_callback``. Sync recovery re-runs
    ``BTA._infer``, which rebuilds the first attempt's graph expansion from its
    committed plan and fails on the stage's own error again (until P8 it died in
    ``_build_subgraph_spec`` with the B37 ``TypeError``: the expansion was saved but
    no breakdown was promoted). Async recovery re-raises the worker's error (U4-A);
  * the workspace tree including ``outputs/``. No BTA run produced the substitute
    answer, so there is no run summary and finalize takes the leaf path (since P6 c3):
    the answer is written as the canonical ``outputs/aggregation_report.md`` and no
    aggregator output is linked. (Until then finalize found the failed attempt's empty
    aggregator workspace and wrote no ``outputs/`` at all.)

A failing aggregator (instead of failing workers) is pinned separately because it
splits sync and async: the async aggregator node swallows the failure into a synthetic
aggregation, so the fallback never runs.
"""

import pytest

from ._golden import check_golden, Normalizer
from .test_golden_bta import CALLS, KINDS, make_bta, run, snapshot, Stub

FALLBACKS = ("external", "default")


@pytest.fixture(autouse=True)
def _fresh_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def fallback_config(fallback):
    if fallback == "external":
        return {"fallback_inferencer": Stub(response="fb")}
    return {"default_return_or_raise": "default answer"}


def failing_worker(sub_query, index):
    return Stub(response=f"w{index}", fail=True)


def run_recording_transitions(bta, kind):
    transitions = []

    def on_fallback(from_func, to_func, exception, total_attempts):
        transitions.append([type(exception).__name__, str(exception), total_attempts])

    out = run(bta, kind, on_fallback_callback=on_fallback)
    return out, transitions


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fallback", FALLBACKS)
def test_finalize_after_fallback(tmp_path, fallback, kind):
    """B36: every worker raises, so every BTA attempt raises; the substitute answer is
    returned and written as the canonical output."""
    bta = make_bta(
        tmp_path,
        worker_inferencers=failing_worker,
        max_retry=1,
        **fallback_config(fallback),
    )
    out, transitions = run_recording_transitions(bta, kind)
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(tmp_path, norm, result=out, calls=CALLS, transitions=transitions)
    check_golden(f"bta/finalize_after_fallback_{fallback}_{kind}", data)


@pytest.mark.parametrize("kind", KINDS)
def test_finalize_after_aggregator_failure(tmp_path, kind):
    """B36 via the aggregator: sync raises and falls back to ``fb``, written as the
    canonical output; async returns a synthetic aggregation of the worker answers (a
    BTA run's result, finalized from its summary) and never calls the fallback."""
    bta = make_bta(
        tmp_path,
        aggregator_inferencer=Stub(response="agg", fail=True),
        max_retry=1,
        **fallback_config("external"),
    )
    out, transitions = run_recording_transitions(bta, kind)
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(tmp_path, norm, result=out, calls=CALLS, transitions=transitions)
    check_golden(f"bta/finalize_aggregator_failure_{kind}", data)
