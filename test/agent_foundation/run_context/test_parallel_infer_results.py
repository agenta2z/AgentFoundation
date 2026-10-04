"""parallel_infer returns one result per input, in input order, whatever the result type."""

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    UncertifiedConcurrentUseError,
)
from attr import attrs


@attrs
class Echo(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        if inference_input == "none":
            return None
        if inference_input == "list":
            return [inference_input]
        return f"r:{inference_input}"


INPUTS = ["a", "b", "c", "d", "e"]
EXPECTED = ["r:a", "r:b", "r:c", "r:d", "r:e"]


def _root():
    return RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"))


@pytest.mark.parametrize("num_workers", [1, 2, 3, 5])
def test_str_results_one_per_input_in_order(num_workers):
    results = Echo().parallel_infer(INPUTS, num_workers=num_workers)
    assert type(results) is list
    assert results == EXPECTED


def test_str_results_under_a_run_context():
    results = Echo().parallel_infer(INPUTS, num_workers=1, run_context=_root())
    assert type(results) is list
    assert results == EXPECTED


@pytest.mark.parametrize("num_workers", [2, 3, 5])
def test_overlapping_items_of_an_uncertified_class_are_refused_under_a_host_ctx(
    num_workers,
):
    """The single-flight guard's rule (P6), applied before any item runs."""
    echo = Echo()
    with pytest.raises(UncertifiedConcurrentUseError, match="parallel_infer"):
        echo.parallel_infer(INPUTS, num_workers=num_workers, run_context=_root())


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_none_and_list_results_stay_single_items(num_workers):
    results = Echo().parallel_infer(["a", "none", "list", "b"], num_workers=num_workers)
    assert results == ["r:a", None, ["list"], "r:b"]


def test_debug_runs_every_input():
    assert Echo().parallel_infer(INPUTS, num_workers=3, debug=True) == EXPECTED
