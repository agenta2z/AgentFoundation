"""M5 §2.4: under a RunContext, MFI does NOT mutate the definition (flow_configs /
predefined_sub_queries); the runtime inputs flow through ctx.node.call instead."""

import asyncio
import copy

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    enter_run,
    exit_run,
)


def _mfi():
    return MultiFlowInferencer(
        flow_configs=[{"input": "PLACEHOLDER_A"}, {"input": "PLACEHOLDER_B"}],
        propagate_runtime_input=True,
        disable_aggregator=True,
    )


def test_definition_unmutated_under_context():
    mfi = _mfi()
    before_flow_configs = copy.deepcopy(mfi.flow_configs)
    before_psq = copy.deepcopy(mfi.predefined_sub_queries)

    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        mfi._apply_runtime_input_propagation("RUNTIME_INPUT")
        # §2.4 invariant: the DEFINITION is provably unmutated under a context.
        assert mfi.flow_configs == before_flow_configs
        assert mfi.predefined_sub_queries == before_psq
        # ...and the runtime inputs flow through the context node instead.
        node = root._store.node("/")
        assert node.call["predefined_sub_queries"] == ["RUNTIME_INPUT", "RUNTIME_INPUT"]
    finally:
        exit_run(tok)


def test_effective_accessor_reads_from_context():
    """BTA's read-through accessor sources the runtime sub-queries from the ctx."""
    mfi = _mfi()
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        mfi._apply_runtime_input_propagation("RUNTIME_INPUT")
        effective = mfi._get_effective_predefined_sub_queries()
        assert effective == ["RUNTIME_INPUT", "RUNTIME_INPUT"]
    finally:
        exit_run(tok)


def test_legacy_path_without_context_still_mutates_byte_identical():
    """No context -> the legacy behavior is preserved exactly (back-compat)."""
    mfi = _mfi()
    # No active run context.
    mfi._apply_runtime_input_propagation("RUNTIME_INPUT")
    assert [c["input"] for c in mfi.flow_configs] == ["RUNTIME_INPUT", "RUNTIME_INPUT"]
    assert mfi.predefined_sub_queries == ["RUNTIME_INPUT", "RUNTIME_INPUT"]
