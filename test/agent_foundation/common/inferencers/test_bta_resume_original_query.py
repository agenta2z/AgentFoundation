"""BTA resume regression — the aggregator's Original User Request survives resume.

Guards a resume-only fidelity bug: on resume, WorkGraph rebuilds the
worker+aggregator subgraph via the ``subgraph_registry`` factories (inside
``_reconstruct_graph_expansions``) BEFORE the breakdown fn runs. The fresh path
threads the original request into the aggregator inside ``_make_breakdown_fn``
(``_original_query=_inf_input``), but on resume that threading is too late —
the aggregator node has already been rebuilt by the factory. So the factories
themselves must forward the attempt's original request; otherwise the
aggregator's ``## Original User Request`` slot (and the synthetic-fallback
``Original task``) render blank on resume.

These are fast unit tests — no real ``claude`` subprocess. Each resumes a run whose
aggregator never finished and asserts the input the resumed aggregator receives
(with ``inject_upstream_artifacts_to_aggregator``, the original request itself).
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from attr import attrib, attrs


@attrs
class _MockInferencer(InferencerBase):
    """Minimal concrete InferencerBase returning a fixed response and recording
    every input it receives.

    Inlined (rather than importing the shared ``_helpers`` copy) so this target
    depends only on ``attrs`` + ``agent_foundation`` — matching the
    self-contained pattern already used by test_breakdown_then_aggregate.py.
    """

    _response: str = attrib(default="mock response")
    inputs: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.inputs.append(inference_input)
        return self._response


_SUBTASKS = ["First subtask", "Second subtask"]

# Breakdown response that yields exactly 2 workers via the json_subtasks parser
# (the same shape test_bta_orchestration.py exercises).
_BREAKDOWN_JSON = (
    "```json\n"
    + json.dumps({"subtasks": [{"description": d} for d in _SUBTASKS]})
    + "\n```"
)

_ORIGINAL_REQUEST = "suggest small improvements to the CLI script"


def _worker(sub_query, index):
    return _MockInferencer(response=f"worker {index}")


def _make_bta(ws_root: str, aggregator: _MockInferencer):
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_MockInferencer(response=_BREAKDOWN_JSON),
        worker_inferencers=_worker,
        aggregator_inferencer=aggregator,
        breakdown_format="json_subtasks",
        enable_result_save=True,
        resume_with_saved_results=True,
        max_breakdown=2,
        workspace=InferencerWorkspace(root=ws_root),
    )
    bta.name = "rq"
    return bta


def _interrupt_before_aggregation(ws_root: str) -> None:
    """Leave the run as a crash after the workers would: the aggregator's and the
    graph's results gone, the breakdown promoted (a mock breakdown declares no
    ``checkpoint_scope="parent"`` extraction, so it is written here)."""
    checkpoints = InferencerWorkspace(root=ws_root).checkpoints_dir
    for name in os.listdir(checkpoints):
        if name.startswith(("aggregator_result", "rq_result")):
            path = os.path.join(checkpoints, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    promoted = os.path.join(checkpoints, "breakdown", "decomposed_subtasks.json")
    os.makedirs(os.path.dirname(promoted), exist_ok=True)
    with open(promoted, "w", encoding="utf-8") as f:
        json.dump({"subtasks": [{"description": d} for d in _SUBTASKS]}, f)


class BtaResumeOriginalQueryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _resume(self, run):
        first = _MockInferencer(response="aggregated")
        run(_make_bta(self.tmpdir, first))
        self.assertEqual(first.inputs, [_ORIGINAL_REQUEST])
        _interrupt_before_aggregation(self.tmpdir)

        resumed = _MockInferencer(response="aggregated")
        bta = _make_bta(self.tmpdir, resumed)
        run(bta)
        return bta, resumed

    def test_the_resumed_aggregator_receives_the_original_request(self):
        bta, resumed = self._resume(lambda b: b.infer(_ORIGINAL_REQUEST))
        self.assertEqual(resumed.inputs, [_ORIGINAL_REQUEST])
        self.assertEqual(bta.breakdown_inferencer.inputs, [])

    def test_the_async_resumed_aggregator_receives_the_original_request(self):
        """research_propose resumes via ``ainfer``."""
        bta, resumed = self._resume(lambda b: asyncio.run(b.ainfer(_ORIGINAL_REQUEST)))
        self.assertEqual(resumed.inputs, [_ORIGINAL_REQUEST])
        self.assertEqual(bta.breakdown_inferencer.inputs, [])


if __name__ == "__main__":
    unittest.main()
