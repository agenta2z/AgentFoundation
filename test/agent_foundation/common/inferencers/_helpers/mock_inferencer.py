"""Shared MockInferencer + SequentialMockInferencer for Layer 1 tests.

These are concrete InferencerBase subclasses that return configurable responses.
Single source of truth — used by all per-inferencer test files via factories.py.
"""

from __future__ import annotations

from typing import Any, Callable, List, Union

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs
class MockInferencer(InferencerBase):
    """Concrete InferencerBase that returns configurable responses.

    Supports static strings, callables (input -> str), and lists (returned in
    order, last value used after exhaustion). Tracks _call_count and
    _received_inputs for assertion.

    This pattern is consistent with the existing
    test_breakdown_then_aggregate.py:MockInferencer and
    test_multi_flow_inferencer.py:SimpleMockInferencer.
    """

    _response: Union[str, Callable, List] = attrib(default="mock response")
    _call_count: int = attrib(default=0, init=False)
    _received_inputs: List = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        self._received_inputs.append(inference_input)
        if isinstance(self._response, list):
            idx = min(self._call_count - 1, len(self._response) - 1)
            return self._response[idx]
        if callable(self._response):
            return self._response(inference_input)
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


@attrs
class SequentialMockInferencer(InferencerBase):
    """Returns different responses per call from a fixed list.

    Use when test logic depends on call ordering (e.g., DualInferencer
    consensus loop where review iteration N has different output than
    iteration N-1).

    After exhausting the list, returns the last item indefinitely.
    """

    _responses: List = attrib(factory=list)
    _call_index: int = attrib(default=0, init=False)
    _received_inputs: List = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._received_inputs.append(inference_input)
        if not self._responses:
            return ""
        idx = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        resp = self._responses[idx]
        return resp(inference_input) if callable(resp) else resp

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)
