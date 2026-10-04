"""API inferencers and ``areset_conversation`` (plan §15, S17).

They keep no vendor conversation (every call sends its whole input), so a reset
changes nothing they send: in particular no ``new_session``-style argument
reaches a provider request, which the AI Gateway leaves build from their call
keywords. Run under the round's agent branch, as ``ConversationalInferencer``
does before every round.
"""

from __future__ import annotations

from unittest.mock import patch

from agent_foundation.common.inferencers.api_inferencers.claude_api_inferencer import (
    ClaudeApiInferencer,
)
from agent_foundation.common.inferencers.api_inferencers.plugboard.plugboard_api_inferencer import (
    PlugboardApiInferencer,
)
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs
from later.unittest import TestCase


@attrs(slots=False)
class _KeywordForwardingLeaf(StreamingInferencerBase):
    """Forwards its call keywords into the provider request, as the AI Gateway
    leaves do (``payload.update(kwargs)``)."""

    requests: list = attrib(factory=list)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        raise NotImplementedError

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.requests.append({"prompt": prompt, **kwargs})
        yield "ok"


async def _drain(stream) -> str:
    return "".join([chunk async for chunk in stream])


class ApiInferencerResetTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.agent = RunContext.root(workspace=None).child("agent")

    async def test_a_reset_adds_nothing_to_forwarded_call_keywords(self) -> None:
        plain, reset = _KeywordForwardingLeaf(), _KeywordForwardingLeaf()
        await reset.areset_conversation(run_context=self.agent)
        for leaf in (plain, reset):
            await leaf.ainfer("hello", run_context=self.agent)
            await _drain(leaf.ainfer_streaming("hello", run_context=self.agent))
        self.assertEqual(reset.requests, plain.requests)
        self.assertEqual(plain.requests, [{"prompt": "hello"}] * 2)

    async def test_plugboard_requests_are_the_same_after_a_reset(self) -> None:
        requests = []

        async def generate_text_streaming(**kwargs):
            requests.append(kwargs)
            yield "ok"

        with patch(
            "agent_foundation.apis.plugboard.generate_text_streaming",
            generate_text_streaming,
        ):
            plain = PlugboardApiInferencer(model_id="test-model")
            reset = PlugboardApiInferencer(model_id="test-model")
            reset.set_messages([{"role": "user", "content": "explicit"}])
            plain.set_messages([{"role": "user", "content": "explicit"}])
            await reset.areset_conversation(run_context=self.agent)
            for leaf in (plain, reset):
                await _drain(leaf.ainfer_streaming("hello", run_context=self.agent))
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1], requests[0])

    async def test_a_conversation_free_api_inferencer_is_left_untouched(self) -> None:
        inf = ClaudeApiInferencer(secret_key="unused")
        before = dict(vars(inf))
        await inf.areset_conversation(run_context=self.agent)
        self.assertEqual(vars(inf), before)
