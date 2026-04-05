# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Unit tests for PlugboardApiInferencer (mock-based, no API keys needed).

Tests cover initialization, set_messages, sync/async inference, and streaming.
All external API calls are mocked.
"""

import unittest
from unittest.mock import AsyncMock, patch

_PATCH_PREFIX = "agent_foundation.apis.plugboard"


def _make_inferencer(**kwargs):
    """Create a PlugboardApiInferencer with defaults."""
    from agent_foundation.common.inferencers.api_inferencers.plugboard.plugboard_api_inferencer import (
        PlugboardApiInferencer,
    )

    kwargs.setdefault("model_id", "test-model")
    return PlugboardApiInferencer(**kwargs)


async def _async_gen(*chunks):
    """Helper: create an async generator yielding the given chunks."""
    for chunk in chunks:
        yield chunk


# ==========================================================================
# Init Tests
# ==========================================================================


class PlugboardApiInferencerInitTest(unittest.TestCase):
    """Test initialization of PlugboardApiInferencer."""

    def test_default_attributes(self):
        inf = _make_inferencer()
        self.assertEqual(inf.system_prompt, "")
        self.assertEqual(inf.max_tokens, 4096)
        self.assertEqual(inf.temperature, 0.7)
        self.assertEqual(inf.pipeline, "usecase-dev-ai")
        self.assertEqual(inf.model_pipeline_overrides, {})
        self.assertIsNone(inf._messages_override)

    def test_dummy_key_set_when_no_key_provided(self):
        inf = _make_inferencer()
        self.assertEqual(inf.secret_key, "plugboard-cat-auth")

    def test_explicit_secret_key_preserved(self):
        inf = _make_inferencer(secret_key="custom-key")
        self.assertEqual(inf.secret_key, "custom-key")

    def test_custom_pipeline(self):
        inf = _make_inferencer(pipeline="my-pipeline")
        self.assertEqual(inf.pipeline, "my-pipeline")

    def test_custom_model_pipeline_overrides(self):
        overrides = {"model-a": "pipeline-a"}
        inf = _make_inferencer(model_pipeline_overrides=overrides)
        self.assertEqual(inf.model_pipeline_overrides, overrides)

    def test_custom_model_id(self):
        inf = _make_inferencer(model_id="custom-model")
        self.assertEqual(inf.model_id, "custom-model")


# ==========================================================================
# set_messages Tests
# ==========================================================================


class PlugboardApiInferencerSetMessagesTest(unittest.TestCase):
    """Test set_messages behavior."""

    def test_set_messages_stores_override(self):
        inf = _make_inferencer()
        msgs = [{"role": "user", "content": "hello"}]
        inf.set_messages(msgs)
        self.assertIs(inf._messages_override, msgs)

    def test_messages_override_initially_none(self):
        inf = _make_inferencer()
        self.assertIsNone(inf._messages_override)

    def test_set_messages_empty_list(self):
        inf = _make_inferencer()
        inf.set_messages([])
        self.assertEqual(inf._messages_override, [])


# ==========================================================================
# Sync _infer Tests
# ==========================================================================


class PlugboardApiInferencerInferTest(unittest.TestCase):
    """Test sync _infer method."""

    @patch(f"{_PATCH_PREFIX}.generate_text", return_value="sync response")
    def test_infer_passes_all_args(self, mock_gen):
        inf = _make_inferencer(
            model_id="m1",
            max_tokens=100,
            temperature=0.5,
            system_prompt="sys",
            pipeline="p1",
            model_pipeline_overrides={"k": "v"},
        )
        inf._infer("hello")
        mock_gen.assert_called_once_with(
            "hello",
            model="m1",
            max_new_tokens=100,
            temperature=0.5,
            system_prompt="sys",
            pipeline="p1",
            model_pipeline_overrides={"k": "v"},
        )

    @patch(f"{_PATCH_PREFIX}.generate_text", return_value="the response")
    def test_infer_returns_response(self, _mock_gen):
        inf = _make_inferencer()
        result = inf._infer("hello")
        self.assertEqual(result, "the response")

    @patch(f"{_PATCH_PREFIX}.generate_text", return_value="ok")
    def test_infer_forwards_extra_kwargs(self, mock_gen):
        inf = _make_inferencer()
        inf._infer("hello", foo="bar", baz=42)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs["foo"], "bar")
        self.assertEqual(kwargs["baz"], 42)

    @patch(f"{_PATCH_PREFIX}.generate_text", return_value="ok")
    def test_infer_with_custom_pipeline(self, mock_gen):
        inf = _make_inferencer(pipeline="custom-pipe")
        inf._infer("hello")
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs["pipeline"], "custom-pipe")


# ==========================================================================
# Async _ainfer Tests
# ==========================================================================


class PlugboardApiInferencerAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Test async _ainfer method."""

    @patch(
        f"{_PATCH_PREFIX}.generate_text_async",
        new_callable=AsyncMock,
        return_value="async response",
    )
    async def test_ainfer_passes_correct_args(self, mock_gen):
        inf = _make_inferencer(
            model_id="m1",
            max_tokens=200,
            temperature=0.3,
            system_prompt="sys",
            pipeline="p1",
            model_pipeline_overrides={"k": "v"},
        )
        await inf._ainfer("hello")
        mock_gen.assert_called_once_with(
            "hello",
            model="m1",
            max_new_tokens=200,
            temperature=0.3,
            system_prompt="sys",
            pipeline="p1",
            model_pipeline_overrides={"k": "v"},
        )

    @patch(
        f"{_PATCH_PREFIX}.generate_text_async",
        new_callable=AsyncMock,
        return_value="async resp",
    )
    async def test_ainfer_returns_response(self, _mock_gen):
        inf = _make_inferencer()
        result = await inf._ainfer("hello")
        self.assertEqual(result, "async resp")

    @patch(
        f"{_PATCH_PREFIX}.generate_text_async",
        new_callable=AsyncMock,
        return_value="ok",
    )
    async def test_ainfer_forwards_extra_kwargs(self, mock_gen):
        inf = _make_inferencer()
        await inf._ainfer("hello", extra="val")
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs["extra"], "val")

    @patch(
        f"{_PATCH_PREFIX}.generate_text_async",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_ainfer_handles_none_response(self, _mock_gen):
        """Plugboard _ainfer uses 'response[:200] if response else ""'.
        None is falsy so the guard works — returns None without crash."""
        inf = _make_inferencer()
        result = await inf._ainfer("hello")
        self.assertIsNone(result)


# ==========================================================================
# Streaming _ainfer_streaming Tests
# ==========================================================================


class PlugboardApiInferencerStreamingTest(unittest.IsolatedAsyncioTestCase):
    """Test _ainfer_streaming method."""

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_yields_chunks(self, mock_stream):
        mock_stream.return_value = _async_gen("chunk1", "chunk2", "chunk3")
        inf = _make_inferencer()
        chunks = [c async for c in inf._ainfer_streaming("hello")]
        self.assertEqual(chunks, ["chunk1", "chunk2", "chunk3"])

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_wraps_prompt_as_user_message(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer()
        _ = [c async for c in inf._ainfer_streaming("my prompt")]
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(
            call_kwargs["prompt_or_messages"],
            [{"role": "user", "content": "my prompt"}],
        )

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_uses_messages_override(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer()
        custom = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        inf.set_messages(custom)
        _ = [c async for c in inf._ainfer_streaming("ignored prompt")]
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs["prompt_or_messages"], custom)

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_clears_messages_override_after_use(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer()
        inf.set_messages([{"role": "user", "content": "test"}])
        _ = [c async for c in inf._ainfer_streaming("prompt")]
        self.assertIsNone(inf._messages_override)

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_passes_all_config(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer(
            model_id="m2",
            max_tokens=512,
            temperature=0.9,
            system_prompt="sys",
            pipeline="p2",
            model_pipeline_overrides={"x": "y"},
        )
        _ = [c async for c in inf._ainfer_streaming("hello")]
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs["model"], "m2")
        self.assertEqual(call_kwargs["max_new_tokens"], 512)
        self.assertEqual(call_kwargs["temperature"], 0.9)
        self.assertEqual(call_kwargs["system_prompt"], "sys")
        self.assertEqual(call_kwargs["pipeline"], "p2")
        self.assertEqual(call_kwargs["model_pipeline_overrides"], {"x": "y"})

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_does_not_forward_kwargs(self, mock_stream):
        """Documents current behavior: kwargs passed to _ainfer_streaming are
        silently dropped and NOT forwarded to generate_text_streaming.
        This differs from _infer/_ainfer which forward **_inference_args."""
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer()
        _ = [c async for c in inf._ainfer_streaming("hello", extra="val")]
        call_kwargs = mock_stream.call_args[1]
        self.assertNotIn("extra", call_kwargs)


if __name__ == "__main__":
    unittest.main()
