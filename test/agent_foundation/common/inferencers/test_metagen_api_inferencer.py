# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Unit tests for MetagenApiInferencer (mock-based, no API keys needed).

Tests cover initialization, set_messages, sync/async inference, and streaming.
All external API calls are mocked.
"""

import unittest
from unittest.mock import AsyncMock, patch

try:
    import metagen  # noqa: F401 — external Meta SDK

    _HAS_METAGEN = True
except ImportError:
    _HAS_METAGEN = False

_PATCH_PREFIX = "agent_foundation.apis.metagen"


def _make_inferencer(**kwargs):
    """Create a MetagenApiInferencer with defaults that skip key resolution."""
    from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
        MetagenApiInferencer,
    )

    kwargs.setdefault("secret_key", "mg-api-test-key")
    kwargs.setdefault("model_id", "test-model")
    return MetagenApiInferencer(**kwargs)


async def _async_gen(*chunks):
    """Helper: create an async generator yielding the given chunks."""
    for chunk in chunks:
        yield chunk


# ==========================================================================
# Init Tests
# ==========================================================================


@unittest.skipUnless(_HAS_METAGEN, "metagen SDK not installed")
class MetagenApiInferencerInitTest(unittest.TestCase):
    """Test initialization of MetagenApiInferencer."""

    @patch(f"{_PATCH_PREFIX}.get_optimal_key_for_model", return_value="mg-api-resolved")
    def test_default_attributes(self, _mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer()
        self.assertEqual(inf.system_prompt, "")
        self.assertEqual(inf.max_tokens, 4096)
        self.assertEqual(inf.temperature, 0.7)
        self.assertIsNone(inf._messages_override)

    @patch(f"{_PATCH_PREFIX}.get_optimal_key_for_model", return_value="mg-api-resolved")
    def test_default_model_id(self, _mock_key):
        from agent_foundation.apis.metagen import MetaGenModels
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer()
        self.assertEqual(inf.model_id, MetaGenModels.CLAUDE_4_6_OPUS)

    @patch(f"{_PATCH_PREFIX}.get_optimal_key_for_model", return_value="mg-api-resolved")
    def test_explicit_model_id_preserved(self, _mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer(model_id="custom-model")
        self.assertEqual(inf.model_id, "custom-model")

    @patch(
        f"{_PATCH_PREFIX}.get_optimal_key_for_model", return_value="mg-api-resolved-key"
    )
    def test_default_secret_key_resolution(self, mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer()
        self.assertEqual(inf.secret_key, "mg-api-resolved-key")
        mock_key.assert_called_once()

    @patch(f"{_PATCH_PREFIX}.get_optimal_key_for_model")
    def test_explicit_secret_key_skips_lookup(self, mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer(secret_key="my-key")
        self.assertEqual(inf.secret_key, "my-key")
        mock_key.assert_not_called()

    @patch(
        f"{_PATCH_PREFIX}.get_optimal_key_for_model",
        side_effect=FileNotFoundError("keys.json not found"),
    )
    def test_file_not_found_raises_attribute_error(self, _mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        with self.assertRaises(AttributeError) as ctx:
            MetagenApiInferencer()
        self.assertIn("keys.json", str(ctx.exception))

    @patch(
        f"{_PATCH_PREFIX}.get_optimal_key_for_model",
        side_effect=RuntimeError("unexpected"),
    )
    def test_generic_exception_raises_attribute_error(self, _mock_key):
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        with self.assertRaises(AttributeError) as ctx:
            MetagenApiInferencer()
        self.assertIn("Unable to find default secret key", str(ctx.exception))


# ==========================================================================
# set_messages Tests
# ==========================================================================


@unittest.skipUnless(_HAS_METAGEN, "metagen SDK not installed")
class MetagenApiInferencerSetMessagesTest(unittest.TestCase):
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


@unittest.skipUnless(_HAS_METAGEN, "metagen SDK not installed")
class MetagenApiInferencerInferTest(unittest.TestCase):
    """Test sync _infer method."""

    @patch(f"{_PATCH_PREFIX}.generate_text", return_value="sync response")
    def test_infer_passes_correct_args(self, mock_gen):
        inf = _make_inferencer(model_id="m1", max_tokens=100, temperature=0.5)
        inf._infer("hello")
        mock_gen.assert_called_once_with(
            "hello",
            model="m1",
            max_new_tokens=100,
            temperature=0.5,
            api_key="mg-api-test-key",
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


# ==========================================================================
# Async _ainfer Tests
# ==========================================================================


@unittest.skipUnless(_HAS_METAGEN, "metagen SDK not installed")
class MetagenApiInferencerAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Test async _ainfer method."""

    @patch(
        f"{_PATCH_PREFIX}.generate_text_async",
        new_callable=AsyncMock,
        return_value="async response",
    )
    async def test_ainfer_passes_correct_args(self, mock_gen):
        inf = _make_inferencer(model_id="m1", max_tokens=200, temperature=0.3)
        await inf._ainfer("hello")
        mock_gen.assert_called_once_with(
            "hello",
            model="m1",
            max_new_tokens=200,
            temperature=0.3,
            metagen_key="mg-api-test-key",
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
        inf = _make_inferencer()
        result = await inf._ainfer("hello")
        self.assertIsNone(result)


# ==========================================================================
# Streaming _ainfer_streaming Tests
# ==========================================================================


@unittest.skipUnless(_HAS_METAGEN, "metagen SDK not installed")
class MetagenApiInferencerStreamingTest(unittest.IsolatedAsyncioTestCase):
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
    async def test_streaming_prepends_system_prompt(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer(system_prompt="Be helpful")
        _ = [c async for c in inf._ainfer_streaming("hello")]
        call_kwargs = mock_stream.call_args[1]
        messages = call_kwargs["prompt_or_messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Be helpful"})
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_no_duplicate_system_prompt(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer(system_prompt="Be helpful")
        inf.set_messages(
            [
                {"role": "system", "content": "Existing system prompt"},
                {"role": "user", "content": "hello"},
            ]
        )
        _ = [c async for c in inf._ainfer_streaming("ignored")]
        call_kwargs = mock_stream.call_args[1]
        messages = call_kwargs["prompt_or_messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0]["content"], "Existing system prompt")

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_no_system_prompt_when_empty(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer(system_prompt="")
        _ = [c async for c in inf._ainfer_streaming("hello")]
        call_kwargs = mock_stream.call_args[1]
        messages = call_kwargs["prompt_or_messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertEqual(len(system_msgs), 0)

    @patch(f"{_PATCH_PREFIX}.generate_text_streaming")
    async def test_streaming_passes_correct_params(self, mock_stream):
        mock_stream.return_value = _async_gen("ok")
        inf = _make_inferencer(model_id="m2", max_tokens=512, temperature=0.9)
        _ = [c async for c in inf._ainfer_streaming("hello")]
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs["model"], "m2")
        self.assertEqual(call_kwargs["max_new_tokens"], 512)
        self.assertEqual(call_kwargs["temperature"], 0.9)
        self.assertEqual(call_kwargs["metagen_key"], "mg-api-test-key")


if __name__ == "__main__":
    unittest.main()
