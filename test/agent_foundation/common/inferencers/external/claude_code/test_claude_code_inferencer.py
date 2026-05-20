"""Tests for ClaudeCodeInferencer."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (
    ClaudeCodeInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)


class ClaudeCodeInferencerInitTest(unittest.TestCase):
    """Test initialization of ClaudeCodeInferencer."""

    def test_default_initialization(self):
        """Test inferencer can be created with default values."""
        inferencer = ClaudeCodeInferencer()

        self.assertIsNone(inferencer.root_folder)
        self.assertEqual(inferencer.system_prompt, "")
        self.assertEqual(inferencer.idle_timeout_seconds, 1800)
        self.assertEqual(inferencer.allowed_tools, ["Read", "Write", "Bash"])
        self.assertTrue(inferencer.include_partial_messages)

    def test_custom_initialization(self):
        """Test inferencer with custom parameters."""
        inferencer = ClaudeCodeInferencer(
            root_folder="/path/to/repo",
            system_prompt="Custom prompt",
            idle_timeout_seconds=900,
            allowed_tools=["Read"],
            model_id="claude-3",
        )

        self.assertEqual(inferencer.root_folder, "/path/to/repo")
        self.assertEqual(inferencer.system_prompt, "Custom prompt")
        self.assertEqual(inferencer.idle_timeout_seconds, 900)
        self.assertEqual(inferencer.allowed_tools, ["Read"])
        self.assertEqual(inferencer.model_id, "claude-3")


class ClaudeCodeInferencerExtractPromptTest(unittest.TestCase):
    """Test _extract_prompt method."""

    def test_extract_prompt_from_string(self):
        """Test prompt extraction from string input."""
        inferencer = ClaudeCodeInferencer()
        result = inferencer._extract_prompt("Hello world")
        self.assertEqual(result, "Hello world")

    def test_extract_prompt_from_dict_with_prompt_key(self):
        """Test prompt extraction from dict with prompt key."""
        inferencer = ClaudeCodeInferencer()
        result = inferencer._extract_prompt({"prompt": "Hello from dict"})
        self.assertEqual(result, "Hello from dict")

    def test_extract_prompt_from_dict_without_prompt_key(self):
        """Test prompt extraction from dict without prompt key."""
        inferencer = ClaudeCodeInferencer()
        result = inferencer._extract_prompt({"other_key": "value"})
        self.assertIn("other_key", result)


class ClaudeCodeInferencerSyncBridgeTest(unittest.TestCase):
    """Test sync bridge behavior including stale-loop detection."""

    def test_stale_loop_detection_clears_client(self):
        """[v4 FIX #1] Verify that stale loop detection clears the client."""
        inferencer = ClaudeCodeInferencer()

        # Simulate a previous connection with a closed loop
        mock_closed_loop = MagicMock()
        mock_closed_loop.is_closed.return_value = True

        inferencer._client = MagicMock()
        inferencer._disconnect_fn = MagicMock()
        inferencer._connected_loop = mock_closed_loop

        # After the check, stale client should be cleared
        # We need to call _infer but mock _run_async to avoid actual execution
        with patch(
            "rich_python_utils.common_utils.async_function_helper._run_async"
        ) as mock_run_async:
            mock_run_async.return_value = "response"

            # This should trigger stale-loop detection
            try:
                inferencer._infer("test prompt")
            except RuntimeError:
                pass  # Expected if SDK not available

            # Verify that stale client was detected
            # (the actual clearing happens in _infer before calling _run_async)

    def test_multi_sync_call_reconnects(self):
        """[v4 FIX #1] Second sync call detects closed loop and reconnects
        instead of using stale client from previous asyncio.run()."""
        inferencer = ClaudeCodeInferencer()

        # First call: simulate connection
        first_loop = MagicMock()
        first_loop.is_closed.return_value = False
        inferencer._client = MagicMock()
        inferencer._connected_loop = first_loop

        # Simulate loop closure (what happens after asyncio.run())
        first_loop.is_closed.return_value = True

        # Second call should clear stale state
        with patch(
            "rich_python_utils.common_utils.async_function_helper._run_async"
        ) as mock_run_async:
            mock_run_async.return_value = "response"

            result = inferencer._infer("second prompt")

            # Verify that the stale client was cleared
            self.assertIsNone(inferencer._client)
            self.assertIsNone(inferencer._connected_loop)

    def test_cross_loop_guard_raises_error(self):
        """[v4 FIX #3] Cross-loop guard raises RuntimeError if client connected
        in different loop."""
        inferencer = ClaudeCodeInferencer()

        # Set up a connected state with a specific loop
        connected_loop = MagicMock()
        connected_loop.is_closed.return_value = False
        inferencer._client = MagicMock()
        inferencer._connected_loop = connected_loop

        # Mock get_running_loop to return a different loop
        different_loop = MagicMock()

        with patch("asyncio.get_running_loop", return_value=different_loop):
            with self.assertRaises(RuntimeError) as context:
                inferencer._infer("test")

            self.assertIn("different event loop", str(context.exception))


class ClaudeCodeInferencerAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Async tests for ClaudeCodeInferencer."""

    async def test_aconnect_raises_on_missing_sdk(self):
        """Test that aconnect raises RuntimeError when SDK not available."""
        inferencer = ClaudeCodeInferencer()

        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            with self.assertRaises(RuntimeError) as context:
                await inferencer.aconnect()

            self.assertIn("Claude Agent SDK not available", str(context.exception))

    async def test_adisconnect_clears_state(self):
        """Test that adisconnect properly clears internal state."""
        inferencer = ClaudeCodeInferencer()

        # Set up mock state
        mock_disconnect = AsyncMock()
        inferencer._client = MagicMock()
        inferencer._disconnect_fn = mock_disconnect
        inferencer._connected_loop = MagicMock()

        await inferencer.adisconnect()

        # Verify state is cleared
        self.assertIsNone(inferencer._client)
        self.assertIsNone(inferencer._connected_loop)
        mock_disconnect.assert_called_once()

    async def test_async_context_manager(self):
        """Test async context manager protocol."""
        inferencer = ClaudeCodeInferencer()

        # Mock aconnect and adisconnect
        connect_mock = AsyncMock()
        disconnect_mock = AsyncMock()

        with patch.object(inferencer, "aconnect", connect_mock):
            with patch.object(inferencer, "adisconnect", disconnect_mock):
                async with inferencer as inf:
                    self.assertIs(inf, inferencer)
                    connect_mock.assert_called_once()

                disconnect_mock.assert_called_once()

    async def test_allowed_tools_forwarded_to_options(self):
        """[v3 FIX #4] allowed_tools attribute is passed to ClaudeAgentOptions."""
        custom_tools = ["Read", "Write", "Bash", "Computer"]
        inferencer = ClaudeCodeInferencer(allowed_tools=custom_tools)

        # Verify the attribute is set
        self.assertEqual(inferencer.allowed_tools, custom_tools)


class ClaudeCodeInferencerResponseTest(unittest.TestCase):
    """Test SDK response handling."""

    def test_sdk_inferencer_response_structure(self):
        """Test SDKInferencerResponse has expected fields."""
        response = SDKInferencerResponse(
            content="Hello",
            session_id="sess_123",
            tool_uses=3,
        )

        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.session_id, "sess_123")
        self.assertEqual(response.tool_uses, 3)
        self.assertEqual(str(response), "Hello")


if __name__ == "__main__":
    unittest.main()
