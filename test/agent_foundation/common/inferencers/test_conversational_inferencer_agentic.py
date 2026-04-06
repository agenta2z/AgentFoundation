

"""Unit tests for ConversationalInferencer agentic loop (mock-based).

Tests run_agentic_loop, context management, tool execution delegation,
prompt rendering, and context compression — all with mocked dependencies.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition


def _make_inferencer(**kwargs):
    """Create a ConversationalInferencer with mocked base_inferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )

    base = MagicMock()
    base.ainfer = AsyncMock(return_value="LLM says hello")
    base.system_prompt = ""
    base.set_messages = MagicMock()
    base.cache_folder = None

    kwargs.setdefault("base_inferencer", base)
    return ConversationalInferencer(**kwargs)


def _make_tool_registry():
    """Create a minimal mock tool registry."""
    return {
        "set_target_path": ToolDefinition(
            name="set_target_path",
            description="Set the target path",
            tool_type="Action",
            category="session",
            aliases=["target-path", "root"],
            parameters=[
                ParameterDef(
                    name="path", type="string", description="Path", required=True
                )
            ],
        ),
        "task": ToolDefinition(
            name="task",
            description="Run a task",
            tool_type="Action",
            category="workflow",
            aliases=[],
            parameters=[
                ParameterDef(
                    name="request",
                    type="string",
                    description="Request",
                    required=True,
                )
            ],
        ),
    }


def _make_prompt_renderer():
    """Mock prompt renderer that echoes variables."""
    renderer = MagicMock()
    renderer.render = MagicMock(
        side_effect=lambda vars: f"RENDERED: {vars.get('current_turn', {}).get('content', '')}"
    )
    renderer.template_source = "mock template source"
    return renderer


# ==========================================================================
# Pure text response (no tool calls)
# ==========================================================================


class AgenticLoopPureTextTest(unittest.IsolatedAsyncioTestCase):
    """When LLM returns pure text, loop exits in 1 iteration."""

    async def test_pure_text_returns_immediately(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf.base_inferencer.ainfer = AsyncMock(
            return_value="Just a text response, no tools."
        )

        result = await inf.run_agentic_loop("Hello")

        self.assertEqual(result.text, "Just a text response, no tools.")
        self.assertEqual(result.iterations_used, 1)
        self.assertEqual(len(result.completed_actions), 0)
        self.assertEqual(result.raw_response, "Just a text response, no tools.")

    async def test_pure_text_populates_logging_fields(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf.base_inferencer.ainfer = AsyncMock(return_value="Hello!")

        result = await inf.run_agentic_loop("Hi")

        self.assertEqual(result.last_template_source, "mock template source")
        self.assertTrue(result.last_rendered_prompt.startswith("RENDERED:"))
        self.assertIn("Hi", result.last_rendered_prompt)


# ==========================================================================
# Single tool call → tool result → pure text response
# ==========================================================================


class AgenticLoopToolCallTest(unittest.IsolatedAsyncioTestCase):
    """LLM returns a tool call, tool executor returns result, LLM responds with text."""

    async def test_single_tool_call_executes_and_loops(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )

        tool_call_response = (
            "I will set the target path.\n"
            '<tool_call>{"name": "set_target_path", "arguments": {"path": "/tmp/test"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(
            side_effect=[tool_call_response, "Target path has been set."]
        )

        tool_executor = AsyncMock(
            return_value=ToolExecutionResult(result="Target path set to: /tmp/test")
        )
        inf.tool_executor = tool_executor

        result = await inf.run_agentic_loop("Set target path to /tmp/test")

        self.assertEqual(result.text, "Target path has been set.")
        self.assertEqual(result.iterations_used, 2)
        self.assertEqual(len(result.completed_actions), 1)
        self.assertEqual(result.completed_actions[0].tool, "set_target_path")
        tool_executor.assert_called_once_with("set_target_path", {"path": "/tmp/test"})

    async def test_tool_call_adds_messages(self):
        """Verify tool results are added to messages as user messages."""
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf.set_messages([{"role": "user", "content": "initial"}])

        tool_call_response = '<tool_call>{"name": "set_target_path", "arguments": {"path": "/x"}}</tool_call>'
        inf.base_inferencer.ainfer = AsyncMock(
            side_effect=[tool_call_response, "Done."]
        )
        inf.tool_executor = AsyncMock(
            return_value=ToolExecutionResult(result="Set to /x")
        )

        await inf.run_agentic_loop("set path")

        messages = inf.get_messages()
        tool_result_msgs = [
            m for m in messages if "[Tool execution results]" in m.get("content", "")
        ]
        self.assertEqual(len(tool_result_msgs), 1)
        self.assertIn("Set to /x", tool_result_msgs[0]["content"])


# ==========================================================================
# Tool name alias resolution
# ==========================================================================


class AgenticLoopAliasTest(unittest.IsolatedAsyncioTestCase):
    """LLM uses a tool alias — resolved to canonical name before execution."""

    async def test_alias_resolved_to_canonical(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )

        tool_call_response = '<tool_call>{"name": "target-path", "arguments": {"path": "/tmp"}}</tool_call>'
        inf.base_inferencer.ainfer = AsyncMock(
            side_effect=[tool_call_response, "Done."]
        )
        inf.tool_executor = AsyncMock(return_value=ToolExecutionResult(result="ok"))

        await inf.run_agentic_loop("set path")

        inf.tool_executor.assert_called_once_with("set_target_path", {"path": "/tmp"})


# ==========================================================================
# Multiple tool calls in single response
# ==========================================================================


class AgenticLoopMultiToolTest(unittest.IsolatedAsyncioTestCase):
    """LLM returns multiple tool calls in one response."""

    async def test_multiple_tools_executed_sequentially(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )

        multi_tool_response = (
            '<tool_call>{"name": "set_target_path", "arguments": {"path": "/a"}}</tool_call>'
            '<tool_call>{"name": "task", "arguments": {"request": "build it"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(
            side_effect=[multi_tool_response, "Both done."]
        )
        inf.tool_executor = AsyncMock(
            side_effect=[
                ToolExecutionResult(result="Path set"),
                ToolExecutionResult(result="Task completed"),
            ]
        )

        result = await inf.run_agentic_loop("set path and run task")

        self.assertEqual(result.iterations_used, 2)
        self.assertEqual(len(result.completed_actions), 2)
        self.assertEqual(result.completed_actions[0].tool, "set_target_path")
        self.assertEqual(result.completed_actions[1].tool, "task")
        self.assertEqual(inf.tool_executor.call_count, 2)


# ==========================================================================
# Max iterations exhaustion
# ==========================================================================


class AgenticLoopExhaustionTest(unittest.IsolatedAsyncioTestCase):
    """Loop should stop after max_iterations even if tools keep being called."""

    async def test_max_iterations_reached(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            max_iterations=2,
        )

        tool_response = (
            '<tool_call>{"name": "task", "arguments": {"request": "more"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(return_value=tool_response)
        inf.tool_executor = AsyncMock(return_value=ToolExecutionResult(result="ok"))

        result = await inf.run_agentic_loop("go")

        self.assertEqual(result.iterations_used, 2)
        self.assertTrue(result.exhausted_max_iterations)


# ==========================================================================
# Tool result truncation
# ==========================================================================


class AgenticLoopTruncationTest(unittest.IsolatedAsyncioTestCase):
    """Large tool results are truncated to max_tool_result_chars."""

    async def test_tool_result_truncated(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            max_tool_result_chars=50,
        )

        tool_response = (
            '<tool_call>{"name": "task", "arguments": {"request": "go"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(side_effect=[tool_response, "Done."])
        inf.tool_executor = AsyncMock(
            return_value=ToolExecutionResult(result="x" * 200)
        )

        await inf.run_agentic_loop("go")

        messages = inf.get_messages()
        tool_result_msg = [
            m for m in messages if "[Tool execution results]" in m.get("content", "")
        ]
        self.assertEqual(len(tool_result_msg), 1)
        self.assertIn("truncated", tool_result_msg[0]["content"].lower())


# ==========================================================================
# Prior context and dynamic context
# ==========================================================================


class ContextManagementTest(unittest.IsolatedAsyncioTestCase):
    """Test prior_context setting and dynamic_context accumulation."""

    async def test_prior_context_passed_to_renderer(self):
        renderer = _make_prompt_renderer()
        captured_vars = {}

        def capture_render(variables):
            captured_vars.update(variables)
            return "rendered"

        renderer.render = MagicMock(side_effect=capture_render)

        inf = _make_inferencer(
            tool_registry=_make_tool_registry(), prompt_renderer=renderer
        )
        inf.set_prior_context({"target_path": "/my/path", "model": "claude-opus"})
        inf.base_inferencer.ainfer = AsyncMock(return_value="ok")

        await inf.run_agentic_loop("hello")

        self.assertEqual(captured_vars.get("target_path"), "/my/path")
        self.assertEqual(captured_vars.get("model"), "claude-opus")

    async def test_dynamic_context_accumulates_across_tool_calls(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )

        tool_response = '<tool_call>{"name": "task", "arguments": {"request": "do stuff"}}</tool_call>'
        inf.base_inferencer.ainfer = AsyncMock(side_effect=[tool_response, "Done."])
        inf.tool_executor = AsyncMock(
            return_value=ToolExecutionResult(result="Task result")
        )

        await inf.run_agentic_loop("do stuff")

        self.assertEqual(len(inf.dynamic_context.completed_actions), 1)
        self.assertEqual(inf.dynamic_context.completed_actions[0].tool, "task")

    async def test_dynamic_context_persists_across_turns(self):
        """Dynamic context from turn 1 is visible in turn 2."""
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )

        tool_response = (
            '<tool_call>{"name": "task", "arguments": {"request": "step1"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(
            side_effect=[tool_response, "Step 1 done."]
        )
        inf.tool_executor = AsyncMock(
            return_value=ToolExecutionResult(result="Step 1 complete")
        )
        await inf.run_agentic_loop("step 1")

        inf.base_inferencer.ainfer = AsyncMock(return_value="All good.")
        await inf.run_agentic_loop("status?")

        self.assertEqual(len(inf.dynamic_context.completed_actions), 1)

    async def test_reset_dynamic_context(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf._dynamic_context.add_action("task", "did something")
        self.assertEqual(len(inf.dynamic_context.completed_actions), 1)

        inf.reset_dynamic_context()
        self.assertEqual(len(inf.dynamic_context.completed_actions), 0)


# ==========================================================================
# Context compression
# ==========================================================================


class ContextCompressionTest(unittest.IsolatedAsyncioTestCase):
    """Test context_compressor is called when threshold is exceeded."""

    async def test_compressor_called_when_threshold_exceeded(self):
        compressor = AsyncMock(return_value="compressed context")
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            context_compressor=compressor,
            compression_threshold=10,
        )
        for i in range(20):
            inf._dynamic_context.add_action(
                f"tool_{i}", f"summary for action {i} which is long enough"
            )

        inf.base_inferencer.ainfer = AsyncMock(return_value="ok")
        await inf.run_agentic_loop("hello")

        compressor.assert_called_once()
        self.assertEqual(inf._dynamic_context._compressed_history, "compressed context")

    async def test_compressor_not_called_when_below_threshold(self):
        compressor = AsyncMock(return_value="compressed")
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            context_compressor=compressor,
            compression_threshold=100000,
        )
        inf.base_inferencer.ainfer = AsyncMock(return_value="ok")
        await inf.run_agentic_loop("hello")

        compressor.assert_not_called()

    async def test_no_compressor_gracefully_handles_large_context(self):
        """When context_compressor is None, should not crash with large context."""
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            context_compressor=None,
        )
        for i in range(50):
            inf._dynamic_context.add_action(f"tool_{i}", "x" * 200)

        inf.base_inferencer.ainfer = AsyncMock(return_value="ok")
        result = await inf.run_agentic_loop("hello")
        self.assertEqual(result.text, "ok")


# ==========================================================================
# Tool executor context_updates
# ==========================================================================


class ContextUpdatesTest(unittest.IsolatedAsyncioTestCase):
    """Test that ToolExecutionResult.context_updates are applied to prior_context."""

    async def test_context_updates_applied(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf.set_prior_context({"target_path": ""})

        tool_response = '<tool_call>{"name": "set_target_path", "arguments": {"path": "/new"}}</tool_call>'
        inf.base_inferencer.ainfer = AsyncMock(side_effect=[tool_response, "Done."])
        inf.tool_executor = AsyncMock(
            return_value=ToolExecutionResult(
                result="Set to /new",
                context_updates={"target_path": "/new"},
            )
        )

        await inf.run_agentic_loop("set path")

        self.assertEqual(inf.prior_context["target_path"], "/new")


# ==========================================================================
# No tool executor configured
# ==========================================================================


class NoToolExecutorTest(unittest.IsolatedAsyncioTestCase):
    """When tool_executor is None, tool calls return error message."""

    async def test_no_executor_returns_error(self):
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
        )
        inf.tool_executor = None

        tool_response = (
            '<tool_call>{"name": "task", "arguments": {"request": "go"}}</tool_call>'
        )
        inf.base_inferencer.ainfer = AsyncMock(side_effect=[tool_response, "Ok."])

        await inf.run_agentic_loop("go")

        messages = inf.get_messages()
        tool_result_msg = [
            m for m in messages if "No tool executor" in m.get("content", "")
        ]
        self.assertEqual(len(tool_result_msg), 1)


# ==========================================================================
# Message management
# ==========================================================================


class MessageManagementTest(unittest.IsolatedAsyncioTestCase):
    """Test set_messages, add_message, get_messages."""

    async def test_set_messages_copies_input(self):
        inf = _make_inferencer()
        original = [{"role": "user", "content": "hi"}]
        inf.set_messages(original)
        original.append({"role": "assistant", "content": "hello"})
        self.assertEqual(len(inf.get_messages()), 1)

    async def test_add_message_appends(self):
        inf = _make_inferencer()
        inf.set_messages([])
        inf.add_message("user", "hello")
        inf.add_message("assistant", "hi")
        self.assertEqual(len(inf.get_messages()), 2)
        self.assertEqual(inf.get_messages()[0], {"role": "user", "content": "hello"})


# ==========================================================================
# InferencerContextCompressor (mock-based)
# ==========================================================================


class InferencerContextCompressorTest(unittest.IsolatedAsyncioTestCase):
    """Test the LLM-based context compressor with mocked inferencer."""

    async def test_compressor_calls_inferencer_with_prompt(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock(
            return_value="- task: did stuff\n- knowledge: found info"
        )

        compressor = InferencerContextCompressor(inferencer=mock_inf)
        context = "- task_0: long summary 0\n- task_1: long summary 1\n" * 20
        await compressor(context, max_length=200)

        mock_inf.ainfer.assert_called_once()
        prompt_arg = mock_inf.ainfer.call_args[0][0]
        self.assertIn("Action History to Compress", prompt_arg)
        self.assertIn("200", prompt_arg)
        self.assertIn(context, prompt_arg)

    async def test_compressor_skips_if_within_budget(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock()

        compressor = InferencerContextCompressor(inferencer=mock_inf)
        short_context = "- task: did something"
        result = await compressor(short_context, max_length=1000)

        mock_inf.ainfer.assert_not_called()
        self.assertEqual(result, short_context)

    async def test_compressor_truncates_if_llm_exceeds_budget(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock(return_value="x" * 500)

        compressor = InferencerContextCompressor(inferencer=mock_inf)
        context = "y" * 600
        result = await compressor(context, max_length=200)

        self.assertLessEqual(len(result), 200)

    async def test_compressor_fallback_on_error(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock(side_effect=RuntimeError("API error"))

        compressor = InferencerContextCompressor(
            inferencer=mock_inf, fallback_on_error=True
        )
        context = "a" * 500
        result = await compressor(context, max_length=200)

        self.assertLessEqual(len(result), 200)
        self.assertIn("truncated", result)

    async def test_compressor_raises_on_error_when_no_fallback(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock(side_effect=RuntimeError("API error"))

        compressor = InferencerContextCompressor(
            inferencer=mock_inf, fallback_on_error=False
        )
        context = "a" * 500

        with self.assertRaises(RuntimeError):
            await compressor(context, max_length=200)

    async def test_compressor_integration_with_dynamic_context(self):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
            ContextBudget,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
            InferencerContextCompressor,
        )

        mock_inf = MagicMock()
        mock_inf.ainfer = AsyncMock(
            return_value="- task_0: summary\n- task_5: summary\n- task_19: summary"
        )

        compressor = InferencerContextCompressor(inferencer=mock_inf)

        # Set a small context_budget.dynamic_context_max so the compressor's
        # max_length arg is smaller than the actual context, triggering LLM call
        inf = _make_inferencer(
            tool_registry=_make_tool_registry(),
            prompt_renderer=_make_prompt_renderer(),
            context_compressor=compressor,
            compression_threshold=100,
            context_budget=ContextBudget(dynamic_context_max=200),
        )
        for i in range(20):
            inf._dynamic_context.add_action(
                f"task_{i}",
                f"Detailed summary for action {i} with extra text to make it long enough",
            )

        inf.base_inferencer.ainfer = AsyncMock(return_value="ok")
        await inf.run_agentic_loop("hello")

        mock_inf.ainfer.assert_called_once()
        self.assertNotEqual(inf._dynamic_context._compressed_history, "")
        self.assertEqual(len(inf._dynamic_context._uncompressed_actions), 0)


# ==========================================================================
# AgenticDynamicContext unit tests
# ==========================================================================


class AgenticDynamicContextTest(unittest.TestCase):
    """Test AgenticDynamicContext data class."""

    def test_add_action_and_to_text(self):
        ctx = AgenticDynamicContext()
        ctx.add_action("task", "did something")
        ctx.add_action("knowledge", "found info")

        text = ctx.to_text()
        self.assertIn("task: did something", text)
        self.assertIn("knowledge: found info", text)
        self.assertEqual(len(ctx.completed_actions), 2)

    def test_compress_and_new_actions(self):
        ctx = AgenticDynamicContext()
        ctx.add_action("task_0", "old action")
        ctx.add_action("task_1", "old action 2")
        ctx.compress("compressed old stuff")
        self.assertEqual(len(ctx._uncompressed_actions), 0)

        ctx.add_action("task_2", "new action")
        text = ctx.to_text()
        self.assertIn("compressed old stuff", text)
        self.assertIn("task_2: new action", text)

    def test_to_dict_and_from_dict(self):
        ctx = AgenticDynamicContext()
        ctx.add_action("task", "summary")
        ctx.compress("compressed text")
        ctx.add_action("task2", "summary2")

        d = ctx.to_dict()
        restored = AgenticDynamicContext.from_dict(d)

        self.assertEqual(len(restored.completed_actions), 2)
        self.assertEqual(restored._compressed_history, "compressed text")

    def test_total_chars(self):
        ctx = AgenticDynamicContext()
        self.assertEqual(ctx.total_chars(), 0)
        ctx.add_action("task", "summary")
        self.assertGreater(ctx.total_chars(), 0)


if __name__ == "__main__":
    unittest.main()
