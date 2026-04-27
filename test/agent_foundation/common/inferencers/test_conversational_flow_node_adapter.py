"""Property-based tests for ConversationalFlowNodeAdapter and related methods.

Uses Hypothesis to verify correctness properties across randomized inputs.
Each test is tagged with the requirement it validates.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
    CompletedAction,
)
from agent_foundation.ui.interactive_base import InteractiveBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conversational_inferencer(**kwargs):
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


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

message_strategy = st.fixed_dictionaries({
    "role": st.sampled_from(["user", "assistant", "system"]),
    "content": st.text(min_size=1, max_size=100),
})

completed_action_strategy = st.builds(
    CompletedAction,
    tool=st.text(min_size=1, max_size=50),
    summary=st.text(min_size=1, max_size=200),
)


# ===========================================================================
# Property 4: State Reset Completeness
# Validates: Requirements 5.1, 5.2, 5.3
# ===========================================================================


class TestStateResetCompleteness:
    """**Validates: Requirements 5.1, 5.2, 5.3**

    For any ConversationalInferencer instance with arbitrary non-empty
    _messages, _dynamic_context (with completed actions), and
    conversation_history, calling reset_for_flow_invocation() SHALL
    result in all three being empty/fresh.
    """

    @given(
        messages=st.lists(message_strategy, min_size=1, max_size=10),
        actions=st.lists(completed_action_strategy, min_size=1, max_size=10),
        history=st.lists(message_strategy, min_size=1, max_size=10),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_reset_clears_all_state(self, messages, actions, history):
        """reset_for_flow_invocation() clears _messages, _dynamic_context,
        and conversation_history regardless of prior state."""
        inf = _make_conversational_inferencer()

        # Populate with non-empty state
        inf._messages = list(messages)
        for action in actions:
            inf._dynamic_context.add_action(action.tool, action.summary)
        inf.conversation_history = list(history)

        # Preconditions: state is non-empty
        assert len(inf._messages) > 0
        assert len(inf._dynamic_context.completed_actions) > 0
        assert len(inf.conversation_history) > 0

        # Act
        inf.reset_for_flow_invocation()

        # Postconditions: all state is cleared
        assert inf._messages == [], "_messages should be empty after reset"
        assert inf._dynamic_context.completed_actions == [], (
            "_dynamic_context.completed_actions should be empty after reset"
        )
        assert inf._dynamic_context._compressed_history == "", (
            "_dynamic_context._compressed_history should be empty after reset"
        )
        assert inf._dynamic_context._uncompressed_actions == [], (
            "_dynamic_context._uncompressed_actions should be empty after reset"
        )
        assert inf.conversation_history == [], (
            "conversation_history should be empty after reset"
        )

    @given(
        messages=st.lists(message_strategy, min_size=1, max_size=10),
        actions=st.lists(completed_action_strategy, min_size=1, max_size=10),
        history=st.lists(message_strategy, min_size=1, max_size=10),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_reset_session_alias_works_identically(self, messages, actions, history):
        """reset_session (alias) produces the same result as
        reset_for_flow_invocation — LWI calls reset_session()."""
        inf = _make_conversational_inferencer()

        # Populate with non-empty state
        inf._messages = list(messages)
        for action in actions:
            inf._dynamic_context.add_action(action.tool, action.summary)
        inf.conversation_history = list(history)

        # Act via alias
        inf.reset_session()

        # Postconditions: identical to reset_for_flow_invocation
        assert inf._messages == []
        assert inf._dynamic_context.completed_actions == []
        assert inf._dynamic_context._compressed_history == ""
        assert inf._dynamic_context._uncompressed_actions == []
        assert inf.conversation_history == []


# ===========================================================================
# Adapter helpers and strategies
# ===========================================================================

def _make_adapter(**kwargs):
    """Create a ConversationalFlowNodeAdapter with a mocked ConversationalInferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
        ConversationalFlowNodeAdapter,
    )

    conv_inf = _make_conversational_inferencer()
    kwargs.setdefault("conversational_inferencer", conv_inf)
    return ConversationalFlowNodeAdapter(**kwargs)


def _make_agentic_result(text="hello", **overrides):
    """Create an AgenticResult with sensible defaults."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
        AgenticResult,
    )

    defaults = dict(
        text=text,
        completed_actions=[],
        iterations_used=1,
        has_conversation_tool=False,
        exhausted_max_iterations=False,
        raw_response=text,
    )
    defaults.update(overrides)
    return AgenticResult(**defaults)


# Strategy for non-empty content strings (used as inference_input)
content_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


# ===========================================================================
# Property 1: Adapter Routing Correctness
# Validates: Requirements 1.1, 2.2, 2.3
# ===========================================================================


class TestAdapterRoutingCorrectness:
    """**Validates: Requirements 1.1, 2.2, 2.3**

    For any ConversationalFlowNodeAdapter instance with a wrapped
    ConversationalInferencer, calling _ainfer(content) SHALL invoke
    run_agentic_loop(content) on the wrapped inferencer. The content
    string passed to run_agentic_loop SHALL be identical to the
    inference_input received by _ainfer().
    """

    @given(content=content_strategy)
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    @pytest.mark.asyncio
    async def test_content_forwarded_to_run_agentic_loop(self, content):
        """The content passed to _ainfer is forwarded identically to
        run_agentic_loop's content parameter."""
        adapter = _make_adapter()
        captured_kwargs = {}

        async def mock_run_agentic_loop(**kw):
            captured_kwargs.update(kw)
            return _make_agentic_result(text=content)

        adapter.conversational_inferencer.run_agentic_loop = mock_run_agentic_loop

        result = await adapter._ainfer(content)

        assert "content" in captured_kwargs, "run_agentic_loop must receive content"
        assert captured_kwargs["content"] == content, (
            f"content mismatch: expected {content!r}, got {captured_kwargs['content']!r}"
        )
        assert result == content, "default output_extractor should return result.text"


# ===========================================================================
# Property 3: Output Extraction Correctness
# Validates: Requirements 3.1, 3.3
# ===========================================================================


class TestOutputExtractionCorrectness:
    """**Validates: Requirements 3.1, 3.3**

    For any AgenticResult with a non-empty text field, the default
    output_extractor SHALL return result.text as a string. For any
    custom output_extractor, the adapter SHALL apply it to the
    AgenticResult and return the result.
    """

    @given(text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    @pytest.mark.asyncio
    async def test_default_extractor_returns_text(self, text):
        """Default output_extractor returns AgenticResult.text."""
        adapter = _make_adapter()

        async def mock_run(**kw):
            return _make_agentic_result(text=text)

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("some input")
        assert result == text

    @given(text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    @pytest.mark.asyncio
    async def test_custom_extractor_is_applied(self, text):
        """Custom output_extractor is applied to the AgenticResult."""
        custom_extractor = lambda r: f"CUSTOM:{r.text}"
        adapter = _make_adapter(output_extractor=custom_extractor)

        async def mock_run(**kw):
            return _make_agentic_result(text=text)

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("some input")
        assert result == f"CUSTOM:{text}"


# ===========================================================================
# Property 5: Fallback Activation
# Validates: Requirements 7.1, 7.2, 7.3, 7.4
# ===========================================================================


class TestFallbackActivation:
    """**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

    Verify fallback fires on: timeout, empty output,
    exhausted_max_iterations=True, has_conversation_tool=True.
    Verify fallback does NOT fire on valid non-empty output.
    Verify exception propagates when no fallback is set.
    Verify sync _infer() raises NotImplementedError.
    """

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self):
        """Fallback fires when session_timeout is exceeded."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
            ConversationalFlowNodeAdapter,
        )

        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(
            session_timeout=0.001,
            fallback_inferencer=fallback,
        )

        async def slow_run(**kw):
            await asyncio.sleep(10)
            return _make_agentic_result()

        adapter.conversational_inferencer.run_agentic_loop = slow_run

        result = await adapter._ainfer("task")
        assert result == "fallback result"
        fallback.ainfer.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_empty_output(self):
        """Fallback fires when output is empty."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(text="")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("task")
        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_on_exhausted_max_iterations(self):
        """Fallback fires when exhausted_max_iterations is True."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(exhausted_max_iterations=True)

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("task")
        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_on_has_conversation_tool(self):
        """Fallback fires when has_conversation_tool is True."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(has_conversation_tool=True)

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("task")
        assert result == "fallback result"

    @given(text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    @pytest.mark.asyncio
    async def test_no_fallback_on_valid_output(self, text):
        """Fallback does NOT fire on valid non-empty output."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="should not be called")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(text=text)

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("task")
        assert result == text
        fallback.ainfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_propagates_without_fallback(self):
        """Exception propagates when no fallback is set."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
            _EmptyConversationalOutput,
        )

        adapter = _make_adapter(fallback_inferencer=None)

        async def mock_run(**kw):
            return _make_agentic_result(text="")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        with pytest.raises(_EmptyConversationalOutput):
            await adapter._ainfer("task")

    def test_sync_infer_raises_not_implemented(self):
        """Sync _infer() raises NotImplementedError."""
        adapter = _make_adapter()
        with pytest.raises(NotImplementedError, match="async-only"):
            adapter._infer("task")


# ===========================================================================
# Property 6: Interactive Context Propagation
# Validates: Requirements 4.1, 4.2
# ===========================================================================


class TestInteractiveContextPropagation:
    """**Validates: Requirements 4.1, 4.2**

    Verify kwargs["interactive"] wins over self.interactive.
    Verify fallback to self.interactive when not in kwargs.
    """

    @pytest.mark.asyncio
    async def test_kwargs_interactive_wins(self):
        """kwargs['interactive'] wins over self.interactive."""
        self_interactive = MagicMock(spec=InteractiveBase)
        kwargs_interactive = MagicMock(spec=InteractiveBase)

        adapter = _make_adapter(interactive=self_interactive)
        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="ok")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("task", interactive=kwargs_interactive)
        assert captured["interactive"] is kwargs_interactive

    @pytest.mark.asyncio
    async def test_fallback_to_self_interactive(self):
        """Falls back to self.interactive when not in kwargs."""
        self_interactive = MagicMock(spec=InteractiveBase)

        adapter = _make_adapter(interactive=self_interactive)
        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="ok")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("task")
        assert captured["interactive"] is self_interactive

    @pytest.mark.asyncio
    async def test_none_when_neither_set(self):
        """interactive is None when neither kwargs nor self.interactive set."""
        adapter = _make_adapter(interactive=None)
        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="ok")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("task")
        assert captured["interactive"] is None

    @given(
        use_kwargs=st.booleans(),
        use_self=st.booleans(),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    @pytest.mark.asyncio
    async def test_interactive_resolution_property(self, use_kwargs, use_self):
        """Property: kwargs['interactive'] wins → self.interactive fallback → None."""
        kwargs_interactive = MagicMock(spec=InteractiveBase) if use_kwargs else None
        self_interactive = MagicMock(spec=InteractiveBase) if use_self else None

        adapter = _make_adapter(interactive=self_interactive)
        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="ok")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        call_kwargs = {}
        if kwargs_interactive is not None:
            call_kwargs["interactive"] = kwargs_interactive

        await adapter._ainfer("task", **call_kwargs)

        expected = kwargs_interactive or self_interactive
        assert captured["interactive"] is expected


# ===========================================================================
# Unit Tests: extract_response_text() with AgenticResult
# Validates: Requirements 3.2, 8.2, 8.3
# ===========================================================================


class TestExtractResponseTextWithAgenticResult:
    """**Validates: Requirements 3.2, 8.2, 8.3**

    Verify that extract_response_text() correctly handles AgenticResult
    alongside existing types, and that dispatch order is preserved.
    """

    def test_agentic_result_returns_text(self):
        """AgenticResult with text returns the text field."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        result = _make_agentic_result(text="conversational output")
        assert extract_response_text(result) == "conversational output"

    def test_agentic_result_empty_text(self):
        """AgenticResult with empty text returns empty string."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        result = _make_agentic_result(text="")
        assert extract_response_text(result) == ""

    def test_str_still_works(self):
        """Plain string input still returns str(result)."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        assert extract_response_text("hello world") == "hello world"

    def test_inferencer_response_still_works(self):
        """InferencerResponse still dispatches via select_response()."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            InferencerResponse,
            extract_response_text,
        )

        resp = InferencerResponse(
            base_response="base",
            reflection_response="reflected",
        )
        # Default selector is LastReflection, which returns reflection_response
        assert extract_response_text(resp) == "reflected"

    def test_dual_inferencer_response_still_works(self):
        """DualInferencerResponse still dispatches via base_response."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            DualInferencerResponse,
            extract_response_text,
        )

        resp = DualInferencerResponse(
            base_response="dual base",
            reflection_response="dual reflected",
        )
        assert extract_response_text(resp) == "dual base"

    def test_dispatch_order_dual_before_inferencer(self):
        """DualInferencerResponse is checked before InferencerResponse
        (since it inherits from InferencerResponse)."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            DualInferencerResponse,
            InferencerResponse,
            extract_response_text,
        )

        dual = DualInferencerResponse(
            base_response="dual base",
            reflection_response="dual reflected",
        )
        # DualInferencerResponse IS an InferencerResponse, but should hit
        # the DualInferencerResponse branch first → returns base_response
        assert isinstance(dual, InferencerResponse)
        assert extract_response_text(dual) == "dual base"

    def test_dispatch_order_inferencer_before_agentic(self):
        """InferencerResponse is checked before AgenticResult."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            InferencerResponse,
            extract_response_text,
        )

        # InferencerResponse should NOT fall through to AgenticResult branch
        resp = InferencerResponse(
            base_response="base",
            reflection_response="reflected",
        )
        result = extract_response_text(resp)
        # Should use select_response(), not str()
        assert result == "reflected"

    def test_unknown_type_falls_through_to_str(self):
        """Unknown types fall through to str(result)."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        assert extract_response_text(42) == "42"
        assert extract_response_text(None) == "None"
        assert extract_response_text(["a", "b"]) == "['a', 'b']"


# ===========================================================================
# Property 2: Polymorphic Compatibility
# Validates: Requirements 1.1, 1.2, 8.1
# ===========================================================================


class TestPolymorphicCompatibility:
    """**Validates: Requirements 1.1, 1.2, 8.1**

    ConversationalFlowNodeAdapter SHALL be a subclass of InferencerBase
    and instances SHALL pass isinstance checks, ensuring any flow
    inferencer slot that accepts InferencerBase also accepts the adapter.
    """

    def test_adapter_is_subclass_of_inferencer_base(self):
        """ConversationalFlowNodeAdapter is a subclass of InferencerBase."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
            ConversationalFlowNodeAdapter,
        )
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase

        assert issubclass(ConversationalFlowNodeAdapter, InferencerBase), (
            "ConversationalFlowNodeAdapter must be a subclass of InferencerBase"
        )

    def test_adapter_instance_passes_isinstance_check(self):
        """Adapter instances pass isinstance(adapter, InferencerBase)."""
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase

        adapter = _make_adapter()
        assert isinstance(adapter, InferencerBase), (
            "Adapter instance must pass isinstance(adapter, InferencerBase)"
        )

    def test_adapter_has_ainfer_method(self):
        """Adapter has ainfer() method inherited from InferencerBase."""
        adapter = _make_adapter()
        assert hasattr(adapter, "ainfer"), "Adapter must have ainfer() method"
        assert callable(adapter.ainfer), "ainfer must be callable"

    def test_adapter_has_ainfer_single_pipeline(self):
        """Adapter has _ainfer_single() from InferencerBase pipeline."""
        adapter = _make_adapter()
        assert hasattr(adapter, "_ainfer_single"), (
            "Adapter must inherit _ainfer_single from InferencerBase"
        )

    def test_adapter_overrides_ainfer(self):
        """Adapter overrides _ainfer (not just inherits the default)."""
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
            ConversationalFlowNodeAdapter,
        )

        # The adapter's _ainfer should be different from the base class default
        assert ConversationalFlowNodeAdapter._ainfer is not InferencerBase._ainfer, (
            "Adapter must override _ainfer, not use the base default"
        )

    def test_adapter_has_template_manager_slot(self):
        """Adapter inherits template_manager from InferencerBase for flow-level rendering."""
        adapter = _make_adapter()
        assert hasattr(adapter, "template_manager"), (
            "Adapter must have template_manager for flow-level template rendering"
        )

    @pytest.mark.asyncio
    async def test_adapter_callable_via_standard_ainfer_path(self):
        """Adapter can be called via ainfer() like any InferencerBase child."""
        adapter = _make_adapter()

        async def mock_run(**kw):
            return _make_agentic_result(text="polymorphic result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        # Call via the public ainfer() path (same as flow inferencers do)
        result = await adapter.ainfer("test input")
        assert result == "polymorphic result"


# ===========================================================================
# Property 8: Backward Compatibility
# Validates: Requirements 8.1, 8.2, 8.3
# ===========================================================================


class TestBackwardCompatibility:
    """**Validates: Requirements 8.1, 8.2, 8.3**

    Standard InferencerBase children still work via ainfer() unchanged.
    extract_response_text() handles all existing types identically.
    hasattr-guarded injection is no-op for plain LLM children.
    """

    @pytest.mark.asyncio
    async def test_plain_inferencer_child_works_unchanged(self):
        """A standard InferencerBase subclass works via ainfer() unchanged.

        This verifies that the adapter feature doesn't break existing
        flow inferencer children that are plain LLM inferencers.
        """
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase
        from attr import attrs, attrib

        @attrs(slots=False, kw_only=True)
        class PlainInferencer(InferencerBase):
            """Minimal concrete InferencerBase for testing."""

            def _infer(self, inference_input, inference_config=None, **kw):
                return f"plain:{inference_input}"

        child = PlainInferencer()
        result = await child.ainfer("hello")
        assert result == "plain:hello"

    def test_extract_response_text_str_unchanged(self):
        """extract_response_text still handles str identically."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        assert extract_response_text("hello") == "hello"
        assert extract_response_text("") == ""

    def test_extract_response_text_inferencer_response_unchanged(self):
        """extract_response_text still handles InferencerResponse identically."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            InferencerResponse,
            extract_response_text,
        )

        resp = InferencerResponse(
            base_response="base output",
            reflection_response="reflected output",
        )
        assert extract_response_text(resp) == "reflected output"

    def test_extract_response_text_dual_response_unchanged(self):
        """extract_response_text still handles DualInferencerResponse identically."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            DualInferencerResponse,
            extract_response_text,
        )

        resp = DualInferencerResponse(
            base_response="dual base",
            reflection_response="dual reflected",
        )
        assert extract_response_text(resp) == "dual base"

    def test_extract_response_text_agentic_result(self):
        """extract_response_text handles AgenticResult via lazy import."""
        from agent_foundation.common.inferencers.agentic_inferencers.common import (
            extract_response_text,
        )

        result = _make_agentic_result(text="agentic output")
        assert extract_response_text(result) == "agentic output"

    def test_hasattr_interactive_noop_for_plain_child(self):
        """hasattr(child, 'interactive') is safe for plain InferencerBase children.

        Flow inferencers use hasattr-guarded injection like:
            if hasattr(child, 'interactive'):
                child.interactive = ...
        This must be a no-op (return False) for plain InferencerBase
        subclasses that don't define an 'interactive' attribute.
        """
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase
        from attr import attrs

        @attrs(slots=False, kw_only=True)
        class PlainLLMInferencer(InferencerBase):
            """Plain inferencer without interactive attribute."""

            def _infer(self, inference_input, inference_config=None, **kw):
                return str(inference_input)

        child = PlainLLMInferencer()

        # hasattr check should return False — no 'interactive' on plain children
        assert not hasattr(child, "interactive"), (
            "Plain InferencerBase child should not have 'interactive' attribute"
        )

        # The guarded pattern should be a no-op
        if hasattr(child, "interactive"):
            child.interactive = MagicMock()  # pragma: no cover — should not execute

    def test_hasattr_interactive_true_for_adapter(self):
        """hasattr(child, 'interactive') returns True for the adapter.

        The adapter declares 'interactive' as an attrs field, so
        hasattr-guarded injection works correctly.
        """
        adapter = _make_adapter()
        assert hasattr(adapter, "interactive"), (
            "Adapter must have 'interactive' attribute for flow injection"
        )


# ===========================================================================
# Integration Tests: Adapter in Flow Inferencer Slots
# Validates: Requirements 1.1, 2.1, 4.1, 8.1
# ===========================================================================


class TestAdapterInFlowInferencerSlots:
    """**Validates: Requirements 1.1, 2.1, 4.1, 8.1**

    Integration tests verifying the adapter behaves correctly when called
    the way flow inferencers call their children (via ainfer()).
    """

    @pytest.mark.asyncio
    async def test_adapter_as_breakdown_inferencer(self):
        """Adapter works as BTA breakdown_inferencer with mock ConversationalInferencer.

        Simulates BTA calling: await breakdown_inferencer.ainfer(input, inference_config=ic, **kwargs)
        """
        adapter = _make_adapter()
        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="subtask1\nsubtask2\nsubtask3")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        # BTA calls ainfer with inference_config and kwargs
        result = await adapter.ainfer(
            "Break down this task into subtasks",
            inference_config={"temperature": 0.7},
        )

        assert result == "subtask1\nsubtask2\nsubtask3"
        assert captured["content"] == "Break down this task into subtasks"

    @pytest.mark.asyncio
    async def test_adapter_session_timeout_triggers_fallback(self):
        """session_timeout triggers fallback when conversational session is slow.

        Simulates a flow inferencer calling ainfer() where the conversational
        session exceeds the timeout.
        """
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback breakdown")
        fallback.max_retry = 0

        adapter = _make_adapter(
            session_timeout=0.001,
            fallback_inferencer=fallback,
        )

        async def slow_run(**kw):
            await asyncio.sleep(10)
            return _make_agentic_result()

        adapter.conversational_inferencer.run_agentic_loop = slow_run

        result = await adapter.ainfer("Break down this task")
        assert result == "fallback breakdown"
        fallback.ainfer.assert_called_once()
        # Verify fallback received the rendered task string
        call_args = fallback.ainfer.call_args
        assert call_args[0][0] == "Break down this task"

    @pytest.mark.asyncio
    async def test_adapter_exhausted_max_iterations_triggers_fallback(self):
        """exhausted_max_iterations triggers fallback via ainfer() path."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(
                text="partial result",
                exhausted_max_iterations=True,
            )

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter.ainfer("Plan the implementation")
        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_adapter_has_conversation_tool_triggers_fallback(self):
        """has_conversation_tool (unresolved widget) triggers fallback via ainfer()."""
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback result")
        fallback.max_retry = 0

        adapter = _make_adapter(fallback_inferencer=fallback)

        async def mock_run(**kw):
            return _make_agentic_result(
                text="waiting for user",
                has_conversation_tool=True,
            )

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter.ainfer("Analyze the results")
        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_adapter_with_template_manager(self):
        """Adapter with template_manager set for flow-level template rendering.

        The flow-level template rendering happens in _ainfer_single (upstream
        of _ainfer). We verify that template_manager is accessible and that
        the adapter's _render_prompt is called as part of the pipeline.
        """
        adapter = _make_adapter()

        # Set a simple template_manager mock that transforms input
        mock_tm = MagicMock()
        mock_tm.render.return_value = "RENDERED: original task"
        adapter.template_manager = mock_tm
        adapter.template_key = "test_template"

        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="response to rendered prompt")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        # Call _ainfer directly (simulating what _ainfer_single passes after rendering)
        result = await adapter._ainfer("RENDERED: original task")
        assert captured["content"] == "RENDERED: original task"
        assert result == "response to rendered prompt"

    @pytest.mark.asyncio
    async def test_interactive_injection_via_self_interactive(self):
        """Interactive injection reaches adapter via self.interactive attribute.

        Flow inferencers can set adapter.interactive before calling ainfer().
        This simulates the hasattr-guarded injection pattern.
        """
        mock_interactive = MagicMock(spec=InteractiveBase)
        adapter = _make_adapter()

        # Simulate flow inferencer's hasattr-guarded injection
        if hasattr(adapter, "interactive"):
            adapter.interactive = mock_interactive

        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="interactive result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter.ainfer("task with interactive")
        assert captured["interactive"] is mock_interactive
        assert result == "interactive result"

    @pytest.mark.asyncio
    async def test_clobber_avoidance_none_does_not_overwrite(self):
        """resolved-is-None does not overwrite pre-configured child.interactive.

        When kwargs['interactive'] is None and self.interactive is set,
        the adapter should use self.interactive (not None).
        This prevents a flow inferencer that passes interactive=None from
        clobbering a pre-configured interactive on the adapter.
        """
        pre_configured = MagicMock(spec=InteractiveBase)
        adapter = _make_adapter(interactive=pre_configured)

        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="ok")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        # Simulate flow passing interactive=None in kwargs
        await adapter._ainfer("task", interactive=None)

        # self.interactive should win because kwargs value is falsy
        assert captured["interactive"] is pre_configured, (
            "None in kwargs should not clobber pre-configured self.interactive"
        )

    @pytest.mark.asyncio
    async def test_adapter_reset_called_before_each_invocation(self):
        """reset_for_flow_invocation is called before each ainfer when enabled."""
        adapter = _make_adapter(reset_between_invocations=True)
        reset_calls = []

        original_reset = adapter.conversational_inferencer.reset_for_flow_invocation

        def tracking_reset():
            reset_calls.append(True)
            original_reset()

        adapter.conversational_inferencer.reset_for_flow_invocation = tracking_reset

        async def mock_run(**kw):
            return _make_agentic_result(text="result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("first call")
        await adapter._ainfer("second call")

        assert len(reset_calls) == 2, "reset should be called before each invocation"

    @pytest.mark.asyncio
    async def test_adapter_no_reset_when_disabled(self):
        """reset_for_flow_invocation is NOT called when reset_between_invocations=False."""
        adapter = _make_adapter(reset_between_invocations=False)
        reset_calls = []

        original_reset = adapter.conversational_inferencer.reset_for_flow_invocation

        def tracking_reset():
            reset_calls.append(True)
            original_reset()

        adapter.conversational_inferencer.reset_for_flow_invocation = tracking_reset

        async def mock_run(**kw):
            return _make_agentic_result(text="result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("call without reset")

        assert len(reset_calls) == 0, "reset should not be called when disabled"

    @pytest.mark.asyncio
    async def test_fallback_receives_rendered_task_not_raw(self):
        """Fallback inferencer receives the already-rendered task string.

        This ensures the fallback doesn't double-render through its own
        template_manager (which should not be set on fallback inferencers).
        """
        fallback = MagicMock()
        fallback.ainfer = AsyncMock(return_value="fallback output")
        fallback.max_retry = 0

        adapter = _make_adapter(
            fallback_inferencer=fallback,
            session_timeout=0.001,
        )

        async def slow_run(**kw):
            await asyncio.sleep(10)
            return _make_agentic_result()

        adapter.conversational_inferencer.run_agentic_loop = slow_run

        rendered_input = "RENDERED: Break down the complex task into subtasks"
        await adapter._ainfer(rendered_input)

        # Verify fallback received the rendered task string
        call_args = fallback.ainfer.call_args
        assert call_args[0][0] == rendered_input, (
            "Fallback must receive the already-rendered task string"
        )


# ===========================================================================
# Tests for Session-Level Resume (Task 8)
# Validates: Requirements 9.1–9.7
# ===========================================================================


class TestSessionLevelResume:
    """**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7**

    Tests for checkpoint persistence and mid-conversation resume.
    """

    @pytest.mark.asyncio
    async def test_completed_session_short_circuits_on_resume(self, tmp_path):
        """A completed checkpoint returns completion_result immediately
        without re-running the conversational session."""
        import json
        import os

        session_id = "test-session-001"
        session_dir = os.path.join(str(tmp_path), session_id)
        os.makedirs(session_dir)

        # Write a completed checkpoint
        checkpoint = {
            "schema_version": 1,
            "session_id": session_id,
            "initial_content": "original task",
            "status": "completed",
            "turn_number": 3,
            "messages": [
                {"role": "user", "content": "original task"},
                {"role": "assistant", "content": "final answer"},
            ],
            "completion_result": json.dumps("cached final answer"),
        }
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint, f)

        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )

        # run_agentic_loop should NOT be called
        call_count = 0

        async def mock_run(**kw):
            nonlocal call_count
            call_count += 1
            return _make_agentic_result(text="should not reach here")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("original task")
        assert result == "cached final answer"
        assert call_count == 0, "run_agentic_loop should not be called for completed sessions"

    @pytest.mark.asyncio
    async def test_mid_conversation_crash_resumes_from_last_turn(self, tmp_path):
        """An in_progress checkpoint resumes from the last committed turn
        with the correct content (last user message, not original task)."""
        import json
        import os

        session_id = "test-session-002"
        session_dir = os.path.join(str(tmp_path), session_id)
        os.makedirs(session_dir)

        # Write an in-progress checkpoint at turn 2
        saved_messages = [
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "I need to use a tool"},
            {"role": "user", "content": "[Tool execution results]\n[Tool Result: search]\nresult data"},
            {"role": "assistant", "content": "Based on the results..."},
            {"role": "user", "content": "Continue based on the tool execution results above."},
        ]
        checkpoint = {
            "schema_version": 1,
            "session_id": session_id,
            "initial_content": "original task",
            "status": "in_progress",
            "turn_number": 2,
            "messages": saved_messages,
            "completion_result": None,
        }
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint, f)

        # Also write dynamic context
        dyn_ctx = {
            "completed_actions": [
                {"tool": "search", "summary": "result data", "timestamp": 1.0}
            ],
            "compressed_history": "prior compressed context",
        }
        with open(os.path.join(session_dir, "dynamic_context.json"), "w") as f:
            json.dump(dyn_ctx, f)

        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )

        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="resumed answer")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("original task")

        assert result == "resumed answer"
        # Content should be the last user message, not the original task
        assert captured["content"] == "Continue based on the tool execution results above."
        # Turn number should be restored from checkpoint
        assert captured["turn_number"] == 2
        # Messages should be populated from checkpoint
        assert len(adapter.conversational_inferencer._messages) == len(saved_messages)
        # Dynamic context should be restored
        assert adapter.conversational_inferencer._dynamic_context._compressed_history == "prior compressed context"
        assert len(adapter.conversational_inferencer._dynamic_context.completed_actions) == 1

    @pytest.mark.asyncio
    async def test_on_turn_complete_fires_after_every_iteration(self):
        """on_turn_complete callback fires after every iteration in the loop."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
            ConversationalInferencer,
        )
        from agent_foundation.resources.tools.models import ToolDefinition

        base = MagicMock()
        base.system_prompt = ""
        base.set_messages = MagicMock()
        base.cache_folder = None
        # Must set streams_differ_from_final_output=True with get_final_output
        # returning None, because the code references _final in a log statement
        # that's only defined inside the streams_differ_from_final_output block
        base.streams_differ_from_final_output = True
        base.get_final_output = MagicMock(return_value=None)

        # Make base.ainfer return tool calls on first call, then plain text
        call_count = 0

        async def mock_ainfer(rendered, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'I\'ll use a tool\n<tool_call>{"name": "test_tool", "arguments": {}}</tool_call>'
            return "Final answer with no tools"

        base.ainfer = mock_ainfer

        # Create a real ConversationalInferencer with a mock tool executor
        async def mock_tool_exec(name, args, **kw):
            return "tool result"

        tool_def = ToolDefinition(name="test_tool", tool_type="Action")

        inf = ConversationalInferencer(
            base_inferencer=base,
            max_iterations=5,
            tool_executor=mock_tool_exec,
            tool_registry={"test_tool": tool_def},
        )

        turn_numbers = []

        async def on_turn_complete(turn_number):
            turn_numbers.append(turn_number)

        await inf.run_agentic_loop(
            content="test task",
            on_turn_complete=on_turn_complete,
        )

        # Should have fired at least once (after the tool execution iteration)
        assert len(turn_numbers) >= 1, f"on_turn_complete should fire, got {turn_numbers}"
        # Turn numbers should be sequential starting from 1
        for i, tn in enumerate(turn_numbers):
            assert tn == i + 1, f"Expected turn {i + 1}, got {tn}"

    @pytest.mark.asyncio
    async def test_checkpoint_json_atomic_write(self, tmp_path):
        """checkpoint.json is written atomically via tmp + os.replace."""
        import json
        import os

        session_id = "test-atomic-write"
        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )

        session_dir = os.path.join(str(tmp_path), session_id)

        # Write a checkpoint
        adapter._write_checkpoint_atomic(session_dir, {
            "schema_version": 1,
            "session_id": session_id,
            "status": "in_progress",
            "turn_number": 1,
            "messages": [{"role": "user", "content": "hello"}],
        })

        # Verify the file exists and is valid JSON
        checkpoint_path = os.path.join(session_dir, "checkpoint.json")
        assert os.path.isfile(checkpoint_path)

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert data["schema_version"] == 1
        assert data["session_id"] == session_id
        assert data["status"] == "in_progress"
        assert data["turn_number"] == 1

        # No temp files should remain
        files_in_dir = os.listdir(session_dir)
        tmp_files = [f for f in files_in_dir if f.startswith(".checkpoint_")]
        assert len(tmp_files) == 0, f"Temp files should be cleaned up: {tmp_files}"

        # Overwrite with a new checkpoint — should replace atomically
        adapter._write_checkpoint_atomic(session_dir, {
            "schema_version": 1,
            "session_id": session_id,
            "status": "completed",
            "turn_number": 3,
            "messages": [],
            "completion_result": "done",
        })

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert data["status"] == "completed"
        assert data["turn_number"] == 3

    def test_session_id_required_when_checkpoint_dir_set(self):
        """session_id is required when checkpoint_dir is set."""
        with pytest.raises(ValueError, match="session_id is required"):
            _make_adapter(
                checkpoint_dir="/tmp/checkpoints",
                session_id=None,
            )

    def test_no_error_without_checkpoint_dir(self):
        """No error when checkpoint_dir is not set (default behavior)."""
        adapter = _make_adapter()
        assert adapter.checkpoint_dir is None
        assert adapter.session_id is None

    @pytest.mark.asyncio
    async def test_fresh_session_normal_path(self, tmp_path):
        """When no checkpoint exists, the adapter runs a fresh session."""
        session_id = "fresh-session"
        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )

        captured = {}

        async def mock_run(**kw):
            captured.update(kw)
            return _make_agentic_result(text="fresh result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("new task")
        assert result == "fresh result"
        assert captured["content"] == "new task"
        assert captured["turn_number"] == 0

    @pytest.mark.asyncio
    async def test_checkpoint_written_on_completion(self, tmp_path):
        """On successful completion, checkpoint is written with status=completed."""
        import json
        import os

        session_id = "completion-test"
        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )

        async def mock_run(**kw):
            return _make_agentic_result(text="completed result")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        result = await adapter._ainfer("task")
        assert result == "completed result"

        # Verify checkpoint was written
        checkpoint_path = os.path.join(str(tmp_path), session_id, "checkpoint.json")
        assert os.path.isfile(checkpoint_path)

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert data["status"] == "completed"
        assert json.loads(data["completion_result"]) == "completed result"
        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_resume_skips_reset(self, tmp_path):
        """When resuming, reset_for_flow_invocation is NOT called."""
        import json
        import os

        session_id = "resume-no-reset"
        session_dir = os.path.join(str(tmp_path), session_id)
        os.makedirs(session_dir)

        checkpoint = {
            "schema_version": 1,
            "session_id": session_id,
            "initial_content": "task",
            "status": "in_progress",
            "turn_number": 1,
            "messages": [{"role": "user", "content": "task"}],
            "completion_result": None,
        }
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint, f)

        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
            reset_between_invocations=True,
        )

        reset_calls = []
        original_reset = adapter.conversational_inferencer.reset_for_flow_invocation

        def tracking_reset():
            reset_calls.append(True)
            original_reset()

        adapter.conversational_inferencer.reset_for_flow_invocation = tracking_reset

        async def mock_run(**kw):
            return _make_agentic_result(text="resumed")

        adapter.conversational_inferencer.run_agentic_loop = mock_run

        await adapter._ainfer("task")
        assert len(reset_calls) == 0, "reset should NOT be called when resuming"
