"""Property-based tests for ReflectiveInferencer.

Uses Hypothesis to verify correctness properties across randomized inputs.
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rich_python_utils.string_utils.formatting.handlebars_format import (
    format_template as handlebars_template_format,
)

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    InferencerResponse,
    ReflectionStyles,
)


def _make_mock_inferencer(return_value="mock_result"):
    """Create a mock InferencerBase-like object for ReflectiveInferencer.

    The mock supports:
    - iter_infer(): returns an iterable yielding a single return_value
    - __call__(): returns return_value (used by reflection_inferencer)
    - set_parent_debuggable(): no-op (called during __attrs_post_init__)
    """
    mock = MagicMock()
    mock.__call__ = MagicMock(return_value=return_value)
    mock.iter_infer = MagicMock(return_value=iter([return_value]))
    mock.set_parent_debuggable = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Property 8: ReflectiveInferencer preserves InferencerResponse return type
# Feature: linear-workflow-inheritance, Property 8: ReflectiveInferencer preserves InferencerResponse return type
# Validates: Requirements 9.7
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    reflection_style=st.sampled_from([
        ReflectionStyles.NoReflection,
        ReflectionStyles.Separate,
        ReflectionStyles.Sequential,
        ReflectionStyles.IntegrateAll,
    ]),
    num_reflections=st.integers(min_value=1, max_value=3),
    input_text=st.text(min_size=1, max_size=50),
)
def test_reflective_inferencer_returns_inferencer_response(
    reflection_style, num_reflections, input_text
):
    """ReflectiveInferencer._infer() always returns InferencerResponse."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.reflective_inferencer import (
        ReflectiveInferencer,
    )

    base_inf = _make_mock_inferencer("base_response")
    reflect_inf = _make_mock_inferencer("reflection_response")

    ri = ReflectiveInferencer(
        base_inferencer=base_inf,
        reflection_inferencer=reflect_inf,
        num_reflections=num_reflections,
        reflection_style=reflection_style,
        reflection_prompt_formatter=handlebars_template_format,
        unpack_single_response=True,
    )

    result = ri._infer(input_text)

    assert isinstance(result, InferencerResponse), (
        f"Expected InferencerResponse, got {type(result)} for style={reflection_style}"
    )
