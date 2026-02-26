"""Unit tests for TraceNormalizer.

Tests cover:
- Known action type mappings (ElementInteraction.Click → click, etc.)
- UserInputsRequired → wait(True) conversion
- Unrecognized action type flagging
- Wait duration normalization (single value, all same, varied)
- custom_type_map override
- Empty/None target handling

Requirements: 2.1, 2.2, 2.3, 2.4, 2.6
"""

import pytest

from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.normalizer import (
    KNOWN_CANONICAL_TYPES,
    TraceNormalizer,
)
from webaxon.automation.meta_agent.web_normalizer_config import WEB_ACTION_TYPE_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(action_type: str, **kwargs) -> TraceStep:
    """Create a TraceStep with the given action_type and optional overrides."""
    return TraceStep(action_type=action_type, **kwargs)


def _make_trace(steps: list[TraceStep], trace_id: str = "t-1") -> ExecutionTrace:
    """Wrap steps in an ExecutionTrace."""
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Action type mapping tests (Req 2.1, 2.2)
# ---------------------------------------------------------------------------

class TestActionTypeMapping:
    """Verify ACTION_TYPE_MAP entries are applied correctly when provided via custom_type_map."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ElementInteraction.Click", "click"),
            ("ElementInteraction.InputText", "input_text"),
            ("ElementInteraction.AppendText", "append_text"),
            ("ElementInteraction.Scroll", "scroll"),
            ("ElementInteraction.ScrollUpToElement", "scroll_up_to_element"),
            ("ElementInteraction.InputAndSubmit", "input_and_submit"),
            ("UserInputsRequired", "wait"),
        ],
    )
    def test_known_mappings(self, raw: str, expected: str):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        assert normalizer.normalize_action_type(raw) == expected

    @pytest.mark.parametrize(
        "canonical",
        ["visit_url", "click", "no_op", "wait", "input_text", "extract_text", "select_option"],
    )
    def test_canonical_types_pass_through(self, canonical: str):
        normalizer = TraceNormalizer()
        assert normalizer.normalize_action_type(canonical) == canonical

    def test_normalize_step_applies_mapping(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step("ElementInteraction.Click")
        result = normalizer.normalize_step(step)
        assert result.action_type == "click"

    def test_normalize_step_preserves_other_fields(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step(
            "ElementInteraction.InputText",
            target="#input",
            args={"text": "hello"},
            reasoning="type greeting",
        )
        result = normalizer.normalize_step(step)
        assert result.action_type == "input_text"
        assert result.target == "#input"
        assert result.args == {"text": "hello"}
        assert result.reasoning == "type greeting"

    def test_empty_default_means_passthrough(self):
        """With no custom_type_map, unknown types pass through unchanged."""
        normalizer = TraceNormalizer()
        assert normalizer.normalize_action_type("ElementInteraction.Click") == "ElementInteraction.Click"

    def test_empty_default_flags_unknown(self):
        """With no custom_type_map, agent-internal types are flagged as unrecognized."""
        normalizer = TraceNormalizer()
        step = _make_step("ElementInteraction.Click")
        result = normalizer.normalize_step(step)
        assert result.metadata.get("unrecognized_action_type") is True


# ---------------------------------------------------------------------------
# UserInputsRequired → wait(True) (Req 2.6)
# ---------------------------------------------------------------------------

class TestUserInputsRequired:
    def test_converts_to_wait_action(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step("UserInputsRequired")
        result = normalizer.normalize_step(step)
        assert result.action_type == "wait"

    def test_sets_wait_true_in_args(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step("UserInputsRequired")
        result = normalizer.normalize_step(step)
        assert result.args is not None
        assert result.args["wait"] is True

    def test_preserves_existing_args(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step("UserInputsRequired", args={"message": "please login"})
        result = normalizer.normalize_step(step)
        assert result.args["wait"] is True
        assert result.args["message"] == "please login"


# ---------------------------------------------------------------------------
# Unrecognized action type flagging (Req 2.3)
# ---------------------------------------------------------------------------

class TestUnrecognizedActionType:
    def test_unknown_type_flagged_in_metadata(self):
        normalizer = TraceNormalizer()
        step = _make_step("SomeCompletelyUnknownAction")
        result = normalizer.normalize_step(step)
        assert result.action_type == "SomeCompletelyUnknownAction"
        assert result.metadata.get("unrecognized_action_type") is True

    def test_known_mapped_type_not_flagged(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step("ElementInteraction.Click")
        result = normalizer.normalize_step(step)
        assert "unrecognized_action_type" not in result.metadata

    def test_canonical_type_not_flagged(self):
        normalizer = TraceNormalizer()
        step = _make_step("visit_url")
        result = normalizer.normalize_step(step)
        assert "unrecognized_action_type" not in result.metadata


# ---------------------------------------------------------------------------
# Wait duration normalization (Req 2.4)
# ---------------------------------------------------------------------------

class TestWaitDurationNormalization:
    def test_single_wait_duration_unchanged(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace([_make_step("wait", args={"seconds": 5.0})]),
        ]
        result = normalizer.normalize(traces)
        assert result[0].steps[0].args["seconds"] == 5.0

    def test_all_same_durations_unchanged(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace(
                [_make_step("wait", args={"seconds": 3.0})], trace_id="t-1"
            ),
            _make_trace(
                [_make_step("wait", args={"seconds": 3.0})], trace_id="t-2"
            ),
        ]
        result = normalizer.normalize(traces)
        assert result[0].steps[0].args["seconds"] == 3.0
        assert result[1].steps[0].args["seconds"] == 3.0

    def test_varied_durations_normalized_to_median(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace(
                [_make_step("wait", args={"seconds": 1.0})], trace_id="t-1"
            ),
            _make_trace(
                [_make_step("wait", args={"seconds": 5.0})], trace_id="t-2"
            ),
            _make_trace(
                [_make_step("wait", args={"seconds": 3.0})], trace_id="t-3"
            ),
        ]
        result = normalizer.normalize(traces)
        # median of [1.0, 5.0, 3.0] = 3.0
        for trace in result:
            assert trace.steps[0].args["seconds"] == 3.0

    def test_even_count_durations_median(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace(
                [_make_step("wait", args={"seconds": 2.0})], trace_id="t-1"
            ),
            _make_trace(
                [_make_step("wait", args={"seconds": 4.0})], trace_id="t-2"
            ),
        ]
        result = normalizer.normalize(traces)
        # median of [2.0, 4.0] = 3.0
        for trace in result:
            assert trace.steps[0].args["seconds"] == 3.0

    def test_wait_without_seconds_not_affected(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace([_make_step("wait", args={"wait": True})]),
        ]
        result = normalizer.normalize(traces)
        assert "seconds" not in result[0].steps[0].args

    def test_non_wait_steps_unaffected_by_duration_normalization(self):
        normalizer = TraceNormalizer()
        traces = [
            _make_trace(
                [
                    _make_step("click", target="#btn"),
                    _make_step("wait", args={"seconds": 10.0}),
                ],
                trace_id="t-1",
            ),
        ]
        result = normalizer.normalize(traces)
        assert result[0].steps[0].action_type == "click"
        assert result[0].steps[0].target == "#btn"


# ---------------------------------------------------------------------------
# custom_type_map override (Req 2.2)
# ---------------------------------------------------------------------------

class TestCustomTypeMap:
    def test_custom_map_adds_new_mapping(self):
        normalizer = TraceNormalizer(
            custom_type_map={"MyCustomAction": "click"}
        )
        assert normalizer.normalize_action_type("MyCustomAction") == "click"

    def test_custom_map_overrides_default(self):
        normalizer = TraceNormalizer(
            custom_type_map={"ElementInteraction.Click": "double_click"}
        )
        result = normalizer.normalize_action_type("ElementInteraction.Click")
        assert result == "double_click"

    def test_web_mappings_work_alongside_custom(self):
        normalizer = TraceNormalizer(
            custom_type_map={**WEB_ACTION_TYPE_MAP, "MyAction": "my_action"}
        )
        assert normalizer.normalize_action_type("ElementInteraction.Scroll") == "scroll"
        assert normalizer.normalize_action_type("MyAction") == "my_action"

    def test_custom_mapped_type_recognized_as_canonical(self):
        """A type mapped via custom_type_map should not be flagged unrecognized."""
        normalizer = TraceNormalizer(
            custom_type_map={"SpecialAction": "special_op"}
        )
        step = _make_step("SpecialAction")
        result = normalizer.normalize_step(step)
        assert result.action_type == "special_op"
        # "special_op" is a map value, so it's in canonical types
        assert "unrecognized_action_type" not in result.metadata


# ---------------------------------------------------------------------------
# Empty / None target handling (Req 2.1)
# ---------------------------------------------------------------------------

class TestTargetHandling:
    def test_none_target_passes_through(self):
        normalizer = TraceNormalizer()
        step = _make_step("click", target=None)
        result = normalizer.normalize_step(step)
        assert result.target is None

    def test_empty_string_target_passes_through(self):
        normalizer = TraceNormalizer()
        step = _make_step("click", target="")
        result = normalizer.normalize_step(step)
        assert result.target == ""

    def test_string_target_passes_through(self):
        normalizer = TraceNormalizer()
        step = _make_step("click", target="#my-button")
        result = normalizer.normalize_step(step)
        assert result.target == "#my-button"

    def test_dict_target_passes_through(self):
        normalizer = TraceNormalizer()
        target = {"strategy": "css", "value": "div.main"}
        step = _make_step("click", target=target)
        result = normalizer.normalize_step(step)
        assert result.target == target


# ---------------------------------------------------------------------------
# Full normalize() integration (Req 2.1, 2.4)
# ---------------------------------------------------------------------------

class TestNormalizeTraces:
    def test_returns_new_trace_objects(self):
        normalizer = TraceNormalizer()
        original = _make_trace([_make_step("ElementInteraction.Click")])
        result = normalizer.normalize([original])
        assert result[0] is not original
        assert result[0].steps[0] is not original.steps[0]

    def test_preserves_trace_metadata(self):
        normalizer = TraceNormalizer()
        original = _make_trace(
            [_make_step("click")], trace_id="my-trace"
        )
        result = normalizer.normalize([original])
        assert result[0].trace_id == "my-trace"
        assert result[0].task_description == "test task"

    def test_normalizes_all_steps(self):
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        traces = [
            _make_trace([
                _make_step("ElementInteraction.Click"),
                _make_step("ElementInteraction.InputText", args={"text": "hi"}),
                _make_step("visit_url", target="https://example.com"),
            ]),
        ]
        result = normalizer.normalize(traces)
        types = [s.action_type for s in result[0].steps]
        assert types == ["click", "input_text", "visit_url"]

    def test_empty_traces_list(self):
        normalizer = TraceNormalizer()
        result = normalizer.normalize([])
        assert result == []

    def test_trace_with_no_steps(self):
        normalizer = TraceNormalizer()
        result = normalizer.normalize([_make_trace([])])
        assert len(result) == 1
        assert result[0].steps == []

    def test_original_step_metadata_preserved(self):
        normalizer = TraceNormalizer()
        step = _make_step("click", metadata={"custom_key": "value"})
        traces = [_make_trace([step])]
        result = normalizer.normalize(traces)
        assert result[0].steps[0].metadata["custom_key"] == "value"

    def test_does_not_mutate_original_traces(self):
        normalizer = TraceNormalizer()
        step = _make_step("ElementInteraction.Click", args={"force": True})
        original = _make_trace([step])
        normalizer.normalize([original])
        assert original.steps[0].action_type == "ElementInteraction.Click"
