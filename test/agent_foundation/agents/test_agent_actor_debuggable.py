"""
Tests for AgentActor inheriting from Debuggable.

Validates that:
1. AgentActor and its subclasses accept `id` and other Debuggable kwargs
2. Logging methods (log_info, log_error, etc.) are available on AgentActor instances
3. Backward compatibility: existing AgentActor usage without `id` still works
4. @attrs subclasses of AgentActor properly inherit Debuggable attrs
"""
import sys
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

# Add source paths
project_root = Path(__file__).parent.parent.parent.parent.parent
rich_python_utils_src = project_root / "SciencePythonUtils" / "src"
agent_foundation_src = project_root / "ScienceModelingTools" / "src"

for path in [rich_python_utils_src, agent_foundation_src]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest
from attr import attrs, attrib

from agent_foundation.agents.agent_actor import AgentActor, AgentActionResult
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.identifiable import Identifiable


# region Test fixtures

def _dummy_actor(input_data):
    """A simple actor callable for testing."""
    return f"processed: {input_data}"


@attrs
class ConcreteActor(AgentActor):
    """Concrete AgentActor subclass for testing (mirrors WebPageMakeAnswerActor pattern)."""
    extra_field: str = attrib(default="default_value")

    def get_actor_input(
            self,
            action_results: Sequence,
            task_input: Any,
            action_type: str,
            action_target: str = None,
            action_args: Mapping = None,
            attachments: Sequence = None
    ):
        return {"action_type": action_type, "target": action_target}

# endregion


# region Inheritance tests

class TestAgentActorInheritance:
    """Test that AgentActor properly inherits from Debuggable."""

    def test_agent_actor_is_debuggable(self):
        """AgentActor should be a subclass of Debuggable."""
        assert issubclass(AgentActor, Debuggable)

    def test_agent_actor_is_identifiable(self):
        """AgentActor should be a subclass of Identifiable (via Debuggable)."""
        assert issubclass(AgentActor, Identifiable)

    def test_agent_actor_instance_is_debuggable(self):
        """AgentActor instances should be Debuggable instances."""
        actor = ConcreteActor(actor=_dummy_actor, target_action_type="test")
        assert isinstance(actor, Debuggable)
        assert isinstance(actor, Identifiable)

# endregion


# region ID parameter tests

class TestAgentActorId:
    """Test that AgentActor accepts and uses the `id` parameter."""

    def test_agent_actor_accepts_id(self):
        """AgentActor should accept an explicit `id` kwarg."""
        actor = ConcreteActor(actor=_dummy_actor, id='my_actor')
        assert actor.id == 'my_actor'

    def test_agent_actor_auto_generates_id(self):
        """AgentActor should auto-generate an id when none is provided."""
        actor = ConcreteActor(actor=_dummy_actor)
        assert actor.id is not None
        assert len(actor.id) > 0

    def test_agent_actor_unique_auto_ids(self):
        """Two AgentActor instances without explicit id should get different ids."""
        actor1 = ConcreteActor(actor=_dummy_actor)
        actor2 = ConcreteActor(actor=_dummy_actor)
        assert actor1.id != actor2.id

    def test_subclass_accepts_id_with_extra_fields(self):
        """Subclass with extra attrs should accept id alongside subclass-specific fields."""
        actor = ConcreteActor(
            actor=_dummy_actor,
            target_action_type="WebPage.MakeAnswer",
            extra_field="custom",
            id='make_answer_actor'
        )
        assert actor.id == 'make_answer_actor'
        assert actor.extra_field == "custom"
        assert actor.target_action_type == "WebPage.MakeAnswer"

# endregion


# region Backward compatibility tests

class TestAgentActorBackwardCompatibility:
    """Test that existing AgentActor usage (without Debuggable kwargs) still works."""

    def test_basic_construction_without_debuggable_kwargs(self):
        """AgentActor should work without any Debuggable-specific kwargs."""
        actor = ConcreteActor(actor=_dummy_actor, target_action_type="test")
        assert actor.actor is _dummy_actor
        assert actor.target_action_type == "test"

    def test_call_still_works(self):
        """The __call__ mechanism should still dispatch correctly."""
        actor = ConcreteActor(actor=_dummy_actor, target_action_type="test_action")
        result = actor(
            action_results=[],
            task_input="input",
            action_type="test_action",
            action_target="target"
        )
        assert result == "processed: {'action_type': 'test_action', 'target': 'target'}"

    def test_call_raises_on_wrong_action_type(self):
        """Call with mismatched action_type should still raise ValueError."""
        actor = ConcreteActor(actor=_dummy_actor, target_action_type="expected")
        with pytest.raises(ValueError, match="cannot match"):
            actor(
                action_results=[],
                task_input="input",
                action_type="wrong_type"
            )

# endregion


# region Logging capability tests

class TestAgentActorLogging:
    """Test that AgentActor instances can use Debuggable logging methods."""

    def test_log_info_available(self):
        """AgentActor instances should have log_info method."""
        actor = ConcreteActor(actor=_dummy_actor, id='test_actor')
        assert hasattr(actor, 'log_info')
        assert callable(actor.log_info)

    def test_log_error_available(self):
        """AgentActor instances should have log_error method."""
        actor = ConcreteActor(actor=_dummy_actor, id='test_actor')
        assert hasattr(actor, 'log_error')

    def test_log_info_with_print_logger(self, capsys):
        """AgentActor should be able to log via print logger."""
        actor = ConcreteActor(
            actor=_dummy_actor,
            id='test_actor',
            logger=print,
            debug_mode=True,
            log_time=False,
            always_add_logging_based_logger=False
        )
        actor.log_info("actor log message", "TestLog")
        captured = capsys.readouterr()
        assert "actor log message" in captured.out

    def test_log_info_with_callable_logger(self):
        """AgentActor should be able to log via callable logger."""
        log_records = []

        def capture_logger(log_data):
            log_records.append(log_data)

        actor = ConcreteActor(
            actor=_dummy_actor,
            id='test_actor',
            logger=capture_logger,
            debug_mode=True,
            always_add_logging_based_logger=False
        )
        actor.log_info("captured message", "TestLog")
        assert len(log_records) == 1
        assert "captured message" in str(log_records[0].get('item', ''))

    def test_debug_mode_controls_debug_logging(self):
        """debug_mode=False should suppress DEBUG-level messages."""
        log_records = []

        def capture_logger(log_data):
            log_records.append(log_data)

        actor = ConcreteActor(
            actor=_dummy_actor,
            id='test_actor',
            logger=capture_logger,
            debug_mode=False,
            always_add_logging_based_logger=False
        )
        actor.log_debug("debug message", "TestLog")
        # debug_mode=False means log threshold is INFO, so DEBUG should be suppressed
        assert len(log_records) == 0

# endregion


# region AgentActionResult unchanged tests

class TestAgentActionResultUnchanged:
    """Verify AgentActionResult is not affected by the AgentActor change."""

    def test_agent_action_result_construction(self):
        """AgentActionResult should still work as before."""
        result = AgentActionResult(summary="done", details="all good")
        assert result.summary == "done"
        assert result.details == "all good"
        assert str(result) == "done"

    def test_agent_action_result_not_debuggable(self):
        """AgentActionResult should NOT be Debuggable (it wasn't changed)."""
        assert not issubclass(AgentActionResult, Debuggable)

# endregion
