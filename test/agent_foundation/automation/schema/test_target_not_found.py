"""
Unit Tests for Target Not Found Exception Classes

Tests the TargetNotFoundError and BranchAlreadyExistsError exception classes
defined in common.py for the target_not_found feature.

**Feature: action-graph-target-not-found**
**Requirements: 8.1, 8.2, 8.3**
"""

# Path resolution - must be first
import sys
from pathlib import Path

# Configuration
PIVOT_FOLDER_NAME = 'test'  # The folder name we're inside of

# Get absolute path to this file
current_file = Path(__file__).resolve()

# Navigate up to find the pivot folder (test directory)
current_path = current_file.parent
while current_path.name != PIVOT_FOLDER_NAME and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name != PIVOT_FOLDER_NAME:
    raise RuntimeError(f"Could not find '{PIVOT_FOLDER_NAME}' folder in path hierarchy")

# ScienceModelingTools root is parent of test/ directory
smt_root = current_path.parent

# Add src directory to path for imports
src_dir = smt_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add SciencePythonUtils if it exists
projects_root = smt_root.parent
science_python_utils_src = projects_root / "SciencePythonUtils" / "src"

if science_python_utils_src.exists() and str(science_python_utils_src) not in sys.path:
    sys.path.insert(0, str(science_python_utils_src))

import pytest
from science_modeling_tools.automation.schema.common import (
    TargetNotFoundError,
    BranchAlreadyExistsError,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
    ExecutionRuntime,
)


# =============================================================================
# Task 7.1: Test TargetNotFoundError message format and attributes
# =============================================================================

class TestTargetNotFoundError:
    """Tests for TargetNotFoundError exception class."""

    # -------------------------------------------------------------------------
    # Test attributes are correctly stored
    # -------------------------------------------------------------------------

    def test_has_action_type_attribute(self):
        """TargetNotFoundError should store action_type attribute."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=4,
            max_retries=3
        )
        assert error.action_type == "click"

    def test_has_target_attribute(self):
        """TargetNotFoundError should store target attribute."""
        target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        error = TargetNotFoundError(
            action_type="click",
            target=target,
            attempt_count=4,
            max_retries=3
        )
        assert error.target is target

    def test_has_attempt_count_attribute(self):
        """TargetNotFoundError should store attempt_count attribute."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=4,
            max_retries=3
        )
        assert error.attempt_count == 4

    def test_has_max_retries_attribute(self):
        """TargetNotFoundError should store max_retries attribute."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=4,
            max_retries=3
        )
        assert error.max_retries == 3

    # -------------------------------------------------------------------------
    # Test message format
    # -------------------------------------------------------------------------

    def test_message_format_with_string_target(self):
        """TargetNotFoundError message should format string targets as-is."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=4,
            max_retries=3
        )
        expected = (
            "Target not found after 4 attempts "
            "(1 initial + 3 retries allowed). "
            "Action: click, Target: submit-btn"
        )
        assert str(error) == expected

    def test_message_format_with_target_spec(self):
        """TargetNotFoundError message should format TargetSpec as 'strategy:value'."""
        target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        error = TargetNotFoundError(
            action_type="click",
            target=target,
            attempt_count=4,
            max_retries=3
        )
        expected = (
            "Target not found after 4 attempts "
            "(1 initial + 3 retries allowed). "
            "Action: click, Target: id:submit-btn"
        )
        assert str(error) == expected

    def test_message_format_with_target_spec_xpath(self):
        """TargetNotFoundError message should format XPath TargetSpec correctly."""
        target = TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@type='submit']")
        error = TargetNotFoundError(
            action_type="click",
            target=target,
            attempt_count=2,
            max_retries=1
        )
        expected = (
            "Target not found after 2 attempts "
            "(1 initial + 1 retries allowed). "
            "Action: click, Target: xpath://button[@type='submit']"
        )
        assert str(error) == expected

    def test_message_format_with_target_spec_css(self):
        """TargetNotFoundError message should format CSS TargetSpec correctly."""
        target = TargetSpec(strategy=TargetStrategy.CSS, value="button.submit")
        error = TargetNotFoundError(
            action_type="input_text",
            target=target,
            attempt_count=3,
            max_retries=2
        )
        expected = (
            "Target not found after 3 attempts "
            "(1 initial + 2 retries allowed). "
            "Action: input_text, Target: css:button.submit"
        )
        assert str(error) == expected

    def test_message_format_with_target_spec_with_fallback(self):
        """TargetNotFoundError message should format TargetSpecWithFallback as 'fallback[N strategies]'."""
        target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy=TargetStrategy.ID, value="submit-btn"),
            TargetSpec(strategy=TargetStrategy.CSS, value="button.submit"),
            TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@type='submit']")
        ])
        error = TargetNotFoundError(
            action_type="click",
            target=target,
            attempt_count=4,
            max_retries=3
        )
        expected = (
            "Target not found after 4 attempts "
            "(1 initial + 3 retries allowed). "
            "Action: click, Target: fallback[3 strategies]"
        )
        assert str(error) == expected

    def test_message_format_with_single_strategy_fallback(self):
        """TargetNotFoundError message should handle single strategy fallback."""
        target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        ])
        error = TargetNotFoundError(
            action_type="click",
            target=target,
            attempt_count=2,
            max_retries=1
        )
        expected = (
            "Target not found after 2 attempts "
            "(1 initial + 1 retries allowed). "
            "Action: click, Target: fallback[1 strategies]"
        )
        assert str(error) == expected

    # -------------------------------------------------------------------------
    # Test singular vs plural "attempt/attempts"
    # -------------------------------------------------------------------------

    def test_message_singular_attempt(self):
        """TargetNotFoundError message should use singular 'attempt' for count=1."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=1,
            max_retries=0
        )
        expected = (
            "Target not found after 1 attempt "
            "(1 initial + 0 retries allowed). "
            "Action: click, Target: submit-btn"
        )
        assert str(error) == expected

    def test_message_plural_attempts_for_two(self):
        """TargetNotFoundError message should use plural 'attempts' for count=2."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=2,
            max_retries=1
        )
        expected = (
            "Target not found after 2 attempts "
            "(1 initial + 1 retries allowed). "
            "Action: click, Target: submit-btn"
        )
        assert str(error) == expected

    def test_message_plural_attempts_for_many(self):
        """TargetNotFoundError message should use plural 'attempts' for count > 1."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=10,
            max_retries=9
        )
        expected = (
            "Target not found after 10 attempts "
            "(1 initial + 9 retries allowed). "
            "Action: click, Target: submit-btn"
        )
        assert str(error) == expected

    # -------------------------------------------------------------------------
    # Test edge cases
    # -------------------------------------------------------------------------

    def test_with_zero_max_retries(self):
        """TargetNotFoundError should handle max_retries=0 correctly."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=1,
            max_retries=0
        )
        assert error.max_retries == 0
        assert "0 retries allowed" in str(error)

    def test_with_different_action_types(self):
        """TargetNotFoundError should work with various action types."""
        action_types = ["click", "input_text", "scroll", "hover", "double_click"]
        for action_type in action_types:
            error = TargetNotFoundError(
                action_type=action_type,
                target="element",
                attempt_count=2,
                max_retries=1
            )
            assert error.action_type == action_type
            assert f"Action: {action_type}" in str(error)

    def test_is_exception_subclass(self):
        """TargetNotFoundError should be a subclass of Exception."""
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=1,
            max_retries=0
        )
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """TargetNotFoundError should be raisable and catchable."""
        with pytest.raises(TargetNotFoundError) as exc_info:
            raise TargetNotFoundError(
                action_type="click",
                target="submit-btn",
                attempt_count=4,
                max_retries=3
            )
        assert exc_info.value.action_type == "click"
        assert exc_info.value.attempt_count == 4


# =============================================================================
# Task 7.2: Test BranchAlreadyExistsError message format and attributes
# =============================================================================

class TestBranchAlreadyExistsError:
    """Tests for BranchAlreadyExistsError exception class."""

    # -------------------------------------------------------------------------
    # Test attributes are correctly stored
    # -------------------------------------------------------------------------

    def test_has_condition_attribute(self):
        """BranchAlreadyExistsError should store condition attribute."""
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type="click"
        )
        assert error.condition == "target_not_found"

    def test_has_action_type_attribute(self):
        """BranchAlreadyExistsError should store action_type attribute."""
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type="click"
        )
        assert error.action_type == "click"

    # -------------------------------------------------------------------------
    # Test message format
    # -------------------------------------------------------------------------

    def test_message_format(self):
        """BranchAlreadyExistsError message should follow expected format."""
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type="click"
        )
        expected = "Branch 'target_not_found' already exists on action 'click'."
        assert str(error) == expected

    def test_message_format_with_different_condition(self):
        """BranchAlreadyExistsError message should include the condition name."""
        error = BranchAlreadyExistsError(
            condition="on_error",
            action_type="input_text"
        )
        expected = "Branch 'on_error' already exists on action 'input_text'."
        assert str(error) == expected

    def test_message_format_with_different_action_type(self):
        """BranchAlreadyExistsError message should include the action type."""
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type="scroll"
        )
        expected = "Branch 'target_not_found' already exists on action 'scroll'."
        assert str(error) == expected

    # -------------------------------------------------------------------------
    # Test edge cases
    # -------------------------------------------------------------------------

    def test_with_various_action_types(self):
        """BranchAlreadyExistsError should work with various action types."""
        action_types = ["click", "input_text", "scroll", "hover", "double_click", "visit_url"]
        for action_type in action_types:
            error = BranchAlreadyExistsError(
                condition="target_not_found",
                action_type=action_type
            )
            assert error.action_type == action_type
            assert f"action '{action_type}'" in str(error)

    def test_is_exception_subclass(self):
        """BranchAlreadyExistsError should be a subclass of Exception."""
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type="click"
        )
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """BranchAlreadyExistsError should be raisable and catchable."""
        with pytest.raises(BranchAlreadyExistsError) as exc_info:
            raise BranchAlreadyExistsError(
                condition="target_not_found",
                action_type="click"
            )
        assert exc_info.value.condition == "target_not_found"
        assert exc_info.value.action_type == "click"

    def test_can_be_caught_as_exception(self):
        """BranchAlreadyExistsError should be catchable as generic Exception."""
        with pytest.raises(Exception) as exc_info:
            raise BranchAlreadyExistsError(
                condition="target_not_found",
                action_type="click"
            )
        assert "Branch 'target_not_found' already exists" in str(exc_info.value)


# =============================================================================
# Task 8: Unit Tests for Action Model Extension
# =============================================================================

# Import Action model for testing
from science_modeling_tools.automation.schema.common import Action


# =============================================================================
# Task 8.1: Test Action with target_not_found_actions field
# =============================================================================

class TestActionTargetNotFoundActionsField:
    """Tests for Action model's target_not_found_actions field."""

    # -------------------------------------------------------------------------
    # Test field accepts valid values
    # -------------------------------------------------------------------------

    def test_action_accepts_none_target_not_found_actions(self):
        """Action should accept None for target_not_found_actions (default)."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        assert action.target_not_found_actions is None

    def test_action_accepts_empty_list_target_not_found_actions(self):
        """Action should accept empty list for target_not_found_actions."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[]
        )
        assert action.target_not_found_actions == []

    def test_action_accepts_single_action_in_target_not_found_actions(self):
        """Action should accept a single Action in target_not_found_actions."""
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action]
        )
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_actions[0].id == "fallback_action"
        assert action.target_not_found_actions[0].type == "click"

    def test_action_accepts_multiple_actions_in_target_not_found_actions(self):
        """Action should accept multiple Actions in target_not_found_actions."""
        fallback_actions = [
            Action(id="fallback_1", type="click", target="btn1"),
            Action(id="fallback_2", type="input_text", target="input1", args={"text": "hello"}),
            Action(id="fallback_3", type="scroll", target="container")
        ]
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=fallback_actions
        )
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 3
        assert action.target_not_found_actions[0].id == "fallback_1"
        assert action.target_not_found_actions[1].id == "fallback_2"
        assert action.target_not_found_actions[2].id == "fallback_3"

    def test_action_with_nested_target_not_found_actions(self):
        """Action should support nested target_not_found_actions (action with branch that has its own branch)."""
        # Inner fallback action
        inner_fallback = Action(
            id="inner_fallback",
            type="click",
            target="inner-btn"
        )
        # Outer fallback action with its own branch
        outer_fallback = Action(
            id="outer_fallback",
            type="click",
            target="outer-btn",
            target_not_found_actions=[inner_fallback]
        )
        # Main action
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[outer_fallback]
        )
        
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        
        outer = action.target_not_found_actions[0]
        assert outer.id == "outer_fallback"
        assert outer.target_not_found_actions is not None
        assert len(outer.target_not_found_actions) == 1
        
        inner = outer.target_not_found_actions[0]
        assert inner.id == "inner_fallback"

    def test_action_with_target_spec_in_target_not_found_actions(self):
        """Action should accept fallback actions with TargetSpec targets."""
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target=TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@id='fallback']")
        )
        action = Action(
            id="test_action",
            type="click",
            target=TargetSpec(strategy=TargetStrategy.ID, value="submit-btn"),
            target_not_found_actions=[fallback_action]
        )
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_actions[0].target.strategy == TargetStrategy.XPATH


# =============================================================================
# Task 8.2: Test Action with target_not_found_config field
# =============================================================================

class TestActionTargetNotFoundConfigField:
    """Tests for Action model's target_not_found_config field."""

    # -------------------------------------------------------------------------
    # Test field accepts valid values
    # -------------------------------------------------------------------------

    def test_action_accepts_none_target_not_found_config(self):
        """Action should accept None for target_not_found_config (default)."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        assert action.target_not_found_config is None

    def test_action_accepts_empty_dict_target_not_found_config(self):
        """Action should accept empty dict for target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={}
        )
        assert action.target_not_found_config == {}

    def test_action_accepts_retry_after_handling_config(self):
        """Action should accept retry_after_handling in target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={"retry_after_handling": True}
        )
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is True

    def test_action_accepts_max_retries_config(self):
        """Action should accept max_retries in target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={"max_retries": 5}
        )
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["max_retries"] == 5

    def test_action_accepts_retry_delay_config(self):
        """Action should accept retry_delay in target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={"retry_delay": 2.5}
        )
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_delay"] == 2.5

    def test_action_accepts_full_target_not_found_config(self):
        """Action should accept all config keys in target_not_found_config."""
        config = {
            "retry_after_handling": True,
            "max_retries": 3,
            "retry_delay": 1.0
        }
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config=config
        )
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is True
        assert action.target_not_found_config["max_retries"] == 3
        assert action.target_not_found_config["retry_delay"] == 1.0

    def test_action_accepts_both_actions_and_config(self):
        """Action should accept both target_not_found_actions and target_not_found_config."""
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        config = {
            "retry_after_handling": True,
            "max_retries": 5,
            "retry_delay": 2.0
        }
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config=config
        )
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is True

    def test_action_config_with_zero_max_retries(self):
        """Action should accept max_retries=0 in target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={"max_retries": 0}
        )
        assert action.target_not_found_config["max_retries"] == 0

    def test_action_config_with_zero_retry_delay(self):
        """Action should accept retry_delay=0 in target_not_found_config."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config={"retry_delay": 0.0}
        )
        assert action.target_not_found_config["retry_delay"] == 0.0


# =============================================================================
# Task 8.3: Test Action JSON serialization with new fields
# =============================================================================

class TestActionJsonSerialization:
    """Tests for Action model JSON serialization with target_not_found fields."""

    # -------------------------------------------------------------------------
    # Test .dict() serialization
    # -------------------------------------------------------------------------

    def test_action_dict_with_none_target_not_found_fields(self):
        """Action.dict() should include None target_not_found fields."""
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        data = action.dict()
        assert "target_not_found_actions" in data
        assert data["target_not_found_actions"] is None
        assert "target_not_found_config" in data
        assert data["target_not_found_config"] is None

    def test_action_dict_with_target_not_found_actions(self):
        """Action.dict() should serialize target_not_found_actions correctly."""
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action]
        )
        data = action.dict()
        assert data["target_not_found_actions"] is not None
        assert len(data["target_not_found_actions"]) == 1
        assert data["target_not_found_actions"][0]["id"] == "fallback_action"
        assert data["target_not_found_actions"][0]["type"] == "click"

    def test_action_dict_with_target_not_found_config(self):
        """Action.dict() should serialize target_not_found_config correctly."""
        config = {
            "retry_after_handling": True,
            "max_retries": 3,
            "retry_delay": 1.5
        }
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config=config
        )
        data = action.dict()
        assert data["target_not_found_config"] is not None
        assert data["target_not_found_config"]["retry_after_handling"] is True
        assert data["target_not_found_config"]["max_retries"] == 3
        assert data["target_not_found_config"]["retry_delay"] == 1.5

    def test_action_dict_with_nested_target_not_found_actions(self):
        """Action.dict() should serialize nested target_not_found_actions correctly."""
        inner_fallback = Action(
            id="inner_fallback",
            type="click",
            target="inner-btn"
        )
        outer_fallback = Action(
            id="outer_fallback",
            type="click",
            target="outer-btn",
            target_not_found_actions=[inner_fallback]
        )
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[outer_fallback]
        )
        data = action.dict()
        
        # Check outer level
        assert data["target_not_found_actions"] is not None
        assert len(data["target_not_found_actions"]) == 1
        
        # Check nested level
        outer_data = data["target_not_found_actions"][0]
        assert outer_data["id"] == "outer_fallback"
        assert outer_data["target_not_found_actions"] is not None
        assert len(outer_data["target_not_found_actions"]) == 1
        
        # Check innermost level
        inner_data = outer_data["target_not_found_actions"][0]
        assert inner_data["id"] == "inner_fallback"

    # -------------------------------------------------------------------------
    # Test .json() serialization
    # -------------------------------------------------------------------------

    def test_action_json_with_none_target_not_found_fields(self):
        """Action.json() should serialize None target_not_found fields."""
        import json
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        json_str = action.json()
        data = json.loads(json_str)
        assert "target_not_found_actions" in data
        assert data["target_not_found_actions"] is None
        assert "target_not_found_config" in data
        assert data["target_not_found_config"] is None

    def test_action_json_with_target_not_found_actions(self):
        """Action.json() should serialize target_not_found_actions correctly."""
        import json
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action]
        )
        json_str = action.json()
        data = json.loads(json_str)
        assert data["target_not_found_actions"] is not None
        assert len(data["target_not_found_actions"]) == 1
        assert data["target_not_found_actions"][0]["id"] == "fallback_action"

    def test_action_json_with_target_not_found_config(self):
        """Action.json() should serialize target_not_found_config correctly."""
        import json
        config = {
            "retry_after_handling": False,
            "max_retries": 5,
            "retry_delay": 0.5
        }
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config=config
        )
        json_str = action.json()
        data = json.loads(json_str)
        assert data["target_not_found_config"] is not None
        assert data["target_not_found_config"]["retry_after_handling"] is False
        assert data["target_not_found_config"]["max_retries"] == 5
        assert data["target_not_found_config"]["retry_delay"] == 0.5

    def test_action_json_with_both_fields(self):
        """Action.json() should serialize both target_not_found fields correctly."""
        import json
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        config = {
            "retry_after_handling": True,
            "max_retries": 3,
            "retry_delay": 1.0
        }
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config=config
        )
        json_str = action.json()
        data = json.loads(json_str)
        
        assert data["target_not_found_actions"] is not None
        assert len(data["target_not_found_actions"]) == 1
        assert data["target_not_found_config"] is not None
        assert data["target_not_found_config"]["retry_after_handling"] is True


# =============================================================================
# Task 8.4: Test Action JSON deserialization with new fields
# =============================================================================

class TestActionJsonDeserialization:
    """Tests for Action model JSON deserialization with target_not_found fields."""

    # -------------------------------------------------------------------------
    # Test Action(**dict) deserialization
    # -------------------------------------------------------------------------

    def test_action_from_dict_with_none_target_not_found_fields(self):
        """Action(**dict) should deserialize None target_not_found fields."""
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_actions": None,
            "target_not_found_config": None
        }
        action = Action(**data)
        assert action.id == "test_action"
        assert action.target_not_found_actions is None
        assert action.target_not_found_config is None

    def test_action_from_dict_without_target_not_found_fields(self):
        """Action(**dict) should handle missing target_not_found fields (defaults to None)."""
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn"
        }
        action = Action(**data)
        assert action.id == "test_action"
        assert action.target_not_found_actions is None
        assert action.target_not_found_config is None

    def test_action_from_dict_with_target_not_found_actions(self):
        """Action(**dict) should deserialize target_not_found_actions correctly."""
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_actions": [
                {
                    "id": "fallback_action",
                    "type": "click",
                    "target": "fallback-btn"
                }
            ]
        }
        action = Action(**data)
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_actions[0].id == "fallback_action"
        assert action.target_not_found_actions[0].type == "click"

    def test_action_from_dict_with_target_not_found_config(self):
        """Action(**dict) should deserialize target_not_found_config correctly."""
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_config": {
                "retry_after_handling": True,
                "max_retries": 3,
                "retry_delay": 1.5
            }
        }
        action = Action(**data)
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is True
        assert action.target_not_found_config["max_retries"] == 3
        assert action.target_not_found_config["retry_delay"] == 1.5

    def test_action_from_dict_with_nested_target_not_found_actions(self):
        """Action(**dict) should deserialize nested target_not_found_actions correctly."""
        data = {
            "id": "main_action",
            "type": "click",
            "target": "main-btn",
            "target_not_found_actions": [
                {
                    "id": "outer_fallback",
                    "type": "click",
                    "target": "outer-btn",
                    "target_not_found_actions": [
                        {
                            "id": "inner_fallback",
                            "type": "click",
                            "target": "inner-btn"
                        }
                    ]
                }
            ]
        }
        action = Action(**data)
        
        # Check outer level
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        
        # Check nested level
        outer = action.target_not_found_actions[0]
        assert outer.id == "outer_fallback"
        assert outer.target_not_found_actions is not None
        assert len(outer.target_not_found_actions) == 1
        
        # Check innermost level
        inner = outer.target_not_found_actions[0]
        assert inner.id == "inner_fallback"

    # -------------------------------------------------------------------------
    # Test Action.parse_raw() deserialization
    # -------------------------------------------------------------------------

    def test_action_parse_raw_with_none_target_not_found_fields(self):
        """Action.parse_raw() should deserialize None target_not_found fields."""
        import json
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_actions": None,
            "target_not_found_config": None
        }
        json_str = json.dumps(data)
        action = Action.parse_raw(json_str)
        assert action.id == "test_action"
        assert action.target_not_found_actions is None
        assert action.target_not_found_config is None

    def test_action_parse_raw_with_target_not_found_actions(self):
        """Action.parse_raw() should deserialize target_not_found_actions correctly."""
        import json
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_actions": [
                {
                    "id": "fallback_action",
                    "type": "click",
                    "target": "fallback-btn"
                }
            ]
        }
        json_str = json.dumps(data)
        action = Action.parse_raw(json_str)
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_actions[0].id == "fallback_action"

    def test_action_parse_raw_with_target_not_found_config(self):
        """Action.parse_raw() should deserialize target_not_found_config correctly."""
        import json
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_config": {
                "retry_after_handling": False,
                "max_retries": 5,
                "retry_delay": 2.0
            }
        }
        json_str = json.dumps(data)
        action = Action.parse_raw(json_str)
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is False
        assert action.target_not_found_config["max_retries"] == 5
        assert action.target_not_found_config["retry_delay"] == 2.0

    def test_action_parse_raw_with_both_fields(self):
        """Action.parse_raw() should deserialize both target_not_found fields correctly."""
        import json
        data = {
            "id": "test_action",
            "type": "click",
            "target": "submit-btn",
            "target_not_found_actions": [
                {
                    "id": "fallback_action",
                    "type": "click",
                    "target": "fallback-btn"
                }
            ],
            "target_not_found_config": {
                "retry_after_handling": True,
                "max_retries": 3,
                "retry_delay": 1.0
            }
        }
        json_str = json.dumps(data)
        action = Action.parse_raw(json_str)
        
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) == 1
        assert action.target_not_found_config is not None
        assert action.target_not_found_config["retry_after_handling"] is True

    # -------------------------------------------------------------------------
    # Test round-trip serialization/deserialization
    # -------------------------------------------------------------------------

    def test_action_round_trip_with_target_not_found_actions(self):
        """Action should survive round-trip serialization with target_not_found_actions."""
        import json
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        original = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action]
        )
        
        # Serialize and deserialize
        json_str = original.json()
        restored = Action.parse_raw(json_str)
        
        # Verify
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.target_not_found_actions is not None
        assert len(restored.target_not_found_actions) == 1
        assert restored.target_not_found_actions[0].id == "fallback_action"

    def test_action_round_trip_with_target_not_found_config(self):
        """Action should survive round-trip serialization with target_not_found_config."""
        import json
        config = {
            "retry_after_handling": True,
            "max_retries": 7,
            "retry_delay": 3.5
        }
        original = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_config=config
        )
        
        # Serialize and deserialize
        json_str = original.json()
        restored = Action.parse_raw(json_str)
        
        # Verify
        assert restored.id == original.id
        assert restored.target_not_found_config is not None
        assert restored.target_not_found_config["retry_after_handling"] is True
        assert restored.target_not_found_config["max_retries"] == 7
        assert restored.target_not_found_config["retry_delay"] == 3.5

    def test_action_round_trip_with_nested_target_not_found_actions(self):
        """Action should survive round-trip serialization with nested target_not_found_actions."""
        import json
        inner_fallback = Action(
            id="inner_fallback",
            type="click",
            target="inner-btn"
        )
        outer_fallback = Action(
            id="outer_fallback",
            type="click",
            target="outer-btn",
            target_not_found_actions=[inner_fallback],
            target_not_found_config={"retry_after_handling": True, "max_retries": 2, "retry_delay": 0.5}
        )
        original = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[outer_fallback],
            target_not_found_config={"retry_after_handling": False, "max_retries": 3, "retry_delay": 1.0}
        )
        
        # Serialize and deserialize
        json_str = original.json()
        restored = Action.parse_raw(json_str)
        
        # Verify main action
        assert restored.id == "main_action"
        assert restored.target_not_found_config["retry_after_handling"] is False
        
        # Verify outer fallback
        outer = restored.target_not_found_actions[0]
        assert outer.id == "outer_fallback"
        assert outer.target_not_found_config["retry_after_handling"] is True
        
        # Verify inner fallback
        inner = outer.target_not_found_actions[0]
        assert inner.id == "inner_fallback"

    def test_action_round_trip_with_target_spec(self):
        """Action should survive round-trip serialization with TargetSpec in fallback actions."""
        import json
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target=TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@id='fallback']")
        )
        original = Action(
            id="test_action",
            type="click",
            target=TargetSpec(strategy=TargetStrategy.ID, value="submit-btn"),
            target_not_found_actions=[fallback_action]
        )
        
        # Serialize and deserialize
        json_str = original.json()
        restored = Action.parse_raw(json_str)
        
        # Verify main action target - after deserialization, target is a TargetSpec object
        assert isinstance(restored.target, TargetSpec)
        assert restored.target.strategy == TargetStrategy.ID
        assert restored.target.value == "submit-btn"
        
        # Verify fallback action target
        fallback = restored.target_not_found_actions[0]
        assert isinstance(fallback.target, TargetSpec)
        assert fallback.target.strategy == TargetStrategy.XPATH
        assert fallback.target.value == "//button[@id='fallback']"


# =============================================================================
# Task 9: Unit Tests for ActionChainHelper
# =============================================================================

# Import ActionGraph and related classes for testing
from science_modeling_tools.automation.schema.action_graph import (
    ActionGraph,
    ActionChainHelper,
    TargetNotFoundContext,
)


# =============================================================================
# Task 9.1: Test action() returns ActionChainHelper (for non-monitor actions)
# =============================================================================

class TestActionReturnsActionChainHelper:
    """Tests that ActionGraph.action() returns ActionChainHelper for non-monitor actions."""

    # -------------------------------------------------------------------------
    # Test basic return type
    # -------------------------------------------------------------------------

    def test_action_returns_action_chain_helper(self):
        """ActionGraph.action() should return ActionChainHelper for non-monitor actions."""
        # Create a mock action executor
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        result = graph.action("click", target="submit-btn")
        
        assert isinstance(result, ActionChainHelper)

    def test_action_returns_action_chain_helper_with_target_spec(self):
        """ActionGraph.action() should return ActionChainHelper when using TargetSpec."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        result = graph.action("click", target=target)
        
        assert isinstance(result, ActionChainHelper)

    def test_action_returns_action_chain_helper_for_various_action_types(self):
        """ActionGraph.action() should return ActionChainHelper for various action types."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        action_types = ["click", "input_text", "scroll", "hover", "double_click", "visit_url"]
        for action_type in action_types:
            result = graph.action(action_type, target="element")
            assert isinstance(result, ActionChainHelper), f"Failed for action type: {action_type}"

    def test_action_returns_action_chain_helper_without_target(self):
        """ActionGraph.action() should return ActionChainHelper even without target."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        result = graph.action("wait", args={"seconds": 1})
        
        assert isinstance(result, ActionChainHelper)

    def test_action_returns_action_chain_helper_with_all_parameters(self):
        """ActionGraph.action() should return ActionChainHelper with all parameters."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        result = graph.action(
            "click",
            target="submit-btn",
            args={"force": True},
            action_id="custom_id",
            condition="some_condition",
            on_error="continue",
            output="result_var",
            timeout=30.0,
            wait=1.0,
            no_action_if_target_not_found=True
        )
        
        assert isinstance(result, ActionChainHelper)


# =============================================================================
# Task 9.2: Test action() returns self for "monitor" action type (special case)
# =============================================================================

class TestActionReturnsGraphForMonitor:
    """Tests that ActionGraph.action() returns self for 'monitor' action type."""

    def test_action_returns_graph_for_monitor_action_type(self):
        """ActionGraph.action() should return self (ActionGraph) for 'monitor' action type."""
        # Note: Monitor action requires webaxon package which may not be available
        # We test that it raises ImportError or returns ActionGraph
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        try:
            result = graph.action(
                "monitor",
                target=TargetSpec(strategy=TargetStrategy.XPATH, value="//div"),
                event_condition="text_changed"
            )
            # If webaxon is available, it should return ActionGraph (self)
            assert result is graph
        except ImportError:
            # Expected if webaxon package is not installed
            pass
        except ValueError as e:
            # May raise ValueError if event_condition is invalid
            if "event_condition" not in str(e):
                raise

    def test_monitor_action_requires_event_condition(self):
        """ActionGraph.action() with 'monitor' should raise ValueError without event_condition."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        with pytest.raises(ValueError) as exc_info:
            graph.action("monitor", target="element")
        
        assert "event_condition" in str(exc_info.value)


# =============================================================================
# Task 9.3: Test target_not_found() returns TargetNotFoundContext
# =============================================================================

class TestTargetNotFoundReturnsContext:
    """Tests that ActionChainHelper.target_not_found() returns TargetNotFoundContext."""

    def test_target_not_found_returns_context(self):
        """ActionChainHelper.target_not_found() should return TargetNotFoundContext."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found()
        
        assert isinstance(result, TargetNotFoundContext)

    def test_target_not_found_returns_context_with_target_spec(self):
        """ActionChainHelper.target_not_found() should return TargetNotFoundContext with TargetSpec."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        helper = graph.action("click", target=target)
        result = helper.target_not_found()
        
        assert isinstance(result, TargetNotFoundContext)

    def test_target_not_found_returns_context_with_parameters(self):
        """ActionChainHelper.target_not_found() should return TargetNotFoundContext with parameters."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found(
            retry_after_handling=True,
            max_retries=5,
            retry_delay=2.0
        )
        
        assert isinstance(result, TargetNotFoundContext)


# =============================================================================
# Task 9.4: Test on_target_not_found() is alias for target_not_found()
# =============================================================================

class TestOnTargetNotFoundAlias:
    """Tests that on_target_not_found() is an alias for target_not_found()."""

    def test_on_target_not_found_is_alias(self):
        """ActionChainHelper.on_target_not_found should be same method as target_not_found."""
        # Check at the class level that on_target_not_found is the same function as target_not_found
        assert ActionChainHelper.on_target_not_found is ActionChainHelper.target_not_found

    def test_on_target_not_found_returns_context(self):
        """ActionChainHelper.on_target_not_found() should return TargetNotFoundContext."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.on_target_not_found()
        
        assert isinstance(result, TargetNotFoundContext)

    def test_on_target_not_found_accepts_same_parameters(self):
        """ActionChainHelper.on_target_not_found() should accept same parameters as target_not_found()."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.on_target_not_found(
            retry_after_handling=True,
            max_retries=5,
            retry_delay=2.0
        )
        
        assert isinstance(result, TargetNotFoundContext)


# =============================================================================
# Task 9.5: Test parameter validation (max_retries, retry_delay, target not None)
# =============================================================================

class TestParameterValidation:
    """Tests for parameter validation in target_not_found()."""

    # -------------------------------------------------------------------------
    # Test max_retries validation
    # -------------------------------------------------------------------------

    def test_max_retries_below_zero_raises_value_error(self):
        """target_not_found() should raise ValueError when max_retries < 0."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(max_retries=-1)
        
        assert "-1" in str(exc_info.value)
        assert "max_retries" in str(exc_info.value)

    def test_max_retries_above_ten_raises_value_error(self):
        """target_not_found() should raise ValueError when max_retries > 10."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(max_retries=11)
        
        assert "11" in str(exc_info.value)
        assert "max_retries" in str(exc_info.value)

    def test_max_retries_zero_is_valid(self):
        """target_not_found() should accept max_retries=0."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found(max_retries=0)
        
        assert isinstance(result, TargetNotFoundContext)

    def test_max_retries_ten_is_valid(self):
        """target_not_found() should accept max_retries=10."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found(max_retries=10)
        
        assert isinstance(result, TargetNotFoundContext)

    # -------------------------------------------------------------------------
    # Test retry_delay validation
    # -------------------------------------------------------------------------

    def test_retry_delay_negative_raises_value_error(self):
        """target_not_found() should raise ValueError when retry_delay < 0."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(retry_delay=-0.5)
        
        assert "-0.5" in str(exc_info.value)
        assert "retry_delay" in str(exc_info.value)

    def test_retry_delay_above_sixty_raises_value_error(self):
        """target_not_found() should raise ValueError when retry_delay > 60."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(retry_delay=61)
        
        assert "61" in str(exc_info.value)
        assert "retry_delay" in str(exc_info.value)

    def test_retry_delay_zero_is_valid(self):
        """target_not_found() should accept retry_delay=0."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found(retry_delay=0)
        
        assert isinstance(result, TargetNotFoundContext)

    def test_retry_delay_sixty_is_valid(self):
        """target_not_found() should accept retry_delay=60."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        result = helper.target_not_found(retry_delay=60)
        
        assert isinstance(result, TargetNotFoundContext)

    # -------------------------------------------------------------------------
    # Test target=None validation
    # -------------------------------------------------------------------------

    def test_target_none_raises_value_error(self):
        """target_not_found() should raise ValueError when action has no target."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action without target
        helper = graph.action("wait", args={"seconds": 1})
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found()
        
        assert "target" in str(exc_info.value).lower()
        assert "wait" in str(exc_info.value)


# =============================================================================
# Task 9.6: Test BranchAlreadyExistsError on duplicate branch
# =============================================================================

class TestBranchAlreadyExistsErrorOnDuplicate:
    """Tests that calling target_not_found() twice raises BranchAlreadyExistsError."""

    def test_duplicate_target_not_found_raises_error(self):
        """Calling target_not_found() twice on same action should raise BranchAlreadyExistsError."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        # First call should succeed
        with helper.target_not_found():
            pass
        
        # Second call should raise BranchAlreadyExistsError
        with pytest.raises(BranchAlreadyExistsError) as exc_info:
            helper.target_not_found()
        
        assert exc_info.value.condition == "target_not_found"
        assert exc_info.value.action_type == "click"

    def test_duplicate_on_target_not_found_raises_error(self):
        """Calling on_target_not_found() twice on same action should raise BranchAlreadyExistsError."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        # First call should succeed
        with helper.on_target_not_found():
            pass
        
        # Second call should raise BranchAlreadyExistsError
        with pytest.raises(BranchAlreadyExistsError) as exc_info:
            helper.on_target_not_found()
        
        assert exc_info.value.condition == "target_not_found"

    def test_mixed_target_not_found_and_alias_raises_error(self):
        """Calling target_not_found() then on_target_not_found() should raise BranchAlreadyExistsError."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        # First call with target_not_found()
        with helper.target_not_found():
            pass
        
        # Second call with on_target_not_found() should raise
        with pytest.raises(BranchAlreadyExistsError):
            helper.on_target_not_found()


# =============================================================================
# Task 9.7: Test as-binding pattern (with graph.action(...) as helper)
# =============================================================================

class TestAsBindingPattern:
    """Tests for the as-binding pattern: with graph.action(...) as helper."""

    def test_as_binding_returns_helper(self):
        """Using 'with graph.action(...) as helper' should bind helper to ActionChainHelper."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        with graph.action("click", target="submit-btn") as helper:
            assert isinstance(helper, ActionChainHelper)

    def test_as_binding_provides_access_to_action_obj(self):
        """As-binding pattern should provide access to action_obj property."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        with graph.action("click", target="submit-btn") as helper:
            assert helper.action_obj is not None
            assert helper.action_obj.type == "click"
            assert helper.action_obj.target == "submit-btn"

    def test_as_binding_is_noop_scope(self):
        """As-binding pattern should be a no-op scope (doesn't change graph state)."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Get initial node count
        initial_node_count = len(graph._nodes)
        
        with graph.action("click", target="submit-btn") as helper:
            # Inside context, graph state should be unchanged
            assert len(graph._nodes) == initial_node_count
        
        # After context, graph state should still be unchanged
        assert len(graph._nodes) == initial_node_count

    def test_as_binding_allows_target_not_found_inside(self):
        """As-binding pattern should allow calling target_not_found() inside."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        with graph.action("click", target="submit-btn") as helper:
            with helper.target_not_found():
                graph.action("click", target="fallback-btn")
        
        # Verify the branch was created
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 1

    def test_as_binding_does_not_suppress_exceptions(self):
        """As-binding pattern should not suppress exceptions."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        with pytest.raises(RuntimeError):
            with graph.action("click", target="submit-btn") as helper:
                raise RuntimeError("Test exception")


# =============================================================================
# Task 9.8: Test action_obj property returns correct Action
# =============================================================================

class TestActionObjProperty:
    """Tests for ActionChainHelper.action_obj property."""

    def test_action_obj_returns_action(self):
        """action_obj property should return an Action object."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        assert isinstance(helper.action_obj, Action)

    def test_action_obj_has_correct_type(self):
        """action_obj should have the correct action type."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        assert helper.action_obj.type == "click"

    def test_action_obj_has_correct_target(self):
        """action_obj should have the correct target."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        assert helper.action_obj.target == "submit-btn"

    def test_action_obj_has_correct_target_spec(self):
        """action_obj should have the correct TargetSpec target."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@id='submit']")
        helper = graph.action("click", target=target)
        
        assert helper.action_obj.target is target
        assert helper.action_obj.target.strategy == TargetStrategy.XPATH
        assert helper.action_obj.target.value == "//button[@id='submit']"

    def test_action_obj_has_correct_args(self):
        """action_obj should have the correct args."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("input_text", target="input-field", args={"text": "hello"})
        
        assert helper.action_obj.args == {"text": "hello"}

    def test_action_obj_has_correct_id(self):
        """action_obj should have the correct action ID."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn", action_id="my_custom_id")
        
        assert helper.action_obj.id == "my_custom_id"

    def test_action_obj_has_auto_generated_id(self):
        """action_obj should have auto-generated ID when not provided."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        assert helper.action_obj.id is not None
        assert helper.action_obj.id.startswith("action_")

    def test_action_obj_has_correct_output(self):
        """action_obj should have the correct output variable."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn", output="result_var")
        
        assert helper.action_obj.output == "result_var"

    def test_action_obj_has_correct_timeout(self):
        """action_obj should have the correct timeout."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn", timeout=30.0)
        
        assert helper.action_obj.timeout == 30.0

    def test_action_obj_has_correct_no_action_if_target_not_found(self):
        """action_obj should have the correct no_action_if_target_not_found flag."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn", no_action_if_target_not_found=True)
        
        assert helper.action_obj.no_action_if_target_not_found is True


# =============================================================================
# Task 9.9: Test method forwarding (action(), condition(), loop(), execute())
# =============================================================================

class TestMethodForwarding:
    """Tests for method forwarding in ActionChainHelper."""

    # -------------------------------------------------------------------------
    # Test action() forwarding
    # -------------------------------------------------------------------------

    def test_action_forwarding_returns_action_chain_helper(self):
        """ActionChainHelper.action() should return ActionChainHelper."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper1 = graph.action("click", target="btn1")
        helper2 = helper1.action("click", target="btn2")
        
        assert isinstance(helper2, ActionChainHelper)

    def test_action_forwarding_creates_new_action(self):
        """ActionChainHelper.action() should create a new action in the graph."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper1 = graph.action("click", target="btn1")
        helper2 = helper1.action("input_text", target="input1", args={"text": "hello"})
        
        # Verify both actions were created
        assert helper1.action_obj.type == "click"
        assert helper2.action_obj.type == "input_text"
        assert helper2.action_obj.args == {"text": "hello"}

    def test_action_forwarding_chain(self):
        """ActionChainHelper.action() should support chaining multiple actions."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = (
            graph.action("click", target="btn1")
            .action("input_text", target="input1", args={"text": "hello"})
            .action("click", target="btn2")
        )
        
        assert isinstance(helper, ActionChainHelper)
        assert helper.action_obj.type == "click"
        assert helper.action_obj.target == "btn2"

    # -------------------------------------------------------------------------
    # Test condition() forwarding
    # -------------------------------------------------------------------------

    def test_condition_forwarding_returns_condition_context(self):
        """ActionChainHelper.condition() should return ConditionContext."""
        from science_modeling_tools.automation.schema.action_graph import ConditionContext
        
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        condition = helper.condition(lambda r: r.success if hasattr(r, 'success') else True)
        
        assert isinstance(condition, ConditionContext)

    def test_condition_forwarding_works_with_context_manager(self):
        """ActionChainHelper.condition() should work with context manager syntax."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        
        with helper.condition(lambda r: True) as branch:
            with branch.if_true():
                graph.action("click", target="success-btn")
            with branch.if_false():
                graph.action("click", target="retry-btn")
        
        # Should not raise any exceptions

    # -------------------------------------------------------------------------
    # Test loop() forwarding
    # -------------------------------------------------------------------------

    def test_loop_forwarding_returns_action_graph(self):
        """ActionChainHelper.loop() should return ActionGraph."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        # loop() requires a condition callable as first argument
        result = helper.loop(condition=lambda r: False, max_loop=3)
        
        assert isinstance(result, ActionGraph)

    def test_loop_forwarding_is_same_graph(self):
        """ActionChainHelper.loop() should return the same graph instance."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        # loop() requires a condition callable as first argument
        result = helper.loop(condition=lambda r: False, max_loop=3)
        
        assert result is graph

    # -------------------------------------------------------------------------
    # Test execute() forwarding
    # -------------------------------------------------------------------------

    def test_execute_forwarding_executes_graph(self):
        """ActionChainHelper.execute() should execute the graph."""
        execution_log = []
        
        def mock_executor(**kwargs):
            execution_log.append(kwargs.get('action_type'))
            return "executed"
        
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        result = helper.execute()
        
        # Verify execution happened
        assert "click" in execution_log

    def test_execute_forwarding_returns_execution_result(self):
        """ActionChainHelper.execute() should return ExecutionResult."""
        from science_modeling_tools.automation.schema.common import ExecutionResult
        
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="btn1")
        result = helper.execute()
        
        assert isinstance(result, ExecutionResult)

    def test_execute_forwarding_with_variables(self):
        """ActionChainHelper.execute() should accept initial_variables."""
        execution_log = []
        
        def mock_executor(**kwargs):
            execution_log.append(kwargs)
            return "executed"
        
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="{button_id}")
        # execute() takes initial_variables as a dict, not keyword arguments
        result = helper.execute(initial_variables={"button_id": "submit-btn"})
        
        # Verify execution happened with variable substitution
        assert len(execution_log) > 0



# =============================================================================
# Task 10: Unit Tests for ActionGraph Builder
# =============================================================================


# =============================================================================
# Task 10.1: Test action() inside target_not_found() adds actions to branch list
# =============================================================================

class TestActionInsideTargetNotFoundAddsToBranchList:
    """Tests that actions defined inside target_not_found() are added to the branch list."""

    def test_single_action_inside_target_not_found_added_to_branch(self):
        """A single action inside target_not_found() should be added to branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            graph.action("click", target="fallback-btn")
        
        # Verify action was added to branch list
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 1
        assert helper.action_obj.target_not_found_actions[0].type == "click"
        assert helper.action_obj.target_not_found_actions[0].target == "fallback-btn"

    def test_multiple_actions_inside_target_not_found_added_to_branch(self):
        """Multiple actions inside target_not_found() should all be added to branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            graph.action("click", target="fallback-btn-1")
            graph.action("input_text", target="input-field", args={"text": "hello"})
            graph.action("click", target="fallback-btn-2")
        
        # Verify all actions were added to branch list
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 3
        assert helper.action_obj.target_not_found_actions[0].target == "fallback-btn-1"
        assert helper.action_obj.target_not_found_actions[1].type == "input_text"
        assert helper.action_obj.target_not_found_actions[2].target == "fallback-btn-2"

    def test_action_with_target_spec_inside_target_not_found(self):
        """Actions with TargetSpec inside target_not_found() should be added to branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        main_target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        fallback_target = TargetSpec(strategy=TargetStrategy.XPATH, value="//button[@class='fallback']")
        
        helper = graph.action("click", target=main_target)
        
        with helper.target_not_found():
            graph.action("click", target=fallback_target)
        
        # Verify action was added with correct TargetSpec
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 1
        branch_action = helper.action_obj.target_not_found_actions[0]
        assert branch_action.target.strategy == TargetStrategy.XPATH
        assert branch_action.target.value == "//button[@class='fallback']"

    def test_action_inside_target_not_found_preserves_all_attributes(self):
        """Actions inside target_not_found() should preserve all their attributes."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            graph.action(
                "input_text",
                target="input-field",
                args={"text": "hello"},
                action_id="custom_fallback_id",
                output="result_var",
                timeout=30.0,
                no_action_if_target_not_found=True
            )
        
        # Verify all attributes were preserved
        branch_action = helper.action_obj.target_not_found_actions[0]
        assert branch_action.type == "input_text"
        assert branch_action.target == "input-field"
        assert branch_action.args == {"text": "hello"}
        assert branch_action.id == "custom_fallback_id"
        assert branch_action.output == "result_var"
        assert branch_action.timeout == 30.0
        assert branch_action.no_action_if_target_not_found is True

    def test_action_via_context_action_method_added_to_branch(self):
        """Actions added via context.action() should be added to branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            ctx.action("click", target="fallback-btn")
        
        # Verify action was added to branch list
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 1
        assert helper.action_obj.target_not_found_actions[0].target == "fallback-btn"


# =============================================================================
# Task 10.2: Test action() outside target_not_found() adds actions to current node
# =============================================================================

class TestActionOutsideTargetNotFoundAddsToCurrentNode:
    """Tests that actions defined outside target_not_found() are added to current node."""

    def test_action_before_target_not_found_added_to_node(self):
        """Actions before target_not_found() should be added to current node."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Add action before target_not_found
        graph.action("click", target="first-btn")
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            graph.action("click", target="fallback-btn")
        
        # Verify first action is in current node, not in branch
        current_node = graph._current_node
        assert len(current_node._actions) == 2  # first-btn and submit-btn
        assert current_node._actions[0].target == "first-btn"
        assert current_node._actions[1].target == "submit-btn"

    def test_action_after_target_not_found_added_to_node(self):
        """Actions after target_not_found() context should be added to current node."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            graph.action("click", target="fallback-btn")
        
        # Add action after target_not_found context
        graph.action("click", target="after-btn")
        
        # Verify after action is in current node, not in branch
        current_node = graph._current_node
        assert len(current_node._actions) == 2  # submit-btn and after-btn
        assert current_node._actions[0].target == "submit-btn"
        assert current_node._actions[1].target == "after-btn"
        
        # Verify branch only has fallback action
        assert len(helper.action_obj.target_not_found_actions) == 1
        assert helper.action_obj.target_not_found_actions[0].target == "fallback-btn"

    def test_actions_not_in_context_not_added_to_branch(self):
        """Actions outside any target_not_found() context should not be in any branch."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Add multiple actions, some with target_not_found
        graph.action("click", target="btn1")
        
        helper = graph.action("click", target="btn2")
        with helper.target_not_found():
            graph.action("click", target="fallback-for-btn2")
        
        graph.action("click", target="btn3")
        
        # Verify btn1 and btn3 are in current node
        current_node = graph._current_node
        assert len(current_node._actions) == 3
        assert current_node._actions[0].target == "btn1"
        assert current_node._actions[1].target == "btn2"
        assert current_node._actions[2].target == "btn3"
        
        # Verify only btn2 has a branch
        assert current_node._actions[0].target_not_found_actions is None
        assert current_node._actions[1].target_not_found_actions is not None
        assert current_node._actions[2].target_not_found_actions is None


# =============================================================================
# Task 10.3: Test nested target_not_found() contexts (actions go to innermost)
# =============================================================================

class TestNestedTargetNotFoundContexts:
    """Tests that nested target_not_found() contexts work correctly."""

    def test_nested_target_not_found_actions_go_to_innermost(self):
        """Actions in nested target_not_found() should go to innermost context."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Outer action with target_not_found
        outer_helper = graph.action("click", target="outer-btn")
        
        with outer_helper.target_not_found():
            # Inner action with its own target_not_found
            inner_helper = graph.action("click", target="inner-btn")
            
            with inner_helper.target_not_found():
                # This should go to inner's branch
                graph.action("click", target="innermost-fallback")
            
            # This should go to outer's branch
            graph.action("click", target="outer-fallback")
        
        # Verify outer branch has inner-btn and outer-fallback
        assert outer_helper.action_obj.target_not_found_actions is not None
        assert len(outer_helper.action_obj.target_not_found_actions) == 2
        assert outer_helper.action_obj.target_not_found_actions[0].target == "inner-btn"
        assert outer_helper.action_obj.target_not_found_actions[1].target == "outer-fallback"
        
        # Verify inner branch has innermost-fallback
        inner_action = outer_helper.action_obj.target_not_found_actions[0]
        assert inner_action.target_not_found_actions is not None
        assert len(inner_action.target_not_found_actions) == 1
        assert inner_action.target_not_found_actions[0].target == "innermost-fallback"

    def test_deeply_nested_target_not_found_contexts(self):
        """Deeply nested target_not_found() contexts should work correctly."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Level 1
        level1_helper = graph.action("click", target="level1-btn")
        
        with level1_helper.target_not_found():
            # Level 2
            level2_helper = graph.action("click", target="level2-btn")
            
            with level2_helper.target_not_found():
                # Level 3
                level3_helper = graph.action("click", target="level3-btn")
                
                with level3_helper.target_not_found():
                    # Deepest level
                    graph.action("click", target="deepest-fallback")
        
        # Verify level 1 branch
        assert level1_helper.action_obj.target_not_found_actions is not None
        assert len(level1_helper.action_obj.target_not_found_actions) == 1
        
        # Verify level 2 branch
        level2_action = level1_helper.action_obj.target_not_found_actions[0]
        assert level2_action.target == "level2-btn"
        assert level2_action.target_not_found_actions is not None
        assert len(level2_action.target_not_found_actions) == 1
        
        # Verify level 3 branch
        level3_action = level2_action.target_not_found_actions[0]
        assert level3_action.target == "level3-btn"
        assert level3_action.target_not_found_actions is not None
        assert len(level3_action.target_not_found_actions) == 1
        
        # Verify deepest level
        deepest_action = level3_action.target_not_found_actions[0]
        assert deepest_action.target == "deepest-fallback"

    def test_sibling_nested_contexts_independent(self):
        """Sibling nested target_not_found() contexts should be independent."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        outer_helper = graph.action("click", target="outer-btn")
        
        with outer_helper.target_not_found():
            # First sibling
            sibling1_helper = graph.action("click", target="sibling1-btn")
            with sibling1_helper.target_not_found():
                graph.action("click", target="sibling1-fallback")
            
            # Second sibling
            sibling2_helper = graph.action("click", target="sibling2-btn")
            with sibling2_helper.target_not_found():
                graph.action("click", target="sibling2-fallback")
        
        # Verify outer branch has both siblings
        assert len(outer_helper.action_obj.target_not_found_actions) == 2
        
        # Verify sibling 1 has its own branch
        sibling1 = outer_helper.action_obj.target_not_found_actions[0]
        assert sibling1.target == "sibling1-btn"
        assert len(sibling1.target_not_found_actions) == 1
        assert sibling1.target_not_found_actions[0].target == "sibling1-fallback"
        
        # Verify sibling 2 has its own branch
        sibling2 = outer_helper.action_obj.target_not_found_actions[1]
        assert sibling2.target == "sibling2-btn"
        assert len(sibling2.target_not_found_actions) == 1
        assert sibling2.target_not_found_actions[0].target == "sibling2-fallback"


# =============================================================================
# Task 10.4: Test context restoration after exception in branch definition
# =============================================================================

class TestContextRestorationAfterException:
    """Tests that context is properly restored after an exception during branch definition."""

    def test_context_restored_after_exception(self):
        """Graph context should be restored after exception in target_not_found()."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        # Verify stack is empty before
        assert len(graph._action_branch_stack) == 0
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found():
                graph.action("click", target="fallback-btn")
                raise RuntimeError("Test exception")
        
        # Verify stack is restored (empty) after exception
        assert len(graph._action_branch_stack) == 0

    def test_actions_after_exception_go_to_current_node(self):
        """Actions after exception should go to current node, not branch."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found():
                graph.action("click", target="fallback-btn")
                raise RuntimeError("Test exception")
        
        # Add action after exception
        graph.action("click", target="after-exception-btn")
        
        # Verify action was added to current node
        current_node = graph._current_node
        assert len(current_node._actions) == 2  # submit-btn and after-exception-btn
        assert current_node._actions[1].target == "after-exception-btn"

    def test_nested_context_restored_after_inner_exception(self):
        """Nested context should be restored after exception in inner context."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        outer_helper = graph.action("click", target="outer-btn")
        
        with outer_helper.target_not_found():
            inner_helper = graph.action("click", target="inner-btn")
            
            with pytest.raises(RuntimeError):
                with inner_helper.target_not_found():
                    graph.action("click", target="inner-fallback")
                    raise RuntimeError("Inner exception")
            
            # This should still go to outer's branch
            graph.action("click", target="after-inner-exception")
        
        # Verify outer branch has inner-btn and after-inner-exception
        assert len(outer_helper.action_obj.target_not_found_actions) == 2
        assert outer_helper.action_obj.target_not_found_actions[0].target == "inner-btn"
        assert outer_helper.action_obj.target_not_found_actions[1].target == "after-inner-exception"
        
        # Verify inner branch was cleaned up
        inner_action = outer_helper.action_obj.target_not_found_actions[0]
        assert inner_action.target_not_found_actions is None

    def test_exception_propagates_after_context_restoration(self):
        """Exception should propagate after context is restored."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(ValueError) as exc_info:
            with helper.target_not_found():
                raise ValueError("Custom error message")
        
        assert "Custom error message" in str(exc_info.value)


# =============================================================================
# Task 10.5: Test partial state cleanup on exception
# =============================================================================

class TestPartialStateCleanupOnException:
    """Tests that partial state is cleaned up on exception."""

    def test_target_not_found_actions_set_to_none_on_exception(self):
        """target_not_found_actions should be set to None on exception."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found():
                graph.action("click", target="fallback-btn")
                raise RuntimeError("Test exception")
        
        # Verify target_not_found_actions was cleaned up
        assert helper.action_obj.target_not_found_actions is None

    def test_target_not_found_config_set_to_none_on_exception(self):
        """target_not_found_config should be set to None on exception."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found(retry_after_handling=True, max_retries=5):
                raise RuntimeError("Test exception")
        
        # Verify target_not_found_config was cleaned up
        assert helper.action_obj.target_not_found_config is None

    def test_both_fields_cleaned_up_on_exception(self):
        """Both target_not_found_actions and config should be cleaned up on exception."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found(retry_after_handling=True, max_retries=3, retry_delay=1.0):
                graph.action("click", target="fallback-btn")
                raise RuntimeError("Test exception")
        
        # Verify both fields were cleaned up
        assert helper.action_obj.target_not_found_actions is None
        assert helper.action_obj.target_not_found_config is None

    def test_partial_actions_cleaned_up_on_exception(self):
        """Partial actions added before exception should be cleaned up."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found():
                graph.action("click", target="fallback-1")
                graph.action("click", target="fallback-2")
                raise RuntimeError("Test exception")
        
        # Verify all partial state was cleaned up
        assert helper.action_obj.target_not_found_actions is None

    def test_successful_context_preserves_state(self):
        """Successful context exit should preserve state (not clean up)."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found(retry_after_handling=True, max_retries=5, retry_delay=2.0):
            graph.action("click", target="fallback-btn")
        
        # Verify state was preserved
        assert helper.action_obj.target_not_found_actions is not None
        assert len(helper.action_obj.target_not_found_actions) == 1
        assert helper.action_obj.target_not_found_config is not None
        assert helper.action_obj.target_not_found_config["retry_after_handling"] is True
        assert helper.action_obj.target_not_found_config["max_retries"] == 5
        assert helper.action_obj.target_not_found_config["retry_delay"] == 2.0


# =============================================================================
# Task 10.6: Test _action_branch_stack management
# =============================================================================

class TestActionBranchStackManagement:
    """Tests for _action_branch_stack management in ActionGraph."""

    def test_stack_empty_initially(self):
        """_action_branch_stack should be empty initially."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        assert len(graph._action_branch_stack) == 0

    def test_stack_pushed_on_context_enter(self):
        """_action_branch_stack should have entry pushed on context enter."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            # Inside context, stack should have one entry
            assert len(graph._action_branch_stack) == 1

    def test_stack_popped_on_context_exit(self):
        """_action_branch_stack should have entry popped on context exit."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            assert len(graph._action_branch_stack) == 1
        
        # After context, stack should be empty
        assert len(graph._action_branch_stack) == 0

    def test_nested_contexts_push_multiple_entries(self):
        """Nested contexts should push multiple entries to stack."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        outer_helper = graph.action("click", target="outer-btn")
        
        with outer_helper.target_not_found():
            assert len(graph._action_branch_stack) == 1
            
            inner_helper = graph.action("click", target="inner-btn")
            
            with inner_helper.target_not_found():
                # Two nested contexts
                assert len(graph._action_branch_stack) == 2
            
            # After inner context exits
            assert len(graph._action_branch_stack) == 1
        
        # After outer context exits
        assert len(graph._action_branch_stack) == 0

    def test_stack_entry_is_branch_actions_list(self):
        """Stack entry should be the same list as parent action's branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            # Stack entry should be same object as action's branch list
            assert graph._action_branch_stack[-1] is helper.action_obj.target_not_found_actions

    def test_actions_added_to_stack_top(self):
        """Actions should be added to the list at top of stack."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found():
            stack_top = graph._action_branch_stack[-1]
            assert len(stack_top) == 0
            
            graph.action("click", target="fallback-btn")
            
            # Action should be in stack top
            assert len(stack_top) == 1
            assert stack_top[0].target == "fallback-btn"

    def test_stack_popped_on_exception(self):
        """Stack should be popped even when exception occurs."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with pytest.raises(RuntimeError):
            with helper.target_not_found():
                assert len(graph._action_branch_stack) == 1
                raise RuntimeError("Test exception")
        
        # Stack should be empty after exception
        assert len(graph._action_branch_stack) == 0

    def test_push_and_pop_methods_work_correctly(self):
        """_push_action_branch_context and _pop_action_branch_context should work correctly."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Test push
        branch_list = []
        graph._push_action_branch_context(branch_list)
        assert len(graph._action_branch_stack) == 1
        assert graph._action_branch_stack[-1] is branch_list
        
        # Test pop
        graph._pop_action_branch_context()
        assert len(graph._action_branch_stack) == 0

    def test_pop_on_empty_stack_is_safe(self):
        """_pop_action_branch_context should be safe to call on empty stack."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Should not raise
        graph._pop_action_branch_context()
        assert len(graph._action_branch_stack) == 0


# =============================================================================
# Task 10.7: Test TargetNotFoundContext method forwarding (action(), condition())
# =============================================================================

class TestTargetNotFoundContextMethodForwarding:
    """Tests for method forwarding in TargetNotFoundContext."""

    # -------------------------------------------------------------------------
    # Test action() forwarding
    # -------------------------------------------------------------------------

    def test_context_action_returns_action_chain_helper(self):
        """TargetNotFoundContext.action() should return ActionChainHelper."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            result = ctx.action("click", target="fallback-btn")
            assert isinstance(result, ActionChainHelper)

    def test_context_action_adds_to_branch(self):
        """TargetNotFoundContext.action() should add action to branch list."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            ctx.action("click", target="fallback-btn")
        
        assert len(helper.action_obj.target_not_found_actions) == 1
        assert helper.action_obj.target_not_found_actions[0].target == "fallback-btn"

    def test_context_action_supports_all_parameters(self):
        """TargetNotFoundContext.action() should support all action parameters."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            ctx.action(
                "input_text",
                target=TargetSpec(strategy=TargetStrategy.ID, value="input-field"),
                args={"text": "hello"},
                action_id="custom_id",
                output="result",
                timeout=30.0
            )
        
        branch_action = helper.action_obj.target_not_found_actions[0]
        assert branch_action.type == "input_text"
        assert branch_action.target.strategy == TargetStrategy.ID
        assert branch_action.args == {"text": "hello"}
        assert branch_action.id == "custom_id"
        assert branch_action.output == "result"
        assert branch_action.timeout == 30.0

    def test_context_action_chaining(self):
        """TargetNotFoundContext.action() should support chaining."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            ctx.action("click", target="btn1").action("click", target="btn2")
        
        # Both actions should be in branch
        assert len(helper.action_obj.target_not_found_actions) == 2
        assert helper.action_obj.target_not_found_actions[0].target == "btn1"
        assert helper.action_obj.target_not_found_actions[1].target == "btn2"

    # -------------------------------------------------------------------------
    # Test condition() forwarding
    # -------------------------------------------------------------------------

    def test_context_condition_returns_condition_context(self):
        """TargetNotFoundContext.condition() should return ConditionContext."""
        from science_modeling_tools.automation.schema.action_graph import ConditionContext
        
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            result = ctx.condition(lambda r: True)
            assert isinstance(result, ConditionContext)

    def test_context_condition_works_with_branches(self):
        """TargetNotFoundContext.condition() should work with if_true/if_false branches."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            with ctx.condition(lambda r: True) as branch:
                with branch.if_true():
                    graph.action("click", target="success-btn")
                with branch.if_false():
                    graph.action("click", target="retry-btn")
        
        # Should not raise any exceptions
        # The condition creates new nodes, so we just verify no errors

    # -------------------------------------------------------------------------
    # Test that forwarding uses graph methods
    # -------------------------------------------------------------------------

    def test_context_action_uses_graph_action(self):
        """TargetNotFoundContext.action() should use graph.action() internally."""
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            # Both should produce same result
            result1 = ctx.action("click", target="via-context")
            result2 = graph.action("click", target="via-graph")
        
        # Both should be in branch
        assert len(helper.action_obj.target_not_found_actions) == 2

    def test_context_condition_uses_graph_condition(self):
        """TargetNotFoundContext.condition() should use graph.condition() internally."""
        from science_modeling_tools.automation.schema.action_graph import ConditionContext
        
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        helper = graph.action("click", target="submit-btn")
        
        with helper.target_not_found() as ctx:
            # Both should return ConditionContext
            result1 = ctx.condition(lambda r: True)
            result2 = graph.condition(lambda r: False)
            
            assert isinstance(result1, ConditionContext)
            assert isinstance(result2, ConditionContext)


# =============================================================================
# Task 11: Unit Tests for Execution
# =============================================================================

# Import ActionNode and related classes for execution testing
from science_modeling_tools.automation.schema.action_node import ActionNode
from science_modeling_tools.automation.schema.action_metadata import ActionMetadataRegistry
import time


# Custom exception classes for testing (to avoid importing from WebAgent)
class ElementNotFoundError(Exception):
    """Custom ElementNotFoundError for testing."""
    pass


class ElementNotFoundException(Exception):
    """Custom ElementNotFoundException for testing."""
    pass


class CustomElementNotFoundError(ElementNotFoundError):
    """Subclass of ElementNotFoundError for MRO testing."""
    pass


# =============================================================================
# Task 11.1: Test execution with target found (mock executor returning result)
# =============================================================================

class TestExecutionWithTargetFound:
    """Tests that execution succeeds when target is found."""

    def test_execution_returns_result_when_target_found(self):
        """Execution should return ActionResult with success=True when target is found."""
        execution_log = []
        
        def mock_executor(**kwargs):
            execution_log.append(kwargs)
            return "clicked successfully"
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert result.value == "clicked successfully"
        assert len(execution_log) == 1

    def test_execution_with_target_spec_returns_result(self):
        """Execution should return result when using TargetSpec and target is found."""
        def mock_executor(**kwargs):
            return {"element": "found", "action": kwargs.get("action_type")}
        
        action = Action(
            id="test_action",
            type="click",
            target=TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert result.value["element"] == "found"
        assert result.value["action"] == "click"

    def test_execution_stores_result_in_context(self):
        """Execution should store result in context variables."""
        def mock_executor(**kwargs):
            return "result_value"
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            output="my_result"
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert context.variables.get("my_result") == "result_value"
        assert context.variables.get("_") == "result_value"


# =============================================================================
# Task 11.2: Test execution with target not found, no branch (re-raises)
# =============================================================================

class TestExecutionTargetNotFoundNoBranch:
    """Tests that exception is re-raised when target not found and no branch exists."""

    def test_element_not_found_error_reraises_without_branch(self):
        """ElementNotFoundError should be re-raised when no target_not_found branch exists."""
        def mock_executor(**kwargs):
            raise ElementNotFoundError("Element not found")
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(ElementNotFoundError) as exc_info:
            node.run(context)
        
        assert "Element not found" in str(exc_info.value)

    def test_element_not_found_exception_reraises_without_branch(self):
        """ElementNotFoundException should be re-raised when no target_not_found branch exists."""
        def mock_executor(**kwargs):
            raise ElementNotFoundException("Element not found")
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(ElementNotFoundException):
            node.run(context)


# =============================================================================
# Task 11.3: Test execution with target not found, branch exists, retry_after_handling=False
# =============================================================================

class TestExecutionTargetNotFoundBranchNoRetry:
    """Tests that branch executes and returns success when retry_after_handling=False."""

    def test_branch_executes_on_element_not_found(self):
        """Branch actions should execute when ElementNotFoundError is raised."""
        execution_log = []
        
        def mock_executor(**kwargs):
            execution_log.append(kwargs.get("action_target"))
            if kwargs.get("action_target") == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        # Create action with target_not_found branch
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        assert "submit-btn" in execution_log
        assert "fallback-btn" in execution_log

    def test_branch_returns_success_without_retry(self):
        """Branch should return success immediately when retry_after_handling=False."""
        call_count = 0
        
        def mock_executor(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("action_target") == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        # Should only call executor twice: once for main action, once for fallback
        assert call_count == 2

    def test_multiple_branch_actions_execute_in_order(self):
        """Multiple branch actions should execute in order."""
        execution_order = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            execution_order.append(target)
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return f"executed {target}"
        
        fallback_actions = [
            Action(id="fallback_1", type="click", target="btn1"),
            Action(id="fallback_2", type="click", target="btn2"),
            Action(id="fallback_3", type="click", target="btn3"),
        ]
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=fallback_actions,
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert execution_order == ["submit-btn", "btn1", "btn2", "btn3"]


# =============================================================================
# Task 11.4: Test execution with target not found, branch exists, retry_after_handling=True
# =============================================================================

class TestExecutionTargetNotFoundBranchWithRetry:
    """Tests that branch executes and retries when retry_after_handling=True."""

    def test_retry_succeeds_after_branch_execution(self):
        """Retry should succeed after branch execution makes target available."""
        call_count = 0
        
        def mock_executor(**kwargs):
            nonlocal call_count
            call_count += 1
            target = kwargs.get("action_target")
            if target == "submit-btn":
                # First call fails, second succeeds
                if call_count == 1:
                    raise ElementNotFoundError("Element not found")
                return "retry succeeded"
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        assert result.success is True
        assert result.metadata.get("retry_succeeded") is True
        assert result.value == "retry succeeded"

    def test_retry_executes_branch_multiple_times(self):
        """Branch should execute multiple times when retry keeps failing."""
        branch_execution_count = 0
        main_execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal branch_execution_count, main_execution_count
            target = kwargs.get("action_target")
            if target == "submit-btn":
                main_execution_count += 1
                raise ElementNotFoundError("Element not found")
            branch_execution_count += 1
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 2,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError):
            node.run(context)
        
        # Branch should execute 3 times (1 initial + 2 retries)
        assert branch_execution_count == 3


# =============================================================================
# Task 11.5: Test max_retries enforcement (including max_retries=0 edge case)
# =============================================================================

class TestMaxRetriesEnforcement:
    """Tests that max_retries is properly enforced."""

    def test_max_retries_zero_executes_branch_once(self):
        """With max_retries=0, branch should execute once then raise TargetNotFoundError."""
        branch_execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal branch_execution_count
            target = kwargs.get("action_target")
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            branch_execution_count += 1
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 0,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError) as exc_info:
            node.run(context)
        
        # Branch should execute exactly once (1 initial, 0 retries)
        assert branch_execution_count == 1
        assert exc_info.value.attempt_count == 1
        assert exc_info.value.max_retries == 0

    def test_max_retries_three_executes_branch_four_times(self):
        """With max_retries=3, branch should execute 4 times (1 initial + 3 retries)."""
        branch_execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal branch_execution_count
            target = kwargs.get("action_target")
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            branch_execution_count += 1
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError) as exc_info:
            node.run(context)
        
        # Branch should execute 4 times (1 initial + 3 retries)
        assert branch_execution_count == 4
        assert exc_info.value.attempt_count == 4
        assert exc_info.value.max_retries == 3

    def test_target_not_found_error_has_correct_attributes(self):
        """TargetNotFoundError should have correct attributes when raised."""
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            # Only raise for the main target, not the fallback
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        target = TargetSpec(strategy=TargetStrategy.ID, value="submit-btn")
        action = Action(
            id="test_action",
            type="click",
            target=target,
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 2,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError) as exc_info:
            node.run(context)
        
        assert exc_info.value.action_type == "click"
        assert exc_info.value.target is target
        assert exc_info.value.attempt_count == 3
        assert exc_info.value.max_retries == 2


# =============================================================================
# Task 11.6: Test retry_delay timing (approximate)
# =============================================================================

class TestRetryDelayTiming:
    """Tests that retry_delay is approximately respected."""

    def test_retry_delay_is_respected(self):
        """Retry should wait approximately retry_delay seconds between attempts."""
        timestamps = []
        
        def mock_executor(**kwargs):
            timestamps.append(time.time())
            target = kwargs.get("action_target")
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 2,
                "retry_delay": 0.1  # 100ms delay
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError):
            node.run(context)
        
        # Check that delays were approximately respected
        # timestamps: [main1, fallback1, main2, fallback2, main3, fallback3]
        # Delays should be between main attempts (after fallback, before retry)
        # We expect at least 2 delays of ~0.1s each
        total_time = timestamps[-1] - timestamps[0]
        # With 2 retries and 0.1s delay each, total should be at least 0.2s
        assert total_time >= 0.15  # Allow some tolerance

    def test_zero_retry_delay_is_fast(self):
        """With retry_delay=0, retries should happen immediately."""
        timestamps = []
        
        def mock_executor(**kwargs):
            timestamps.append(time.time())
            target = kwargs.get("action_target")
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 2,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError):
            node.run(context)
        
        # With no delay, total time should be very small
        total_time = timestamps[-1] - timestamps[0]
        assert total_time < 0.5  # Should be much faster than 0.5s


# =============================================================================
# Task 11.7: Test integration with no_action_if_target_not_found flag
# =============================================================================

class TestNoActionIfTargetNotFoundIntegration:
    """Tests that no_action_if_target_not_found flag still works when no branch exists."""

    def test_no_action_if_target_not_found_skips_action(self):
        """Action should be skipped when no_action_if_target_not_found=True and no branch."""
        execution_log = []
        
        def mock_executor(**kwargs):
            execution_log.append(kwargs)
            # Simulate target not found by returning None or similar
            if kwargs.get("no_action_if_target_not_found"):
                return None  # Executor handles skip internally
            raise ElementNotFoundError("Element not found")
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            no_action_if_target_not_found=True
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Should succeed (action was skipped, not failed)
        assert result.success is True
        assert len(execution_log) == 1
        assert execution_log[0]["no_action_if_target_not_found"] is True

    def test_branch_takes_precedence_over_no_action_flag(self):
        """Branch should execute even when no_action_if_target_not_found=True."""
        execution_log = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            execution_log.append(target)
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return "fallback executed"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            no_action_if_target_not_found=True,  # This flag is set
            target_not_found_actions=[fallback_action],  # But branch exists
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Branch should have executed
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        assert "fallback-btn" in execution_log


# =============================================================================
# Task 11.8: Test graph reusability (execute twice)
# =============================================================================

class TestGraphReusability:
    """Tests that graph can be executed multiple times."""

    def test_action_node_can_execute_twice(self):
        """ActionNode should be reusable for multiple executions."""
        execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal execution_count
            execution_count += 1
            return f"execution_{execution_count}"
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        # First execution
        context1 = ExecutionRuntime()
        result1 = node.run(context1)
        
        # Second execution
        context2 = ExecutionRuntime()
        result2 = node.run(context2)
        
        assert result1.success is True
        assert result2.success is True
        assert result1.value == "execution_1"
        assert result2.value == "execution_2"
        assert execution_count == 2

    def test_action_node_with_branch_can_execute_twice(self):
        """ActionNode with target_not_found branch should be reusable."""
        execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal execution_count
            execution_count += 1
            target = kwargs.get("action_target")
            if target == "submit-btn":
                raise ElementNotFoundError("Element not found")
            return f"fallback_{execution_count}"
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        # First execution
        context1 = ExecutionRuntime()
        result1 = node.run(context1)
        
        # Second execution
        context2 = ExecutionRuntime()
        result2 = node.run(context2)
        
        assert result1.success is True
        assert result2.success is True
        assert result1.metadata.get("branch_executed") is True
        assert result2.metadata.get("branch_executed") is True


# =============================================================================
# Task 11.9: Test empty branch execution (retry_after_handling=True and False)
# =============================================================================

class TestEmptyBranchExecution:
    """Tests that empty branches work correctly.
    
    Note: Empty branches (target_not_found_actions=[]) are treated the same as
    no branch (target_not_found_actions=None) because an empty list is falsy.
    This means the exception is re-raised rather than handled.
    """

    def test_empty_branch_reraises_exception(self):
        """Empty branch should re-raise exception (treated same as no branch)."""
        def mock_executor(**kwargs):
            raise ElementNotFoundError("Element not found")
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[],  # Empty branch - treated as no branch
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        # Empty branch is treated as no branch, so exception is re-raised
        with pytest.raises(ElementNotFoundError):
            node.run(context)

    def test_empty_branch_with_retry_true_also_reraises(self):
        """Empty branch with retry_after_handling=True should also re-raise."""
        retry_count = 0
        
        def mock_executor(**kwargs):
            nonlocal retry_count
            retry_count += 1
            raise ElementNotFoundError("Element not found")
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[],  # Empty branch - treated as no branch
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": 2,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        # Empty branch is treated as no branch, so exception is re-raised
        with pytest.raises(ElementNotFoundError):
            node.run(context)
        
        # Should only have tried once (no retry because no branch)
        assert retry_count == 1


# =============================================================================
# Task 11.10: Test executor raising non-matching exception (propagates)
# =============================================================================

class TestNonMatchingExceptionPropagates:
    """Tests that non-matching exceptions propagate unchanged."""

    def test_value_error_propagates_unchanged(self):
        """ValueError should propagate unchanged, not trigger branch."""
        branch_executed = False
        
        def mock_executor(**kwargs):
            raise ValueError("Invalid argument")
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(ValueError) as exc_info:
            node.run(context)
        
        assert "Invalid argument" in str(exc_info.value)

    def test_file_not_found_error_propagates_unchanged(self):
        """FileNotFoundError should propagate unchanged (not element-not-found)."""
        def mock_executor(**kwargs):
            raise FileNotFoundError("File not found")
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(FileNotFoundError):
            node.run(context)

    def test_runtime_error_propagates_unchanged(self):
        """RuntimeError should propagate unchanged, not trigger branch."""
        def mock_executor(**kwargs):
            raise RuntimeError("Something went wrong")
        
        fallback_action = Action(
            id="fallback_action",
            type="click",
            target="fallback-btn"
        )
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn",
            target_not_found_actions=[fallback_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(RuntimeError):
            node.run(context)


# =============================================================================
# Task 11.11-11.14: Test _is_element_not_found_error() method
# =============================================================================

class TestIsElementNotFoundError:
    """Tests for ActionNode._is_element_not_found_error() method."""

    def _create_test_node(self):
        """Helper to create a test ActionNode."""
        def mock_executor(**kwargs):
            return "executed"
        
        action = Action(
            id="test_action",
            type="click",
            target="submit-btn"
        )
        
        return ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )

    # -------------------------------------------------------------------------
    # Task 11.11: Test with ElementNotFoundError exact name match
    # -------------------------------------------------------------------------

    def test_element_not_found_error_exact_match(self):
        """_is_element_not_found_error should return True for ElementNotFoundError."""
        node = self._create_test_node()
        error = ElementNotFoundError("Element not found")
        
        assert node._is_element_not_found_error(error) is True

    # -------------------------------------------------------------------------
    # Task 11.12: Test with ElementNotFoundException exact name match
    # -------------------------------------------------------------------------

    def test_element_not_found_exception_exact_match(self):
        """_is_element_not_found_error should return True for ElementNotFoundException."""
        node = self._create_test_node()
        error = ElementNotFoundException("Element not found")
        
        assert node._is_element_not_found_error(error) is True

    # -------------------------------------------------------------------------
    # Task 11.13: Test with subclass (MRO check)
    # -------------------------------------------------------------------------

    def test_subclass_of_element_not_found_error(self):
        """_is_element_not_found_error should return True for subclass of ElementNotFoundError."""
        node = self._create_test_node()
        error = CustomElementNotFoundError("Custom element not found")
        
        assert node._is_element_not_found_error(error) is True

    def test_target_not_found_error_exact_match(self):
        """_is_element_not_found_error should return True for TargetNotFoundError."""
        node = self._create_test_node()
        error = TargetNotFoundError(
            action_type="click",
            target="submit-btn",
            attempt_count=1,
            max_retries=0
        )
        
        assert node._is_element_not_found_error(error) is True

    # -------------------------------------------------------------------------
    # Task 11.14: Test returns False for FileNotFoundError
    # -------------------------------------------------------------------------

    def test_file_not_found_error_returns_false(self):
        """_is_element_not_found_error should return False for FileNotFoundError."""
        node = self._create_test_node()
        error = FileNotFoundError("File not found")
        
        assert node._is_element_not_found_error(error) is False

    def test_value_error_returns_false(self):
        """_is_element_not_found_error should return False for ValueError."""
        node = self._create_test_node()
        error = ValueError("Invalid value")
        
        assert node._is_element_not_found_error(error) is False

    def test_runtime_error_returns_false(self):
        """_is_element_not_found_error should return False for RuntimeError."""
        node = self._create_test_node()
        error = RuntimeError("Runtime error")
        
        assert node._is_element_not_found_error(error) is False

    def test_key_error_returns_false(self):
        """_is_element_not_found_error should return False for KeyError."""
        node = self._create_test_node()
        error = KeyError("key")
        
        assert node._is_element_not_found_error(error) is False

    def test_generic_exception_returns_false(self):
        """_is_element_not_found_error should return False for generic Exception."""
        node = self._create_test_node()
        error = Exception("Generic error")
        
        assert node._is_element_not_found_error(error) is False

    def test_not_found_in_name_but_not_exact_match_returns_false(self):
        """_is_element_not_found_error should return False for exceptions with 'NotFound' in name but not exact match."""
        node = self._create_test_node()
        
        # Create a custom exception with 'NotFound' in name but not exact match
        class PageNotFoundError(Exception):
            pass
        
        error = PageNotFoundError("Page not found")
        
        # Should return False because it's not an exact match
        assert node._is_element_not_found_error(error) is False
