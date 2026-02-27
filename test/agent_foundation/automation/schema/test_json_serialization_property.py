"""Property-based tests for ActionTypeMetadata JSON serialization.

This module contains property-based tests using hypothesis to verify
JSON serialization round-trip and robustness as specified in the design document.

**Feature: action-metadata-consolidation, Property 6: JSON serialization round-trip**
**Validates: Requirements 7.1**

**Feature: action-metadata-consolidation, Property 7, 8, 9: JSON robustness**
**Validates: Requirements 7.2, 7.3, 7.4**
"""
import sys
import json
from pathlib import Path

# Setup import paths
_current_file = Path(__file__).resolve()
_test_dir = _current_file.parent
while _test_dir.name != 'test' and _test_dir.parent != _test_dir:
    _test_dir = _test_dir.parent
_project_root = _test_dir.parent
_src_dir = _project_root / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
# Add SciencePythonUtils if needed
_workspace_root = _project_root.parent
_rich_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _rich_python_utils_src.exists() and str(_rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_rich_python_utils_src))

from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError
import pytest

from agent_foundation.automation.schema.action_metadata import (
    ActionTypeMetadata,
    ActionMemoryMode,
    TargetStrategy,
    CompositeActionConfig,
    CompositeActionStep,
)


# Strategy for generating valid action names (non-empty, stripped)
action_name_strategy = st.text(min_size=1, max_size=30).filter(lambda x: x.strip())

# Strategy for generating valid argument names
arg_name_strategy = st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isidentifier())

# Strategy for generating valid composite action steps
composite_step_strategy = st.builds(
    CompositeActionStep,
    action=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    element_index=st.integers(min_value=0, max_value=10),
    arg_prefix=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
)

# Strategy for generating valid composite action configs (with non-empty steps)
composite_action_strategy = st.one_of(
    st.none(),
    st.builds(
        CompositeActionConfig,
        mode=st.just("sequential"),
        steps=st.lists(composite_step_strategy, min_size=1, max_size=3),
    )
)

# Strategy for generating valid memory mode combinations
def valid_memory_mode_pair():
    """Generate valid (base_mode, incremental_mode) pairs."""
    return st.one_of(
        # NONE base -> NONE incremental only
        st.tuples(st.just(ActionMemoryMode.NONE), st.just(ActionMemoryMode.NONE)),
        # TARGET base -> TARGET or NONE incremental
        st.tuples(st.just(ActionMemoryMode.TARGET), st.sampled_from([ActionMemoryMode.TARGET, ActionMemoryMode.NONE])),
        # FULL base -> any incremental
        st.tuples(st.just(ActionMemoryMode.FULL), st.sampled_from([ActionMemoryMode.FULL, ActionMemoryMode.TARGET, ActionMemoryMode.NONE])),
    )

# Strategy for generating valid ActionTypeMetadata instances
@st.composite
def valid_action_metadata(draw):
    """Generate valid ActionTypeMetadata instances for testing."""
    name = draw(action_name_strategy)
    description = draw(st.one_of(st.none(), st.text(min_size=1, max_size=100)))
    default_strategy = draw(st.one_of(st.none(), st.sampled_from(list(TargetStrategy))))
    requires_target = draw(st.booleans())
    supported_args = draw(st.lists(arg_name_strategy, max_size=5, unique=True))
    # required_args must be subset of supported_args
    required_args = draw(st.lists(st.sampled_from(supported_args) if supported_args else st.nothing(), max_size=len(supported_args), unique=True)) if supported_args else []
    allow_follow_up = draw(st.booleans())
    allow_attachments = draw(st.booleans())
    base_mode, incremental_mode = draw(valid_memory_mode_pair())
    capture_after = draw(st.booleans())
    composite_action = draw(composite_action_strategy)
    
    return ActionTypeMetadata(
        name=name,
        description=description,
        default_strategy=default_strategy,
        requires_target=requires_target,
        supported_args=supported_args,
        required_args=required_args,
        allow_follow_up=allow_follow_up,
        allow_attachments=allow_attachments,
        base_memory_mode=base_mode,
        incremental_change_mode=incremental_mode,
        capture_incremental_memory_after_action=capture_after,
        composite_action=composite_action,
    )


# **Feature: action-metadata-consolidation, Property 6: JSON serialization round-trip**
# **Validates: Requirements 7.1**
@settings(max_examples=100)
@given(metadata=valid_action_metadata())
def test_json_round_trip_preserves_all_fields(metadata):
    """Property 6: For any valid ActionTypeMetadata instance, serializing to JSON
    and then deserializing should produce an equivalent ActionTypeMetadata instance
    with all fields preserved.
    
    This validates Requirement 7.1: WHEN action types are loaded from JSON THEN the
    system SHALL support all metadata fields including optional memory modes.
    """
    # Serialize to JSON dict
    json_dict = metadata.to_json_dict()
    
    # Deserialize back
    restored = ActionTypeMetadata.from_json_dict(json_dict)
    
    # Verify all fields are preserved
    assert restored.name == metadata.name, f"name mismatch: {restored.name} != {metadata.name}"
    assert restored.description == metadata.description, f"description mismatch"
    assert restored.get_strategy_value() == metadata.get_strategy_value(), f"default_strategy mismatch"
    assert restored.requires_target == metadata.requires_target, f"requires_target mismatch"
    assert restored.supported_args == metadata.supported_args, f"supported_args mismatch"
    assert restored.required_args == metadata.required_args, f"required_args mismatch"
    assert restored.allow_follow_up == metadata.allow_follow_up, f"allow_follow_up mismatch"
    assert restored.allow_attachments == metadata.allow_attachments, f"allow_attachments mismatch"
    assert restored.base_memory_mode == metadata.base_memory_mode, f"base_memory_mode mismatch"
    assert restored.incremental_change_mode == metadata.incremental_change_mode, f"incremental_change_mode mismatch"
    assert restored.capture_incremental_memory_after_action == metadata.capture_incremental_memory_after_action, f"capture_incremental_memory_after_action mismatch"
    
    # Verify composite action
    if metadata.composite_action is None:
        assert restored.composite_action is None, "composite_action should be None"
    else:
        assert restored.composite_action is not None, "composite_action should not be None"
        assert restored.composite_action.mode == metadata.composite_action.mode, "composite_action.mode mismatch"
        assert len(restored.composite_action.steps) == len(metadata.composite_action.steps), "composite_action.steps length mismatch"
        for i, (restored_step, original_step) in enumerate(zip(restored.composite_action.steps, metadata.composite_action.steps)):
            assert restored_step.action == original_step.action, f"step[{i}].action mismatch"
            assert restored_step.element_index == original_step.element_index, f"step[{i}].element_index mismatch"
            assert restored_step.arg_prefix == original_step.arg_prefix, f"step[{i}].arg_prefix mismatch"


# **Feature: action-metadata-consolidation, Property 6: JSON string round-trip**
# **Validates: Requirements 7.1**
@settings(max_examples=100)
@given(metadata=valid_action_metadata())
def test_json_string_round_trip(metadata):
    """Property 6 (string variant): For any valid ActionTypeMetadata instance,
    serializing to JSON string and then deserializing should produce an equivalent instance.
    
    This validates Requirement 7.1: WHEN action types are loaded from JSON THEN the
    system SHALL support all metadata fields including optional memory modes.
    """
    # Serialize to JSON string
    json_str = metadata.to_json_string()
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict), "JSON string should parse to a dict"
    
    # Deserialize back
    restored = ActionTypeMetadata.from_json_string(json_str)
    
    # Verify key fields are preserved
    assert restored.name == metadata.name
    assert restored.base_memory_mode == metadata.base_memory_mode
    assert restored.incremental_change_mode == metadata.incremental_change_mode


# =============================================================================
# Property 7, 8, 9: JSON Robustness Tests
# =============================================================================

# Strategy for generating invalid memory mode strings
invalid_memory_mode_strategy = st.text(min_size=1, max_size=20).filter(
    lambda x: x.lower() not in ('full', 'target', 'none', '')
)

# Strategy for generating invalid memory mode combinations
# These are combinations that violate the memory mode constraints
invalid_memory_mode_pair_strategy = st.one_of(
    # NONE base with non-NONE incremental (invalid)
    st.tuples(st.just('none'), st.sampled_from(['full', 'target'])),
    # TARGET base with FULL incremental (invalid)
    st.tuples(st.just('target'), st.just('full')),
)


# **Feature: action-metadata-consolidation, Property 7: Invalid JSON raises validation errors**
# **Validates: Requirements 7.2**
@settings(max_examples=100)
@given(invalid_mode=invalid_memory_mode_strategy)
def test_invalid_memory_mode_string_raises_error(invalid_mode):
    """Property 7: For any JSON data with invalid memory mode strings,
    attempting to deserialize into ActionTypeMetadata should raise a clear validation error.
    
    This validates Requirement 7.2: WHEN JSON contains invalid data THEN the system
    SHALL raise clear validation errors.
    """
    json_data = {
        "name": "test_action",
        "base_memory_mode": invalid_mode,  # Invalid memory mode string
    }
    
    with pytest.raises(ValidationError) as exc_info:
        ActionTypeMetadata.from_json_dict(json_data)
    
    # Verify error message mentions the invalid value
    error_str = str(exc_info.value)
    assert "base_memory_mode" in error_str.lower() or "invalid" in error_str.lower(), \
        f"Error should mention base_memory_mode or invalid, got: {error_str}"


# **Feature: action-metadata-consolidation, Property 7: Invalid memory mode combinations raise errors**
# **Validates: Requirements 7.2**
@settings(max_examples=100)
@given(mode_pair=invalid_memory_mode_pair_strategy)
def test_invalid_memory_mode_combination_raises_error(mode_pair):
    """Property 7: For any JSON data with invalid memory mode combinations,
    attempting to deserialize into ActionTypeMetadata should raise a validation error.
    
    This validates Requirement 7.2: WHEN JSON contains invalid data THEN the system
    SHALL raise clear validation errors.
    """
    base_mode, incremental_mode = mode_pair
    json_data = {
        "name": "test_action",
        "base_memory_mode": base_mode,
        "incremental_change_mode": incremental_mode,
    }
    
    with pytest.raises(ValidationError) as exc_info:
        ActionTypeMetadata.from_json_dict(json_data)
    
    # Verify error is about memory mode constraints
    error_str = str(exc_info.value)
    assert "memory" in error_str.lower() or "mode" in error_str.lower(), \
        f"Error should mention memory mode constraint, got: {error_str}"


# **Feature: action-metadata-consolidation, Property 7: Negative element_index raises error**
# **Validates: Requirements 7.2**
@settings(max_examples=100)
@given(negative_index=st.integers(max_value=-1))
def test_negative_element_index_raises_error(negative_index):
    """Property 7: For any JSON data with negative element indices in composite steps,
    attempting to deserialize should raise a validation error.
    
    This validates Requirement 7.2: WHEN JSON contains invalid data THEN the system
    SHALL raise clear validation errors.
    """
    json_data = {
        "name": "test_composite",
        "composite_action": {
            "mode": "sequential",
            "steps": [
                {
                    "action": "click",
                    "element_index": negative_index,  # Invalid negative index
                    "arg_prefix": "click"
                }
            ]
        }
    }
    
    with pytest.raises(ValidationError):
        ActionTypeMetadata.from_json_dict(json_data)


# **Feature: action-metadata-consolidation, Property 8: Missing optional fields use defaults**
# **Validates: Requirements 7.3**
@settings(max_examples=100)
@given(name=action_name_strategy)
def test_missing_optional_fields_use_defaults(name):
    """Property 8: For any JSON data missing optional fields,
    deserializing into ActionTypeMetadata should succeed and use sensible default values.
    
    This validates Requirement 7.3: WHEN JSON is missing required fields THEN the system
    SHALL use sensible defaults where appropriate.
    """
    # Minimal JSON with only required field (name)
    json_data = {"name": name}
    
    # Should succeed without errors
    metadata = ActionTypeMetadata.from_json_dict(json_data)
    
    # Verify defaults are applied
    assert metadata.name == name
    assert metadata.description is None, "description should default to None"
    assert metadata.default_strategy is None, "default_strategy should default to None"
    assert metadata.requires_target is True, "requires_target should default to True"
    assert metadata.supported_args == [], "supported_args should default to empty list"
    assert metadata.required_args == [], "required_args should default to empty list"
    assert metadata.allow_follow_up is False, "allow_follow_up should default to False"
    assert metadata.allow_attachments is False, "allow_attachments should default to False"
    assert metadata.base_memory_mode == ActionMemoryMode.NONE, "base_memory_mode should default to NONE"
    assert metadata.incremental_change_mode == ActionMemoryMode.NONE, "incremental_change_mode should default to NONE"
    assert metadata.capture_incremental_memory_after_action is True, "capture_incremental_memory_after_action should default to True"
    assert metadata.composite_action is None, "composite_action should default to None"


# **Feature: action-metadata-consolidation, Property 8: Partial optional fields use defaults for missing**
# **Validates: Requirements 7.3**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    description=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    requires_target=st.booleans(),
)
def test_partial_optional_fields_use_defaults(name, description, requires_target):
    """Property 8: For any JSON data with some optional fields present and others missing,
    deserializing should use defaults only for the missing fields.
    
    This validates Requirement 7.3: WHEN JSON is missing required fields THEN the system
    SHALL use sensible defaults where appropriate.
    """
    json_data = {
        "name": name,
        "description": description,
        "requires_target": requires_target,
        # Other optional fields are missing
    }
    
    metadata = ActionTypeMetadata.from_json_dict(json_data)
    
    # Provided fields should be preserved
    assert metadata.name == name
    assert metadata.description == description
    assert metadata.requires_target == requires_target
    
    # Missing fields should use defaults
    assert metadata.base_memory_mode == ActionMemoryMode.NONE
    assert metadata.incremental_change_mode == ActionMemoryMode.NONE
    assert metadata.allow_follow_up is False
    assert metadata.allow_attachments is False


# **Feature: action-metadata-consolidation, Property 9: Unknown JSON fields are ignored**
# **Validates: Requirements 7.4**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    unknown_field_name=st.text(min_size=1, max_size=20).filter(
        lambda x: x.strip() and x.isidentifier() and x not in [
            'name', 'description', 'default_strategy', 'requires_target',
            'supported_args', 'required_args', 'allow_follow_up',
            'allow_attachments', 'base_memory_mode', 'incremental_change_mode',
            'capture_incremental_memory_after_action', 'composite_action'
        ]
    ),
    unknown_field_value=st.one_of(
        st.text(max_size=50),
        st.integers(),
        st.booleans(),
        st.none(),
    ),
)
def test_unknown_fields_are_ignored(name, unknown_field_name, unknown_field_value):
    """Property 9: For any JSON data containing extra unknown fields not in the
    ActionTypeMetadata schema, deserializing should succeed and ignore the unknown fields.
    
    This validates Requirement 7.4: WHEN JSON contains unknown fields THEN the system
    SHALL ignore them without errors.
    """
    json_data = {
        "name": name,
        unknown_field_name: unknown_field_value,  # Unknown field
    }
    
    # Should succeed without errors
    metadata = ActionTypeMetadata.from_json_dict(json_data)
    
    # Known field should be preserved
    assert metadata.name == name
    
    # Unknown field should not be accessible (not stored)
    assert not hasattr(metadata, unknown_field_name) or \
           getattr(metadata, unknown_field_name, None) is None, \
           f"Unknown field '{unknown_field_name}' should be ignored"


# **Feature: action-metadata-consolidation, Property 9: Multiple unknown fields are ignored**
# **Validates: Requirements 7.4**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    num_unknown_fields=st.integers(min_value=1, max_value=5),
)
def test_multiple_unknown_fields_are_ignored(name, num_unknown_fields):
    """Property 9: For any JSON data containing multiple unknown fields,
    deserializing should succeed and ignore all unknown fields.
    
    This validates Requirement 7.4: WHEN JSON contains unknown fields THEN the system
    SHALL ignore them without errors.
    """
    json_data = {"name": name}
    
    # Add multiple unknown fields
    for i in range(num_unknown_fields):
        json_data[f"unknown_field_{i}"] = f"value_{i}"
    
    # Should succeed without errors
    metadata = ActionTypeMetadata.from_json_dict(json_data)
    
    # Known field should be preserved
    assert metadata.name == name
    
    # Verify the metadata object doesn't have the unknown fields
    for i in range(num_unknown_fields):
        field_name = f"unknown_field_{i}"
        assert not hasattr(metadata, field_name), \
            f"Unknown field '{field_name}' should be ignored"


if __name__ == '__main__':
    print("Running property-based tests for JSON serialization...")
    print()
    
    tests = [
        ("Property 6: JSON round-trip preserves all fields", 
         test_json_round_trip_preserves_all_fields),
        ("Property 6: JSON string round-trip", 
         test_json_string_round_trip),
        ("Property 7: Invalid memory mode string raises error",
         test_invalid_memory_mode_string_raises_error),
        ("Property 7: Invalid memory mode combination raises error",
         test_invalid_memory_mode_combination_raises_error),
        ("Property 7: Negative element_index raises error",
         test_negative_element_index_raises_error),
        ("Property 8: Missing optional fields use defaults",
         test_missing_optional_fields_use_defaults),
        ("Property 8: Partial optional fields use defaults",
         test_partial_optional_fields_use_defaults),
        ("Property 9: Unknown fields are ignored",
         test_unknown_fields_are_ignored),
        ("Property 9: Multiple unknown fields are ignored",
         test_multiple_unknown_fields_are_ignored),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Error: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll property-based tests passed! ✓")
