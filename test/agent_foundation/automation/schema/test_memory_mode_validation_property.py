"""Property-based tests for ActionTypeMetadata validation.

This module contains property-based tests using hypothesis to verify
memory mode validation constraints and composite action validation
as specified in the design document.

**Feature: action-metadata-consolidation, Property 1, 2, 3, 4: Memory mode validation**
**Validates: Requirements 2.2, 2.3, 2.4, 2.6**

**Feature: action-metadata-consolidation, Property 5: Composite action requires steps**
**Validates: Requirements 3.2**
"""
import sys
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


# Strategy for generating valid action names
action_name_strategy = st.text(min_size=1, max_size=30).filter(lambda x: x.strip())


# **Feature: action-metadata-consolidation, Property 1: Memory mode validation for NONE base mode**
# **Validates: Requirements 2.2**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    incremental_mode=st.sampled_from([ActionMemoryMode.TARGET, ActionMemoryMode.FULL]),
)
def test_none_base_mode_requires_none_incremental(name, incremental_mode):
    """Property 1: For any ActionTypeMetadata with base_memory_mode=NONE,
    attempting to set incremental_change_mode to any value other than NONE
    should raise a validation error.
    
    This validates Requirement 2.2: WHEN base_memory_mode is NONE THEN the system
    SHALL require incremental_change_mode to be NONE.
    """
    # Attempting to create with NONE base and non-NONE incremental should fail
    with pytest.raises(ValidationError) as exc_info:
        ActionTypeMetadata(
            name=name,
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=incremental_mode,
        )
    
    # Verify the error message mentions the constraint
    error_str = str(exc_info.value)
    assert "base_memory_mode=NONE" in error_str or "NONE" in error_str


# **Feature: action-metadata-consolidation, Property 1: Memory mode validation for NONE base mode (valid case)**
# **Validates: Requirements 2.2**
@settings(max_examples=100)
@given(name=action_name_strategy)
def test_none_base_mode_with_none_incremental_is_valid(name):
    """Property 1 (valid case): For any ActionTypeMetadata with base_memory_mode=NONE,
    setting incremental_change_mode to NONE should succeed.
    
    This validates Requirement 2.2: WHEN base_memory_mode is NONE THEN the system
    SHALL require incremental_change_mode to be NONE.
    """
    # Creating with NONE base and NONE incremental should succeed
    metadata = ActionTypeMetadata(
        name=name,
        base_memory_mode=ActionMemoryMode.NONE,
        incremental_change_mode=ActionMemoryMode.NONE,
    )
    
    assert metadata.base_memory_mode == ActionMemoryMode.NONE
    assert metadata.incremental_change_mode == ActionMemoryMode.NONE


# **Feature: action-metadata-consolidation, Property 2: Memory mode validation for TARGET base mode**
# **Validates: Requirements 2.3**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    incremental_mode=st.sampled_from([ActionMemoryMode.TARGET, ActionMemoryMode.NONE]),
)
def test_target_base_mode_allows_target_or_none_incremental(name, incremental_mode):
    """Property 2 (valid case): For any ActionTypeMetadata with base_memory_mode=TARGET,
    incremental_change_mode can be TARGET or NONE without raising validation errors.
    
    This validates Requirement 2.3: WHEN base_memory_mode is TARGET THEN the system
    SHALL allow incremental_change_mode to be TARGET or NONE only.
    """
    # Creating with TARGET base and TARGET or NONE incremental should succeed
    metadata = ActionTypeMetadata(
        name=name,
        base_memory_mode=ActionMemoryMode.TARGET,
        incremental_change_mode=incremental_mode,
    )
    
    assert metadata.base_memory_mode == ActionMemoryMode.TARGET
    assert metadata.incremental_change_mode == incremental_mode


# **Feature: action-metadata-consolidation, Property 2: Memory mode validation for TARGET base mode (invalid case)**
# **Validates: Requirements 2.3**
@settings(max_examples=100)
@given(name=action_name_strategy)
def test_target_base_mode_rejects_full_incremental(name):
    """Property 2 (invalid case): For any ActionTypeMetadata with base_memory_mode=TARGET,
    attempting to set incremental_change_mode to FULL should raise a validation error.
    
    This validates Requirement 2.3: WHEN base_memory_mode is TARGET THEN the system
    SHALL allow incremental_change_mode to be TARGET or NONE only.
    """
    # Attempting to create with TARGET base and FULL incremental should fail
    with pytest.raises(ValidationError) as exc_info:
        ActionTypeMetadata(
            name=name,
            base_memory_mode=ActionMemoryMode.TARGET,
            incremental_change_mode=ActionMemoryMode.FULL,
        )
    
    # Verify the error message mentions the constraint
    error_str = str(exc_info.value)
    assert "TARGET" in error_str or "incremental_change_mode" in error_str


# **Feature: action-metadata-consolidation, Property 3: Memory mode validation for FULL base mode**
# **Validates: Requirements 2.4**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    incremental_mode=st.sampled_from([ActionMemoryMode.FULL, ActionMemoryMode.TARGET, ActionMemoryMode.NONE]),
)
def test_full_base_mode_allows_any_incremental(name, incremental_mode):
    """Property 3: For any ActionTypeMetadata with base_memory_mode=FULL,
    incremental_change_mode can be FULL, TARGET, or NONE without raising validation errors.
    
    This validates Requirement 2.4: WHEN base_memory_mode is FULL THEN the system
    SHALL allow incremental_change_mode to be FULL, TARGET, or NONE.
    """
    # Creating with FULL base and any incremental mode should succeed
    metadata = ActionTypeMetadata(
        name=name,
        base_memory_mode=ActionMemoryMode.FULL,
        incremental_change_mode=incremental_mode,
    )
    
    assert metadata.base_memory_mode == ActionMemoryMode.FULL
    assert metadata.incremental_change_mode == incremental_mode


# **Feature: action-metadata-consolidation, Property 4: Default memory modes**
# **Validates: Requirements 2.6**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    requires_target=st.booleans(),
    allow_follow_up=st.booleans(),
    allow_attachments=st.booleans(),
)
def test_default_memory_modes_are_none(name, requires_target, allow_follow_up, allow_attachments):
    """Property 4: For any ActionTypeMetadata instance created without specifying
    memory modes, both base_memory_mode and incremental_change_mode should default to NONE.
    
    This validates Requirement 2.6: WHEN memory modes are not specified THEN the system
    SHALL default to NONE for backward compatibility.
    """
    # Create ActionTypeMetadata without specifying memory modes
    metadata = ActionTypeMetadata(
        name=name,
        requires_target=requires_target,
        allow_follow_up=allow_follow_up,
        allow_attachments=allow_attachments,
        # Explicitly NOT setting base_memory_mode or incremental_change_mode
    )
    
    # Both memory modes should default to NONE
    assert metadata.base_memory_mode == ActionMemoryMode.NONE, \
        f"Expected base_memory_mode=NONE, got {metadata.base_memory_mode}"
    assert metadata.incremental_change_mode == ActionMemoryMode.NONE, \
        f"Expected incremental_change_mode=NONE, got {metadata.incremental_change_mode}"


# **Feature: action-metadata-consolidation, Property 4: Default memory modes (with other fields)**
# **Validates: Requirements 2.6**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    supported_args=st.lists(st.text(min_size=1, max_size=20).filter(lambda x: x.strip()), max_size=5),
    description=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
)
def test_default_memory_modes_with_various_fields(name, supported_args, description):
    """Property 4 (extended): For any ActionTypeMetadata instance created with various
    other fields but without specifying memory modes, both base_memory_mode and 
    incremental_change_mode should default to NONE.
    
    This validates Requirement 2.6: WHEN memory modes are not specified THEN the system
    SHALL default to NONE for backward compatibility.
    """
    # Create ActionTypeMetadata with various fields but no memory modes
    metadata = ActionTypeMetadata(
        name=name,
        supported_args=supported_args,
        description=description,
        # Explicitly NOT setting base_memory_mode or incremental_change_mode
    )
    
    # Both memory modes should default to NONE
    assert metadata.base_memory_mode == ActionMemoryMode.NONE, \
        f"Expected base_memory_mode=NONE, got {metadata.base_memory_mode}"
    assert metadata.incremental_change_mode == ActionMemoryMode.NONE, \
        f"Expected incremental_change_mode=NONE, got {metadata.incremental_change_mode}"


# Strategy for generating valid composite action steps
composite_step_strategy = st.builds(
    CompositeActionStep,
    action=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    element_index=st.integers(min_value=0, max_value=10),
    arg_prefix=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
)


# **Feature: action-metadata-consolidation, Property 5: Composite action requires steps**
# **Validates: Requirements 3.2**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    mode=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
)
def test_composite_action_with_empty_steps_raises_error(name, mode):
    """Property 5 (invalid case): For any ActionTypeMetadata where composite_action
    is set to a non-None value with empty steps, a validation error should be raised.
    
    This validates Requirement 3.2: WHEN composite_action mode is set THEN the system
    SHALL require composite_steps to be defined.
    """
    # Create a CompositeActionConfig with empty steps
    composite_config = CompositeActionConfig(mode=mode, steps=[])
    
    # Attempting to create ActionTypeMetadata with empty composite steps should fail
    with pytest.raises(ValidationError) as exc_info:
        ActionTypeMetadata(
            name=name,
            composite_action=composite_config,
        )
    
    # Verify the error message mentions the constraint
    error_str = str(exc_info.value)
    assert "composite_action" in error_str.lower() or "steps" in error_str.lower()


# **Feature: action-metadata-consolidation, Property 5: Composite action requires steps (valid case)**
# **Validates: Requirements 3.2**
@settings(max_examples=100)
@given(
    name=action_name_strategy,
    mode=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    steps=st.lists(composite_step_strategy, min_size=1, max_size=5),
)
def test_composite_action_with_non_empty_steps_is_valid(name, mode, steps):
    """Property 5 (valid case): For any ActionTypeMetadata where composite_action
    is set with non-empty steps, the creation should succeed.
    
    This validates Requirement 3.2: WHEN composite_action mode is set THEN the system
    SHALL require composite_steps to be defined.
    """
    # Create a CompositeActionConfig with non-empty steps
    composite_config = CompositeActionConfig(mode=mode, steps=steps)
    
    # Creating ActionTypeMetadata with non-empty composite steps should succeed
    metadata = ActionTypeMetadata(
        name=name,
        composite_action=composite_config,
    )
    
    assert metadata.composite_action is not None
    assert len(metadata.composite_action.steps) == len(steps)
    assert metadata.composite_action.mode == mode


# **Feature: action-metadata-consolidation, Property 5: Composite action requires steps (None case)**
# **Validates: Requirements 3.2**
@settings(max_examples=100)
@given(name=action_name_strategy)
def test_composite_action_none_is_valid(name):
    """Property 5 (None case): For any ActionTypeMetadata where composite_action
    is None (not set), the creation should succeed without validation errors.
    
    This validates Requirement 3.2: The validation only applies when composite_action
    is set to a non-None value.
    """
    # Creating ActionTypeMetadata without composite_action should succeed
    metadata = ActionTypeMetadata(
        name=name,
        composite_action=None,
    )
    
    assert metadata.composite_action is None


if __name__ == '__main__':
    print("Running property-based tests for ActionTypeMetadata validation...")
    print()
    
    tests = [
        ("Property 1 (invalid): NONE base mode requires NONE incremental", 
         test_none_base_mode_requires_none_incremental),
        ("Property 1 (valid): NONE base mode with NONE incremental is valid", 
         test_none_base_mode_with_none_incremental_is_valid),
        ("Property 2 (valid): TARGET base mode allows TARGET or NONE incremental", 
         test_target_base_mode_allows_target_or_none_incremental),
        ("Property 2 (invalid): TARGET base mode rejects FULL incremental", 
         test_target_base_mode_rejects_full_incremental),
        ("Property 3: FULL base mode allows any incremental mode", 
         test_full_base_mode_allows_any_incremental),
        ("Property 4: Default memory modes are NONE", 
         test_default_memory_modes_are_none),
        ("Property 4 (extended): Default memory modes with various fields", 
         test_default_memory_modes_with_various_fields),
        ("Property 5 (invalid): Composite action with empty steps raises error", 
         test_composite_action_with_empty_steps_raises_error),
        ("Property 5 (valid): Composite action with non-empty steps is valid", 
         test_composite_action_with_non_empty_steps_is_valid),
        ("Property 5 (None): Composite action None is valid", 
         test_composite_action_none_is_valid),
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
