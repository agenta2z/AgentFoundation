"""
Action Schema Constants and Enums

Contains all constants, enums, and default action configurations for the
action metadata system. This is the authoritative source of truth for
default action definitions.

A JSON reference artifact is available at:
    artifacts/default_action_types.json

The JSON file shows the expected format for custom action configurations
that can be loaded via ActionMetadataRegistry.load_from_file().
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, field_validator

# Import TargetStrategy from common.py (single source of truth)
from .common import TargetStrategy


# =============================================================================
# Enums
# =============================================================================

class ActionMemoryMode(str, Enum):
    """
    Memory capture modes for UI automation actions.

    Memory modes control how UI state is captured during actions that may change
    what's visible on screen. This is a generic UI automation concern applicable
    to any framework (web, desktop, mobile) where actions can change visible content.

    - FULL: Capture full UI state (entire page/screen)
    - TARGET: Capture target element only (localized capture)
    - NONE: No memory capture (default for most actions)
    """
    FULL = 'full'      # Capture full UI state
    TARGET = 'target'  # Capture target element only
    NONE = 'none'      # No memory capture


# =============================================================================
# Action Name Constants
# =============================================================================

# Generic UI action name constants
# These provide a standard vocabulary for common UI automation actions
ACTION_NAME_CLICK = 'click'
ACTION_NAME_INPUT_TEXT = 'input_text'
ACTION_NAME_APPEND_TEXT = 'append_text'
ACTION_NAME_SCROLL = 'scroll'
ACTION_NAME_SCROLL_UP_TO_ELEMENT = 'scroll_up_to_element'
ACTION_NAME_VISIT_URL = 'visit_url'
ACTION_NAME_WAIT = 'wait'
ACTION_NAME_NO_OP = 'no_op'  # No actual operation; may capture latest state in action results
ACTION_NAME_INPUT_AND_SUBMIT = 'input_and_submit'  # Composite action


# =============================================================================
# Composite Action Models (needed for default actions)
# =============================================================================

class CompositeActionStep(BaseModel):
    """Step in a composite action sequence."""
    action: str  # Sub-action type
    element_index: int  # Which resolved element to use (must be >= 0)
    arg_prefix: str  # Prefix for extracting arguments
    
    @field_validator('element_index')
    @classmethod
    def validate_element_index(cls, v):
        """Validate that element_index is non-negative."""
        if v < 0:
            raise ValueError(f"element_index must be non-negative, got {v}")
        return v


class CompositeActionConfig(BaseModel):
    """Configuration for composite actions."""
    mode: str = "sequential"  # Currently only "sequential" supported
    steps: List[CompositeActionStep]


# =============================================================================
# Default Action Configurations
# =============================================================================

def get_default_actions():
    """
    Get the default action metadata configurations.
    
    This is the authoritative source of truth for default action configurations.
    Returns a list of ActionTypeMetadata instances.
    
    Note: This function imports ActionTypeMetadata locally to avoid circular imports.
    
    Returns:
        List of ActionTypeMetadata instances for all default actions.
    """
    # Import here to avoid circular dependency
    from agent_foundation.automation.schema.action_metadata import ActionTypeMetadata
    
    return [
        ActionTypeMetadata(
            name=ACTION_NAME_CLICK,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            description="Click on a UI element",
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_INPUT_TEXT,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            supported_args=["text", "clear_content"],
            required_args=["text"],
            arg_types={"text": "str", "clear_content": "bool"},
            description="Input text into a field",
            allow_attachments=True,
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_APPEND_TEXT,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            supported_args=["text"],
            required_args=["text"],
            arg_types={"text": "str"},
            description="Append text to existing field content",
            allow_attachments=True,
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_VISIT_URL,
            default_strategy=TargetStrategy.LITERAL,
            requires_target=True,
            description="Navigate to a URL",
            # NONE/NONE: Navigation replaces page content entirely, so capturing
            # memory before/after doesn't make sense - the page is completely new
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_SCROLL,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            supported_args=["direction", "amount"],
            arg_types={"direction": "str", "amount": "int|float"},
            description="Scroll an element or page",
            allow_follow_up=True,
            base_memory_mode=ActionMemoryMode.TARGET,
            incremental_change_mode=ActionMemoryMode.TARGET
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_SCROLL_UP_TO_ELEMENT,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            description="Scroll page until element is visible",
            allow_follow_up=True,
            base_memory_mode=ActionMemoryMode.TARGET,
            incremental_change_mode=ActionMemoryMode.TARGET
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_WAIT,
            default_strategy=None,
            requires_target=False,
            supported_args=["seconds"],
            required_args=["seconds"],
            arg_types={"seconds": "float"},
            description="Wait for a specified duration",
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        ActionTypeMetadata(
            name=ACTION_NAME_NO_OP,
            default_strategy=None,
            requires_target=False,
            description="No actual operation; may capture latest state in action results",
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE
        ),
        # Composite action: input_and_submit
        ActionTypeMetadata(
            name=ACTION_NAME_INPUT_AND_SUBMIT,
            default_strategy=TargetStrategy.FRAMEWORK_ID,
            requires_target=True,
            supported_args=["text", "clear_content"],
            required_args=["text"],
            arg_types={"text": "str", "clear_content": "bool"},
            description="Input text and click submit button (composite action)",
            allow_attachments=True,
            base_memory_mode=ActionMemoryMode.NONE,
            incremental_change_mode=ActionMemoryMode.NONE,
            composite_action=CompositeActionConfig(
                mode="sequential",
                steps=[
                    CompositeActionStep(
                        action="input_text",
                        element_index=0,
                        arg_prefix="input_text_"
                    ),
                    CompositeActionStep(
                        action="click",
                        element_index=1,
                        arg_prefix="click_"
                    )
                ]
            )
        ),
    ]
