"""
Action Type Metadata System

Defines metadata for action types including default strategies, required arguments,
behavior flags, and memory capture modes. Supports loading custom metadata from JSON files.

Memory modes are a generic UI automation concern - any framework that performs actions
changing visible content (scrolling, navigation) needs to capture state before/after changes.

Enums, constants, and default actions are defined in constants.py.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from rich_python_utils.common_utils.typing_helper import parse_type_string

# Import enums and constants from constants.py
from agent_foundation.automation.schema.constants import (
    ActionMemoryMode,
    TargetStrategy,
    CompositeActionStep,
    CompositeActionConfig,
    ACTION_NAME_CLICK,
    ACTION_NAME_INPUT_TEXT,
    ACTION_NAME_APPEND_TEXT,
    ACTION_NAME_SCROLL,
    ACTION_NAME_SCROLL_UP_TO_ELEMENT,
    ACTION_NAME_VISIT_URL,
    ACTION_NAME_WAIT,
    ACTION_NAME_NO_OP,
    ACTION_NAME_INPUT_AND_SUBMIT,
    get_default_actions,
)

# Re-export
__all__ = [
    'ActionMemoryMode',
    'TargetStrategy',
    'CompositeActionStep',
    'CompositeActionConfig',
    'ActionTypeMetadata',
    'ActionMetadataRegistry',
    'ACTION_NAME_CLICK',
    'ACTION_NAME_INPUT_TEXT',
    'ACTION_NAME_APPEND_TEXT',
    'ACTION_NAME_SCROLL',
    'ACTION_NAME_SCROLL_UP_TO_ELEMENT',
    'ACTION_NAME_VISIT_URL',
    'ACTION_NAME_WAIT',
    'ACTION_NAME_NO_OP',
    'ACTION_NAME_INPUT_AND_SUBMIT',
]


class ActionTypeMetadata(BaseModel):
    """
    Metadata for an action type.
    
    Supports both simple actions (click, input) and composite actions
    (input_and_submit). Includes memory capture configuration for UI
    automation scenarios where actions change visible content.
    
    The default_strategy field accepts both string values and TargetStrategy 
    enum for backward compatibility.
    """
    # Core identification
    name: str  # Action type name (e.g., "click", "input_text")
    description: Optional[str] = None  # Human-readable description
    
    # Target resolution
    default_strategy: Optional[Union[str, TargetStrategy]] = None
    requires_target: bool = True  # Whether this action requires a target element
    
    # Arguments
    supported_args: List[str] = Field(default_factory=list)
    required_args: List[str] = Field(default_factory=list)
    arg_types: Dict[str, str] = Field(default_factory=dict)
    # Maps arg names to type specification strings: 'float', 'int', 'bool', 'str', or union 'float|int'
    # Supports union types via pipe syntax: 'float|int' means try float first, then int
    # Example: {'seconds': 'float', 'count': 'int', 'value': 'float|int|str'}
    # If not specified or 'any', acts like Any - no type coercion applied

    # Behavior flags
    allow_follow_up: bool = False  # Whether action can be followed by similar actions
    allow_attachments: bool = False  # Whether action accepts attachments
    
    # Memory capture (generic UI automation feature)
    base_memory_mode: ActionMemoryMode = ActionMemoryMode.NONE
    incremental_change_mode: ActionMemoryMode = ActionMemoryMode.NONE
    capture_incremental_memory_after_action: bool = True
    
    # Composite actions
    composite_action: Optional[CompositeActionConfig] = None
    
    class Config:
        """Pydantic model configuration."""
        extra = 'ignore'  # Ignore unknown fields in JSON (Requirement 7.4)
    
    @field_validator('default_strategy', mode='before')
    @classmethod
    def normalize_strategy(cls, v):
        """Normalize default_strategy to TargetStrategy enum when possible."""
        if v is None:
            return None
        if isinstance(v, TargetStrategy):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            for strategy in TargetStrategy:
                if strategy.value == v_lower:
                    return strategy
            return v
        return v
    
    @field_validator('base_memory_mode', mode='before')
    @classmethod
    def normalize_base_memory_mode(cls, v):
        """Normalize base_memory_mode to ActionMemoryMode enum."""
        if v is None:
            return ActionMemoryMode.NONE
        if isinstance(v, ActionMemoryMode):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            for mode in ActionMemoryMode:
                if mode.value == v_lower:
                    return mode
            raise ValueError(
                f"Invalid base_memory_mode: {v}. "
                f"Must be one of: {[m.value for m in ActionMemoryMode]}"
            )
        return v
    
    @field_validator('incremental_change_mode', mode='before')
    @classmethod
    def normalize_incremental_change_mode(cls, v):
        """Normalize incremental_change_mode to ActionMemoryMode enum."""
        if v is None:
            return ActionMemoryMode.NONE
        if isinstance(v, ActionMemoryMode):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            for mode in ActionMemoryMode:
                if mode.value == v_lower:
                    return mode
            raise ValueError(
                f"Invalid incremental_change_mode: {v}. "
                f"Must be one of: {[m.value for m in ActionMemoryMode]}"
            )
        return v
    
    @model_validator(mode='after')
    def validate_memory_mode_constraints(self):
        """
        Validate memory mode constraints.
        
        Rules:
        - If base_memory_mode is NONE, incremental_change_mode must be NONE
        - If base_memory_mode is TARGET, incremental_change_mode must be TARGET or NONE
        - If base_memory_mode is FULL, incremental_change_mode can be FULL, TARGET, or NONE
        """
        base_mode = self.base_memory_mode
        incremental_mode = self.incremental_change_mode
        
        if base_mode == ActionMemoryMode.NONE and incremental_mode != ActionMemoryMode.NONE:
            raise ValueError(
                f"base_memory_mode=NONE requires incremental_change_mode=NONE, "
                f"got incremental_change_mode={incremental_mode.value}"
            )
        
        if base_mode == ActionMemoryMode.TARGET:
            if incremental_mode not in (ActionMemoryMode.TARGET, ActionMemoryMode.NONE):
                raise ValueError(
                    f"base_memory_mode=TARGET allows incremental_change_mode=(TARGET|NONE), "
                    f"got incremental_change_mode={incremental_mode.value}"
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_composite_action_requirements(self):
        """Validate composite action configuration has non-empty steps."""
        if self.composite_action is not None:
            if not self.composite_action.steps:
                raise ValueError(
                    "composite_action requires steps to be defined and non-empty"
                )
        return self
    
    @computed_field
    @property
    def parsed_arg_types(self) -> Dict[str, Tuple[type, ...]]:
        """
        Pre-computed Python types from arg_types for efficient runtime coercion.

        Returns:
            Dict mapping arg names to tuples of Python types.
            Empty dict if arg_types is empty.

        Example:
            If arg_types = {'seconds': 'float', 'value': 'float|int'}
            Returns: {'seconds': (float,), 'value': (float, int)}
        """
        return {
            arg_name: parse_type_string(type_str)
            for arg_name, type_str in self.arg_types.items()
        }

    def get_strategy_value(self) -> Optional[str]:
        """Get the string value of the default_strategy."""
        if self.default_strategy is None:
            return None
        if isinstance(self.default_strategy, TargetStrategy):
            return self.default_strategy.value
        return self.default_strategy
    
    def get_memory_modes(self) -> Tuple[ActionMemoryMode, ActionMemoryMode]:
        """Get memory capture modes as (base_memory_mode, incremental_change_mode)."""
        return (self.base_memory_mode, self.incremental_change_mode)
    
    def to_json_dict(self) -> Dict:
        """Serialize to a JSON-compatible dictionary."""
        return self.model_dump(mode='json')
    
    def to_json_string(self, indent: Optional[int] = None) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=indent)
    
    @classmethod
    def from_json_dict(cls, data: Dict) -> 'ActionTypeMetadata':
        """Deserialize from a JSON-compatible dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_json_string(cls, json_str: str) -> 'ActionTypeMetadata':
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)



class ActionMetadataRegistry:
    """Registry of action type metadata."""

    def __init__(self, metadata_file: Optional[str] = None):
        """
        Initialize action metadata registry.

        Args:
            metadata_file: Optional path to custom metadata JSON file.
                          If None, loads default metadata from constants.py.
        """
        self.metadata: Dict[str, ActionTypeMetadata] = {}

        if metadata_file:
            self.load_from_file(metadata_file)
        else:
            self._load_defaults()

    def _load_defaults(self):
        """
        Load default action metadata from constants.py.
        
        The authoritative source of truth for default action configurations
        is in constants.py (get_default_actions function).
        
        A JSON reference artifact is available at:
            artifacts/default_action_types.json
        """
        for action in get_default_actions():
            self.metadata[action.name] = action

    def get_default_strategy(self, action_type: str) -> Optional[str]:
        """Get default target resolution strategy for an action type."""
        if action_type in self.metadata:
            return self.metadata[action_type].default_strategy
        return None

    def get_metadata(self, action_type: str) -> Optional[ActionTypeMetadata]:
        """Get metadata for an action type."""
        return self.metadata.get(action_type)

    def requires_target(self, action_type: str) -> bool:
        """Check if action type requires a target element."""
        if action_type in self.metadata:
            return self.metadata[action_type].requires_target
        return True  # Default to requiring target

    def load_from_file(self, filepath: str):
        """
        Load action metadata from JSON file.

        Args:
            filepath: Path to JSON file containing action metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Action metadata file not found: {filepath}")

        with open(path, 'r') as f:
            data = json.load(f)

        self.metadata = {}
        for item in data.get("actions", []):
            meta = ActionTypeMetadata(**item)
            self.metadata[meta.name] = meta

    def register_action(self, metadata: ActionTypeMetadata):
        """Register a new action type or override existing one."""
        self.metadata[metadata.name] = metadata

    def list_actions(self) -> List[str]:
        """Get list of all registered action types."""
        return list(self.metadata.keys())
    
    def get_memory_modes(self, action_type: str) -> Tuple[ActionMemoryMode, ActionMemoryMode]:
        """
        Get memory capture modes for an action type.
        
        Returns (NONE, NONE) if action type not found (safe default).
        """
        metadata = self.get_metadata(action_type)
        if metadata:
            return metadata.get_memory_modes()
        return (ActionMemoryMode.NONE, ActionMemoryMode.NONE)
    
    def requires_memory_capture(self, action_type: str) -> bool:
        """Check if an action type requires memory capture."""
        base_mode, incremental_mode = self.get_memory_modes(action_type)
        return base_mode != ActionMemoryMode.NONE or incremental_mode != ActionMemoryMode.NONE
    
    def get_capture_incremental_after_action(self, action_type: str) -> bool:
        """Check if incremental memory should be captured after action execution."""
        metadata = self.get_metadata(action_type)
        if metadata:
            return metadata.capture_incremental_memory_after_action
        return True  # Default to capturing after action
