"""
Arithmetic action metadata registry.

Defines action types for arithmetic operations: set, add, subtract, multiply, divide, power.
"""

import sys
from pathlib import Path

# Setup import paths
_current_file = Path(__file__).resolve()
_examples_dir = _current_file.parent
while _examples_dir.name != 'examples' and _examples_dir.parent != _examples_dir:
    _examples_dir = _examples_dir.parent
_project_root = _examples_dir.parent
_src_dir = _project_root / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_workspace_root = _project_root.parent
_rich_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _rich_python_utils_src.exists() and str(_rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_rich_python_utils_src))

from agent_foundation.automation.schema.action_metadata import (
    ActionMetadataRegistry,
    ActionTypeMetadata,
)


def create_arithmetic_registry() -> ActionMetadataRegistry:
    """Create registry with arithmetic action metadata."""
    registry = ActionMetadataRegistry()

    actions = [
        ActionTypeMetadata(
            name="set",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Set accumulator to specified value",
        ),
        ActionTypeMetadata(
            name="add",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Add value to accumulator",
        ),
        ActionTypeMetadata(
            name="subtract",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Subtract value from accumulator",
        ),
        ActionTypeMetadata(
            name="multiply",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Multiply accumulator by value",
        ),
        ActionTypeMetadata(
            name="divide",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Divide accumulator by value",
        ),
        ActionTypeMetadata(
            name="power",
            requires_target=False,
            supported_args=["value"],
            required_args=["value"],
            arg_types={"value": "float"},
            description="Raise accumulator to power",
        ),
    ]

    for action in actions:
        registry.register_action(action)

    return registry
