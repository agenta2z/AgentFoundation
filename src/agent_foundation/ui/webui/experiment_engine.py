# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Experiment flow engine for scripted conversation demos."""

import importlib.resources
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class IterationStepInfo:
    """Information about a step within an iteration."""

    index: int
    name: str


@dataclass
class IterationConfig:
    """Configuration for an iteration within the experiment flow."""

    id: str
    name: str
    description: str
    start_step_index: int
    end_step_index: int
    steps: List[IterationStepInfo] = field(default_factory=list)


@dataclass
class InputFieldConfig:
    """Configuration for a user input field in a message."""

    variable_name: str  # Name of the variable to store the input value
    placeholder: str = "Enter your input..."  # Placeholder text
    multiline: bool = False  # Whether to show a multiline textarea
    optional: bool = True  # Whether the input is optional
    default_value: str = ""  # Default value for the input field
    collapsible: bool = False  # Whether the input field can be collapsed/expanded
    initially_collapsed: bool = False  # Whether the input starts in collapsed state
    mode_selector: Optional[Dict[str, Any]] = (
        None  # Mode selector configuration for research mode dropdown
    )


@dataclass
class ContinueButtonConfig:
    """Configuration for continue button in progress_header."""

    label: str = "🚀 Continue"
    style: str = "primary"  # "primary" or "secondary"


@dataclass
class ProgressHeaderConfig:
    """Configuration for progress header that appears before progress sections.

    The progress header is shown after pre_messages complete and before progress
    sections start. It can contain:
    - Content: Markdown text explaining what the user should do
    - Input field: Optional text input for user to provide context
    - Continue button: Button to start the progress animation

    The animation pauses at this phase until user clicks continue.
    """

    content: str  # Markdown content to display
    input_field: Optional[InputFieldConfig] = None  # Optional input field
    continue_button: ContinueButtonConfig = field(
        default_factory=lambda: ContinueButtonConfig()
    )


@dataclass
class MessageConfig:
    """Configuration for a single message."""

    content: str
    type: str = "text"  # "text" or "progress"
    delay_min: float = 0.5
    delay_max: float = 2.0
    file: Optional[str] = None  # relative path to file
    input_field: Optional[InputFieldConfig] = (
        None  # Optional input field for user input
    )
    editable_list: Optional[Dict[str, Any]] = (
        None  # Optional editable list configuration (generic for queries, proposals, etc.)
    )

    def get_delay(self) -> float:
        return random.uniform(self.delay_min, self.delay_max)


@dataclass
class ParallelGroup:
    """Group of messages to display in parallel."""

    items: List[MessageConfig]


@dataclass
class ProgressMessage:
    """A message in a progress section."""

    content: str
    delay: float = 1.0
    status: str = "pending"


@dataclass
class TaskConfig:
    """Configuration for a single task in task_progress sections."""

    id: str
    title: str
    messages: List[ProgressMessage]
    result: Optional[Dict[str, Any]] = None  # {delta_latency, is_better, result_message}


@dataclass
class ProgressSection:
    """Configuration for a progress section."""

    slot: str
    title: str
    messages: List[ProgressMessage]
    collapsible: bool = True
    initial_state: str = "expanded"
    show_history: bool = True
    message_delay_multiplier: float = 1.0
    appearance_delay_min: float = 0.0
    appearance_delay_max: float = 0.0
    prompt_file: Optional[str] = None  # Path to prompt file for this section
    enabled: bool = True  # Whether to show this section (default True)
    input_field: Optional[InputFieldConfig] = (
        None  # Optional input field for the section
    )
    # Task progress specific fields
    section_type: Optional[str] = None  # "task_progress" for task progress panels
    tasks: Optional[List[TaskConfig]] = None  # List of tasks for task_progress type


@dataclass
class ReturnToFlowConfig:
    """Configuration for return-to-flow button when in inferencer mode."""

    label: str = "↩️ Return to Demo Flow"
    style: str = "primary"


@dataclass
class InferencerActionConfig:
    """Configuration for triggering an inferencer (e.g., DevMate)."""

    type: str = "devmate"  # inferencer type
    prompt: Optional[str] = None  # Template name (e.g., "devmate_prompt")
    prompt_template: Optional[str] = None  # Inline template with {{ user_input }} etc.
    engine: Optional[str] = None  # Template engine (e.g., "jinja")
    model_name: Optional[str] = None  # Override model name
    max_tokens: Optional[int] = None  # Override max tokens
    context_files: Optional[List[str]] = None  # Files to include as context


@dataclass
class DefaultInferencerConfig:
    """Default inferencer settings for the flow."""

    type: str = "devmate"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestedAction:
    """Configuration for a suggested action button."""

    label: str  # Button text (e.g., "✅ Continue to proposals")
    action_type: str  # "continue" | "input_prefix" | "branch_with_input" | "inferencer"
    prefix: Optional[str] = None  # Prefix text for input_prefix type
    style: str = "primary"  # "primary" (blue) or "secondary" (gray)
    user_message: Optional[str] = None  # Message to display as user message
    inferencer_config: Optional[InferencerActionConfig] = None  # For inferencer actions
    return_to_flow: Optional[ReturnToFlowConfig] = None  # Button config for returning
    target_step: Optional[str] = None  # Target step ID for branch_with_input
    input_config: Optional[Dict[str, Any]] = None  # Input config for branch_with_input


@dataclass
class StepConfig:
    """Configuration for a conversation step."""

    id: str
    role: str  # "assistant" or "user"
    messages: List[Union[MessageConfig, ParallelGroup]]
    wait_for_user: bool = False
    pre_delay_min: float = 2.0  # Delay before showing messages (thinking time)
    pre_delay_max: float = 5.0
    post_delay_min: float = 0.0  # Delay after showing messages
    post_delay_max: float = 0.0
    suggested_actions: List[SuggestedAction] = field(default_factory=list)
    suggested_actions_message: Optional[str] = None
    progress_sections: List[ProgressSection] = field(default_factory=list)
    progress_header: Optional[ProgressHeaderConfig] = (
        None  # Header with input before progress
    )
    pre_messages: List[Union[MessageConfig, ParallelGroup]] = field(
        default_factory=list
    )
    post_messages: List[Union[MessageConfig, ParallelGroup]] = field(
        default_factory=list
    )
    keep_progress_sections: bool = False
    post_messages_only: bool = False

    def get_pre_delay(self) -> float:
        """Get random pre-delay in seconds."""
        return random.uniform(self.pre_delay_min, self.pre_delay_max)

    def get_post_delay(self) -> float:
        """Get random post-delay in seconds."""
        return random.uniform(self.post_delay_min, self.post_delay_max)


@dataclass
class ExperimentFlowConfig:
    """Full experiment flow configuration."""

    name: str
    description: str
    steps: List[StepConfig]
    default_inferencer: Optional[DefaultInferencerConfig] = None
    code_entry_point: Optional[str] = None  # Relative path from fbsource root
    context_files_root: Optional[str] = (
        None  # Root path for context files (e.g., "fbcode/_tony_dev/...")
    )
    metadata: Optional[Dict[str, Any]] = None


class ExperimentFlowEngine:
    """State machine for experiment flow playback."""

    def __init__(
        self,
        config: ExperimentFlowConfig,
        base_path: str,
        flow_name: str = "",
        start_step_index: int = 0,
    ):
        self.config = config
        self.base_path = base_path
        self.flow_name = flow_name  # Store flow name for package resource lookups
        self.current_step_index = start_step_index
        self._iterations: List[IterationConfig] = self._parse_iterations()

    def _parse_iterations(self) -> List[IterationConfig]:
        """Parse iteration metadata from config."""
        iterations = []
        if self.config.metadata and "iterations" in self.config.metadata:
            for iter_data in self.config.metadata["iterations"]:
                steps = [
                    IterationStepInfo(index=s["index"], name=s["name"])
                    for s in iter_data.get("steps", [])
                ]
                iterations.append(
                    IterationConfig(
                        id=iter_data["id"],
                        name=iter_data["name"],
                        description=iter_data.get("description", ""),
                        start_step_index=iter_data["start_step_index"],
                        end_step_index=iter_data["end_step_index"],
                        steps=steps,
                    )
                )
        return iterations

    def get_current_step(self) -> Optional[StepConfig]:
        if self.current_step_index < len(self.config.steps):
            return self.config.steps[self.current_step_index]
        return None

    def advance_step(self) -> bool:
        self.current_step_index += 1
        return not self.is_complete()

    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.config.steps)

    def get_iterations(self) -> List[IterationConfig]:
        """Get all available iterations."""
        return self._iterations

    def get_current_iteration(self) -> Optional[IterationConfig]:
        """Get the iteration containing the current step."""
        for iteration in self._iterations:
            if (
                iteration.start_step_index
                <= self.current_step_index
                <= iteration.end_step_index
            ):
                return iteration
        return None

    def jump_to_step(self, step_index: int) -> bool:
        """Jump to a specific step by index.

        Args:
            step_index: The 0-based index of the step to jump to.

        Returns:
            True if jump was successful, False if index out of range.
        """
        if 0 <= step_index < len(self.config.steps):
            self.current_step_index = step_index
            return True
        return False

    def jump_to_step_by_id(self, step_id: str) -> bool:
        """Jump to a specific step by ID.

        Args:
            step_id: The ID of the step to jump to (e.g., "step_5").

        Returns:
            True if jump was successful, False if step not found.
        """
        for i, step in enumerate(self.config.steps):
            if step.id == step_id:
                self.current_step_index = i
                return True
        return False

    def jump_to_iteration(self, iteration_id: str, step_offset: int = 0) -> bool:
        """Jump to a specific step within an iteration.

        Args:
            iteration_id: The iteration ID (e.g., "iteration_1" or "1").
            step_offset: The step offset within the iteration (0-based, default 0).

        Returns:
            True if jump was successful, False if iteration not found.
        """
        # Support both "iteration_1" and "1" formats
        normalized_id = iteration_id
        if not iteration_id.startswith("iteration_"):
            normalized_id = f"iteration_{iteration_id}"

        for iteration in self._iterations:
            if iteration.id == normalized_id:
                target_step = iteration.start_step_index + step_offset
                # Ensure we don't go past the iteration's end
                if target_step > iteration.end_step_index:
                    target_step = iteration.end_step_index
                return self.jump_to_step(target_step)
        return False

    def get_steps_for_iteration(self, iteration_id: str) -> List[StepConfig]:
        """Get all steps within a specific iteration.

        Args:
            iteration_id: The iteration ID (e.g., "iteration_1" or "1").

        Returns:
            List of StepConfig objects for the iteration, or empty list if not found.
        """
        # Support both "iteration_1" and "1" formats
        normalized_id = iteration_id
        if not iteration_id.startswith("iteration_"):
            normalized_id = f"iteration_{iteration_id}"

        for iteration in self._iterations:
            if iteration.id == normalized_id:
                return self.config.steps[
                    iteration.start_step_index : iteration.end_step_index + 1
                ]
        return []

    def read_file(self, relative_path: str) -> str:
        """Read a file from the flow directory.

        Tries filesystem path first, then package resources.
        """
        # Try filesystem first
        full_path = os.path.join(self.base_path, relative_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()

        # Try package resources using stored flow_name
        try:
            flow_name = self.flow_name or os.path.basename(self.base_path)
            package = "chatbot_demo_react"
            resource_path = f"experiment_configs/{flow_name}/{relative_path}"

            print(f"[ExperimentFlowEngine] Trying package resource: {resource_path}")

            with importlib.resources.files(package).joinpath(resource_path).open(
                "r"
            ) as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find file '{relative_path}' at {full_path} "
                f"or in package resources (experiment_configs/{self.flow_name}/{relative_path}): {e}"
            )

    def get_context_files_up_to_step(
        self, step_index: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Extract file paths from mock flow messages up to (and including) the specified step.

        This method collects all files referenced in messages from step 0 to step_index.
        DevMate will read the file contents itself, so only paths are provided.

        In Buck2 builds, uses importlib.resources to get the actual resource paths.

        Args:
            step_index: The step index up to which to collect files.
                       If None, uses current_step_index.

        Returns:
            List of dicts with 'name' and 'path' keys for each referenced file.

        Example:
            >>> engine = ExperimentFlowEngine(config, base_path)
            >>> context_files = engine.get_context_files_up_to_step(2)
            >>> for f in context_files:
            ...     print(f"{f['name']}: {f['path']}")
        """
        if step_index is None:
            step_index = self.current_step_index

        context_files = []
        seen_files: set = set()  # Avoid duplicates

        for i in range(min(step_index + 1, len(self.config.steps))):
            step = self.config.steps[i]
            for msg in step.messages:
                if isinstance(msg, ParallelGroup):
                    for item in msg.items:
                        if item.file and item.file not in seen_files:
                            full_path = self._resolve_file_path(item.file)
                            context_files.append(
                                {
                                    "name": os.path.basename(item.file),
                                    "path": full_path,
                                }
                            )
                            seen_files.add(item.file)
                elif hasattr(msg, "file") and msg.file and msg.file not in seen_files:
                    full_path = self._resolve_file_path(msg.file)
                    context_files.append(
                        {
                            "name": os.path.basename(msg.file),
                            "path": full_path,
                        }
                    )
                    seen_files.add(msg.file)

        return context_files

    def _resolve_file_path(self, relative_path: str) -> str:
        """
        Resolve a relative file path to a clean fbcode path.

        Uses context_files_root from config if available (simplest approach).
        Otherwise falls back to filesystem/package resource detection.

        Args:
            relative_path: Path relative to the flow directory.

        Returns:
            Clean path starting from 'fbcode/...' or the original path.
        """
        # Simple approach: use context_files_root from config if available
        if self.config.context_files_root:
            # Just combine the root with the relative path
            # e.g., "fbcode/_tony_dev/.../coscience_experiment" + "files/context/proposals/x.md"
            return f"{self.config.context_files_root}/{relative_path}"

        # Fallback: Try filesystem first
        full_path = os.path.join(self.base_path, relative_path)
        if os.path.exists(full_path):
            return self._clean_fbcode_path(full_path)

        # Fallback: Try package resources (Buck2 build mode)
        try:
            flow_name = self.flow_name or os.path.basename(self.base_path)
            package = "chatbot_demo_react"
            resource_path = f"experiment_configs/{flow_name}/{relative_path}"

            resource = importlib.resources.files(package).joinpath(resource_path)
            resolved_path = str(resource)
            return self._clean_fbcode_path(resolved_path)
        except Exception as e:
            print(
                f"[ExperimentFlowEngine] Warning: Could not resolve path for '{relative_path}': {e}"
            )
            return self._clean_fbcode_path(full_path)

    def _clean_fbcode_path(self, path: str) -> str:
        """
        Clean a Buck2 build path to a clean fbcode path.

        Buck2 paths look like:
        buck-out/v2/gen/fbcode/<hash>/_tony_dev/ScienceModelingTools/...

        We want to return:
        fbcode/_tony_dev/ScienceModelingTools/...

        Args:
            path: Full path that may contain buck-out prefixes

        Returns:
            Clean path starting from 'fbcode/...'
        """
        # Find _tony_dev/ which marks the start of the actual source path under fbcode
        # This handles Buck2 paths like: .../fbcode/<hash>/_tony_dev/...
        idx = path.find("/_tony_dev/")
        if idx != -1:
            # Return "fbcode/_tony_dev/..."
            return "fbcode" + path[idx:]

        # Also check without leading slash (for paths starting with _tony_dev/)
        if path.startswith("_tony_dev/"):
            return "fbcode/" + path

        # Fall back to looking for the last /fbcode/ in standard paths
        last_fbcode_idx = path.rfind("/fbcode/")
        if last_fbcode_idx != -1:
            return path[last_fbcode_idx + 1 :]

        # Check if path starts with 'fbcode/'
        if path.startswith("fbcode/"):
            return path

        # No fbcode found, return original
        return path


class ExperimentFlowLoader:
    """Loads and validates experiment flow JSON files."""

    @staticmethod
    def load(flow_name: str, base_dir: str) -> tuple[ExperimentFlowConfig, str]:
        """Load flow from experiment_configs/<flow_name>/flow.json

        Tries multiple locations:
        1. First tries the filesystem path (for development)
        2. Falls back to package resources (for Buck builds)
        """
        # Try filesystem first (development mode)
        flow_dir = os.path.join(base_dir, "experiment_configs", flow_name)
        flow_path = os.path.join(flow_dir, "flow.json")

        if os.path.exists(flow_path):
            with open(flow_path, "r") as f:
                data = json.load(f)
            config = ExperimentFlowLoader._parse_config(data)
            return config, flow_dir

        # Try package resources (Buck build mode)
        try:
            # In Buck builds, resources are available via importlib.resources
            package = "chatbot_demo_react"
            resource_path = f"experiment_configs/{flow_name}/flow.json"

            # Use importlib.resources to get the resource file
            with importlib.resources.files(package).joinpath(resource_path).open(
                "r"
            ) as f:
                data = json.load(f)

            # Get the base path for the flow directory
            flow_dir_resource = importlib.resources.files(package).joinpath(
                f"experiment_configs/{flow_name}"
            )
            flow_dir = str(flow_dir_resource)

            config = ExperimentFlowLoader._parse_config(data)
            return config, flow_dir
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find experiment flow '{flow_name}' at {flow_path} "
                f"or in package resources: {e}"
            )

    @staticmethod
    def _parse_config(data: Dict) -> ExperimentFlowConfig:
        # Parse metadata
        metadata = data.get("metadata", {})
        name = metadata.get("name", "Experiment Flow")
        description = metadata.get("description", "")

        # Parse default inferencer config (optional)
        default_inferencer = None
        default_inf_data = metadata.get("default_inferencer")
        if default_inf_data:
            default_inferencer = DefaultInferencerConfig(
                type=default_inf_data.get("type", "devmate"),
                config=default_inf_data.get("config", {}),
            )

        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            step = ExperimentFlowLoader._parse_step(step_data)
            steps.append(step)

        return ExperimentFlowConfig(
            name=name,
            description=description,
            steps=steps,
            default_inferencer=default_inferencer,
            code_entry_point=metadata.get("code_entry_point"),
            context_files_root=metadata.get("context_files_root"),
            metadata=metadata,
        )

    @staticmethod
    def _parse_step(data: Dict) -> StepConfig:
        messages = []
        for msg_data in data.get("messages", []):
            if isinstance(msg_data, dict) and msg_data.get("parallel"):
                items = [
                    ExperimentFlowLoader._parse_message(m) for m in msg_data["items"]
                ]
                messages.append(ParallelGroup(items=items))
            else:
                messages.append(ExperimentFlowLoader._parse_message(msg_data))

        # Parse pre_messages
        pre_messages = []
        for msg_data in data.get("pre_messages", []):
            if isinstance(msg_data, dict) and msg_data.get("parallel"):
                items = [
                    ExperimentFlowLoader._parse_message(m) for m in msg_data["items"]
                ]
                pre_messages.append(ParallelGroup(items=items))
            else:
                pre_messages.append(ExperimentFlowLoader._parse_message(msg_data))

        # Parse post_messages
        post_messages = []
        for msg_data in data.get("post_messages", []):
            if isinstance(msg_data, dict) and msg_data.get("parallel"):
                items = [
                    ExperimentFlowLoader._parse_message(m) for m in msg_data["items"]
                ]
                post_messages.append(ParallelGroup(items=items))
            else:
                post_messages.append(ExperimentFlowLoader._parse_message(msg_data))

        # Parse pre_delay and post_delay
        pre_delay = data.get("pre_delay", {})
        post_delay = data.get("post_delay", {})

        # Parse suggested actions (optional) - supports both old array format and new object format
        suggested_actions = []
        suggested_actions_message = None
        suggested_actions_data = data.get("suggested_actions")
        if suggested_actions_data:
            # New format: object with "message" and "actions" keys
            if isinstance(suggested_actions_data, dict):
                suggested_actions_message = suggested_actions_data.get("message")
                actions_list = suggested_actions_data.get("actions", [])
                for action_data in actions_list:
                    suggested_actions.append(
                        ExperimentFlowLoader._parse_suggested_action(action_data)
                    )
            # Old format: array of actions
            elif isinstance(suggested_actions_data, list):
                for action_data in suggested_actions_data:
                    suggested_actions.append(
                        ExperimentFlowLoader._parse_suggested_action(action_data)
                    )

        # Parse progress_sections (array of sections) with backward compatibility
        progress_sections = []
        progress_data = data.get("progress_sections", [])
        # Support legacy "progress_section" (singular) for backward compatibility
        if not progress_data and "progress_section" in data:
            progress_data = [data["progress_section"]]
        for section_data in progress_data:
            progress_sections.append(
                ExperimentFlowLoader._parse_progress_section(section_data)
            )

        # Parse progress_header if present
        progress_header = None
        progress_header_data = data.get("progress_header")
        if progress_header_data:
            progress_header = ExperimentFlowLoader._parse_progress_header(
                progress_header_data
            )

        return StepConfig(
            id=data.get("id", ""),
            role=data.get("role", "assistant"),
            messages=messages,
            wait_for_user=data.get("wait_for_user", False),
            pre_delay_min=pre_delay.get("min", 2.0),
            pre_delay_max=pre_delay.get("max", 5.0),
            post_delay_min=post_delay.get("min", 0.0),
            post_delay_max=post_delay.get("max", 0.0),
            suggested_actions=suggested_actions,
            suggested_actions_message=suggested_actions_message,
            progress_sections=progress_sections,
            progress_header=progress_header,
            pre_messages=pre_messages,
            post_messages=post_messages,
            keep_progress_sections=data.get("keep_progress_sections", False),
            post_messages_only=data.get("post_messages_only", False),
        )

    @staticmethod
    def _parse_progress_section(data: Dict) -> ProgressSection:
        """Parse a progress_section configuration."""
        messages = []
        for msg_data in data.get("messages", []):
            messages.append(
                ProgressMessage(
                    content=msg_data.get("content", ""),
                    delay=msg_data.get("delay", 1.0),
                    status=msg_data.get("status", "pending"),
                )
            )

        # Parse appearance_delay (can be number or {min, max} dict)
        appearance_delay = data.get("appearance_delay", 0.0)
        if isinstance(appearance_delay, dict):
            appearance_delay_min = appearance_delay.get("min", 0.0)
            appearance_delay_max = appearance_delay.get("max", appearance_delay_min)
        else:
            appearance_delay_min = float(appearance_delay) if appearance_delay else 0.0
            appearance_delay_max = appearance_delay_min

        # Parse task_progress specific fields
        section_type = data.get("type")  # e.g., "task_progress"
        tasks = None
        if section_type == "task_progress":
            tasks_data = data.get("tasks", [])
            tasks = []
            for task_data in tasks_data:
                task_messages = []
                for msg_data in task_data.get("messages", []):
                    task_messages.append(
                        ProgressMessage(
                            content=msg_data.get("content", ""),
                            delay=msg_data.get("delay", 1.0),
                            status=msg_data.get("status", "pending"),
                        )
                    )
                tasks.append(
                    TaskConfig(
                        id=task_data.get("id", ""),
                        title=task_data.get("title", ""),
                        messages=task_messages,
                        result=task_data.get("result"),
                    )
                )

        return ProgressSection(
            slot=data.get("slot", "thinking"),
            title=data.get("title", "Thinking..."),
            messages=messages,
            collapsible=data.get("collapsible", True),
            initial_state=data.get("initial_state", "expanded"),
            show_history=data.get("show_history", True),
            message_delay_multiplier=data.get("message_delay_multiplier", 1.0),
            appearance_delay_min=appearance_delay_min,
            appearance_delay_max=appearance_delay_max,
            prompt_file=data.get("prompt_file"),
            enabled=data.get(
                "enabled", True
            ),  # Default to True for backward compatibility
            section_type=section_type,
            tasks=tasks,
        )

    @staticmethod
    def _parse_suggested_action(data: Dict) -> SuggestedAction:
        """Parse a suggested action configuration including inferencer settings."""
        # Parse inferencer config if present
        inferencer_config = None
        inf_data = data.get("inferencer_config")
        if inf_data:
            inferencer_config = InferencerActionConfig(
                type=inf_data.get("type", "devmate"),
                prompt=inf_data.get("prompt"),  # Template name (e.g., "devmate_prompt")
                prompt_template=inf_data.get("prompt_template"),  # Inline template
                engine=inf_data.get("engine"),  # Template engine (e.g., "jinja")
                model_name=inf_data.get("model_name"),
                max_tokens=inf_data.get("max_tokens"),
                context_files=inf_data.get("context_files"),
            )

        # Parse return_to_flow config if present
        return_to_flow = None
        rtf_data = data.get("return_to_flow")
        if rtf_data:
            return_to_flow = ReturnToFlowConfig(
                label=rtf_data.get("label", "↩️ Return to Demo Flow"),
                style=rtf_data.get("style", "primary"),
            )

        return SuggestedAction(
            label=data.get("label", ""),
            action_type=data.get("action_type", "continue"),
            prefix=data.get("prefix"),
            style=data.get("style", "primary"),
            user_message=data.get("user_message"),  # New: user message to display
            inferencer_config=inferencer_config,
            return_to_flow=return_to_flow,
            target_step=data.get("target_step"),  # For branch_with_input
            input_config=data.get("input_config"),  # For branch_with_input
        )

    @staticmethod
    def _parse_message(data: Dict) -> MessageConfig:
        delay = data.get("delay", {})

        # Parse input_field if present
        input_field = None
        input_field_data = data.get("input_field")
        if input_field_data:
            input_field = InputFieldConfig(
                variable_name=input_field_data.get("variable_name", "user_input"),
                placeholder=input_field_data.get("placeholder", "Enter your input..."),
                multiline=input_field_data.get("multiline", False),
                optional=input_field_data.get("optional", True),
                default_value=input_field_data.get("default_value", ""),
                collapsible=input_field_data.get("collapsible", False),
                initially_collapsed=input_field_data.get("initially_collapsed", False),
                mode_selector=input_field_data.get("mode_selector"),
            )

        # Parse editable_list if present (pass through as-is for frontend)
        editable_list = data.get("editable_list")

        return MessageConfig(
            content=data.get("content", ""),
            type=data.get("type", "text"),
            delay_min=delay.get("min", 0.5),
            delay_max=delay.get("max", 2.0),
            file=data.get("file"),
            input_field=input_field,
            editable_list=editable_list,
        )

    @staticmethod
    def _parse_progress_header(data: Dict) -> ProgressHeaderConfig:
        """Parse a progress_header configuration."""
        # Parse input_field if present
        input_field = None
        input_field_data = data.get("input_field")
        if input_field_data:
            input_field = InputFieldConfig(
                variable_name=input_field_data.get("variable_name", "user_input"),
                placeholder=input_field_data.get("placeholder", "Enter your input..."),
                multiline=input_field_data.get("multiline", False),
                optional=input_field_data.get("optional", True),
                default_value=input_field_data.get("default_value", ""),
            )

        # Parse continue_button if present
        continue_button = ContinueButtonConfig()
        continue_button_data = data.get("continue_button")
        if continue_button_data:
            continue_button = ContinueButtonConfig(
                label=continue_button_data.get("label", "🚀 Continue"),
                style=continue_button_data.get("style", "primary"),
            )

        return ProgressHeaderConfig(
            content=data.get("content", ""),
            input_field=input_field,
            continue_button=continue_button,
        )
