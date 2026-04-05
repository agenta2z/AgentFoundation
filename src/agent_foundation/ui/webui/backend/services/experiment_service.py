# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
Experiment Service - Wraps existing experiment_engine.py for FastAPI.

This service provides a clean interface for the FastAPI routes to interact
with the experiment flow logic from the original Dash app.

The key improvement over the Dash version is TIME-BASED progress animation:
- When a step with progress_sections starts, we store start_time_ms and
  pre-compute cumulative_delays_ms for each message
- get_progress_state() calculates revealed_counts from elapsed time (stateless)
- This allows the WebSocket to push updates without maintaining complex state
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatbot_demo_react.experiment_engine import (
    TaskConfig,
    ContinueButtonConfig,
    ExperimentFlowConfig,
    ExperimentFlowEngine,
    ExperimentFlowLoader,
    InputFieldConfig,
    IterationConfig,
    MessageConfig,
    ParallelGroup,
    ProgressHeaderConfig,
    ProgressSection,
    StepConfig,
)

logger = logging.getLogger(__name__)

# Startup message to verify Buck is loading the correct version
print("=" * 60)
print("EXPERIMENT_SERVICE.PY LOADED - prompt_file support enabled!")
print("Using local experiment_engine from chatbot_demo_react")
print("=" * 60)


@dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str  # "user" or "assistant"
    content: str
    file_path: str | None = None
    message_type: str = "text"  # "text" or "progress"


@dataclass
class ProgressState:
    """State of progress animation for real-time updates.

    Uses TIME-BASED animation with phases:
    1. pre_delay: Wait before showing pre_messages
    2. pre_messages: Show messages sequentially, each REPLACES previous
    3. progress_header: Show progress header with input field, wait for user to continue
    4. progress: Progress sections animation
    5. post_delay: Wait after progress completes
    6. post_messages: Show messages sequentially, each APPENDS (not replaces)
    7. complete: Animation finished

    Attributes:
        sections: List of section configs with messages
        revealed_counts: Dict mapping section slot to number of revealed messages
        is_animating: Whether animation is currently running
        current_step_id: ID of the step being animated
        start_time_ms: Timestamp (in ms) when animation started
        cumulative_delays_ms: Pre-computed delays for each section's messages
        total_duration_ms: Total animation duration (longest section)
        is_complete: Whether all sections have completed
        phase: Current animation phase
        pre_messages: List of pre_messages for this step
        post_messages: List of post_messages for this step
        pre_delay_ms: Delay before pre_messages start
        post_delay_ms: Delay after progress before post_messages
        pre_message_delays_ms: Cumulative delays for pre_messages
        post_message_delays_ms: Cumulative delays for post_messages
        current_pre_message_index: Which pre_message is currently showing
        current_post_message_count: How many post_messages are revealed
        phase_start_time_ms: When the current phase started
        keep_progress_sections: Whether to keep progress sections visible after animation completes
        progress_header: Progress header config (content, input_field, continue_button)
        waiting_for_progress_header: Whether we're waiting for user to click continue
    """

    sections: list[dict[str, Any]] = field(default_factory=list)
    revealed_counts: dict[str, int] = field(default_factory=dict)
    is_animating: bool = False
    current_step_id: str = ""
    # Time-based animation fields
    start_time_ms: float = 0.0
    cumulative_delays_ms: dict[str, list[float]] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    is_complete: bool = False
    # Phase-based animation fields
    phase: str = "idle"  # idle, pre_delay, pre_messages, progress_header, progress, post_delay, post_messages, complete
    pre_messages: list[dict[str, Any]] = field(default_factory=list)
    post_messages: list[dict[str, Any]] = field(default_factory=list)
    pre_delay_ms: float = 0.0
    post_delay_ms: float = 0.0
    pre_message_delays_ms: list[float] = field(default_factory=list)
    post_message_delays_ms: list[float] = field(default_factory=list)
    current_pre_message_index: int = -1  # -1 means no pre_message shown yet
    current_post_message_count: int = 0
    phase_start_time_ms: float = 0.0
    keep_progress_sections: bool = (
        False  # Whether to keep progress sections visible after animation
    )
    group_sections_by_step: bool = (
        True  # UI config: whether to group sections by step (default: true)
    )
    # Progress header fields
    progress_header: dict[str, Any] | None = None  # Progress header config
    waiting_for_progress_header: bool = (
        False  # Whether we're waiting for user to click continue
    )


@dataclass
class ExperimentState:
    """Full state of the experiment for frontend."""

    current_step_index: int
    current_step_id: str
    is_complete: bool
    messages: list[ChatMessage]
    suggested_actions: dict[
        str, Any
    ]  # Format: {message: str | None, actions: list[dict]}
    progress_state: ProgressState
    is_waiting_for_user: bool


class ExperimentService:
    """Service wrapper for ExperimentFlowEngine.

    This provides an async-friendly interface for FastAPI routes.

    Key improvement over Dash version: TIME-BASED progress animation.
    Instead of tracking state with async sleep, we:
    1. Store start_time_ms and pre-compute cumulative_delays_ms when animation starts
    2. get_progress_state() calculates revealed_counts from elapsed time (stateless)
    3. WebSocket pushes updates every 200ms with calculated state
    """

    def __init__(
        self, flow_name: str = "coscience_experiment", delay_multiplier: float = 1.0
    ) -> None:
        """Initialize the experiment service with a specific flow.

        Args:
            flow_name: Name of the experiment flow to load.
            delay_multiplier: Multiplier for all delays (e.g., 0.5 = 2x faster, 0.2 = 5x faster).
        """
        self.flow_name = flow_name
        self.delay_multiplier = delay_multiplier
        self._engine: ExperimentFlowEngine | None = None
        self._config: ExperimentFlowConfig | None = None
        self._base_path: str = ""
        self._messages: list[ChatMessage] = []
        self._progress_state = ProgressState()
        self._animation_task: asyncio.Task[None] | None = None
        # Track if we need to process step messages after animation completes
        self._pending_step_messages: bool = False

        # Initialize the engine
        self._initialize_engine()

    def _scale_delay(self, delay_sec: float) -> float:
        """Scale a delay value by the delay multiplier.

        Args:
            delay_sec: Delay in seconds.

        Returns:
            Scaled delay in seconds.
        """
        return delay_sec * self.delay_multiplier

    def _initialize_engine(self) -> None:
        """Load the experiment flow configuration."""
        # Use local experiment_configs in chatbot_demo_react
        # Path: backend/services/experiment_service.py -> backend/services -> backend -> chatbot_demo_react
        chatbot_demo_react_dir = Path(__file__).resolve().parent.parent.parent
        experiment_configs_dir = chatbot_demo_react_dir / "experiment_configs"

        if not experiment_configs_dir.exists():
            # Fallback: try chatbot_demo directory (for backward compatibility)
            chatbot_demo_dir = chatbot_demo_react_dir.parent / "chatbot_demo"
            if chatbot_demo_dir.exists():
                logger.warning(
                    f"Using fallback chatbot_demo directory: {chatbot_demo_dir}"
                )
                base_dir = str(chatbot_demo_dir)
            else:
                raise FileNotFoundError(
                    f"experiment_configs not found at {experiment_configs_dir} "
                    f"or {chatbot_demo_dir}"
                )
        else:
            base_dir = str(chatbot_demo_react_dir)

        logger.info(f"Loading flow '{self.flow_name}' from {base_dir}")

        try:
            self._config, self._base_path = ExperimentFlowLoader.load(
                self.flow_name, base_dir
            )
            self._engine = ExperimentFlowEngine(
                self._config, self._base_path, self.flow_name
            )
            logger.info(
                f"Loaded flow '{self._config.name}' with {len(self._config.steps)} steps"
            )
        except Exception as e:
            logger.error(f"Failed to load flow '{self.flow_name}': {e}")
            raise

    def reset(self) -> None:
        """Reset the experiment to the beginning."""
        self._messages = []
        self._progress_state = ProgressState()
        if self._animation_task:
            self._animation_task.cancel()
            self._animation_task = None
        self._initialize_engine()

    def get_current_step(self) -> StepConfig | None:
        """Get the current step configuration."""
        if self._engine is None:
            return None
        return self._engine.get_current_step()

    def get_state(self) -> ExperimentState:
        """Get the full experiment state for the frontend."""
        step = self.get_current_step()

        # Build suggested_actions in new format: {message, actions}
        suggested_actions_list = []
        suggested_actions_message = None
        if step and step.suggested_actions:
            suggested_actions_list = [
                {
                    "label": action.label,
                    "action_type": action.action_type,
                    "style": action.style,
                    "prefix": action.prefix,
                    "user_message": action.user_message,
                    "target_step": action.target_step,
                    "input_config": action.input_config,
                }
                for action in step.suggested_actions
            ]
            suggested_actions_message = step.suggested_actions_message

        suggested_actions = {
            "message": suggested_actions_message,
            "actions": suggested_actions_list,
        }

        return ExperimentState(
            current_step_index=self._engine.current_step_index if self._engine else 0,
            current_step_id=step.id if step else "",
            is_complete=self._engine.is_complete() if self._engine else True,
            messages=self._messages,
            suggested_actions=suggested_actions,
            progress_state=self._progress_state,
            is_waiting_for_user=step.wait_for_user if step else False,
        )

    async def process_user_input(self, user_input: str) -> ExperimentState:
        """Process user input and advance the experiment.

        Returns the updated experiment state.
        """
        # Add user message
        self._messages.append(ChatMessage(role="user", content=user_input))

        # Process current step
        await self._process_current_step()

        return self.get_state()

    async def handle_action(self, action_index: int) -> ExperimentState:
        """Handle a suggested action click.

        Returns the updated experiment state.
        """
        step = self.get_current_step()
        if not step or action_index >= len(step.suggested_actions):
            return self.get_state()

        action = step.suggested_actions[action_index]

        if action.action_type == "continue":
            # Check if there's a target_step specified
            target_step = getattr(action, "target_step", None)
            if target_step and self._engine:
                # Jump to specific step by ID
                self._engine.jump_to_step_by_id(target_step)
            elif self._engine:
                # Advance to next step
                self._engine.advance_step()
            await self._process_current_step()

        elif action.action_type == "input_prefix":
            # Add prefix as user input and advance
            if action.prefix:
                self._messages.append(ChatMessage(role="user", content=action.prefix))
            if self._engine:
                self._engine.advance_step()
            await self._process_current_step()

        return self.get_state()

    async def handle_branch_action(
        self, action_index: int, input_value: str, target_step: str | None
    ) -> ExperimentState:
        """Handle a branch action with user input.

        This navigates to a specific branch step and starts its animation.

        Args:
            action_index: Index of the action that triggered this branch
            input_value: User input from the branch input field
            target_step: ID of the step to navigate to

        Returns the updated experiment state.
        """
        # Add user message showing the branch request
        user_message = f"🔍 Deep dive on: {input_value}"
        self._messages.append(ChatMessage(role="user", content=user_message))

        # Navigate to the target branch step
        if target_step and self._engine:
            self._engine.jump_to_step_by_id(target_step)

        # Process the branch step
        await self._process_current_step()

        return self.get_state()

    async def _process_current_step(self) -> None:
        """Process the current step, adding messages and handling progress animation.

        IMPORTANT: If a step has progress_sections, we start the animation and
        RETURN IMMEDIATELY. We do NOT add step messages here - the WebSocket
        will handle sending step messages AFTER the animation completes.

        This matches the Dash version's behavior where:
        1. JS poller handles progress UI via /api/progress-animation
        2. Step messages are only sent via WebSocket when is_complete=true

        Special case: post_messages_only steps skip progress animation and only
        show post_messages after a delay.
        """
        step = self.get_current_step()
        if not step:
            return

        # Handle post_messages_only steps (no progress animation, just show post_messages)
        if step.post_messages_only:
            await self._start_post_messages_only_animation(step)
            return  # Exit early - let WebSocket handle the rest

        # Handle steps with progress sections OR steps with pre_messages (even if progress_sections is empty [])
        # Note: Empty list [] is falsy in Python, so we check for pre_messages as well
        if step.progress_sections or step.pre_messages:
            # Start animation and RETURN IMMEDIATELY
            # DO NOT add step messages here - WebSocket handles after completion
            await self._animate_progress_sections(step.progress_sections or [])
            return  # Exit early - let WebSocket handle the rest

        # Only add step messages if NO progress sections
        for msg in step.messages:
            if isinstance(msg, ParallelGroup):
                for item in msg.items:
                    self._add_message_from_config(item, step.role)
            elif isinstance(msg, MessageConfig):
                self._add_message_from_config(msg, step.role)

        # If step doesn't wait for user, auto-advance
        if not step.wait_for_user and self._engine:
            self._engine.advance_step()
            # Recursively process next step (with delay)
            await asyncio.sleep(0.5)
            await self._process_current_step()

    async def _start_post_messages_only_animation(self, step: StepConfig) -> None:
        """Start animation for post_messages_only steps.

        These steps skip progress sections but can still show pre_messages before
        post_messages. Uses the same phase-based animation system.

        Animation phases for post_messages_only:
        1. pre_delay: Wait before showing pre_messages
        2. pre_messages: Show pre_messages sequentially (if any)
        3. post_delay: Wait after pre_messages before post_messages
        4. post_messages: Show post_messages sequentially
        5. complete: Animation finished

        Args:
            step: The step config with post_messages_only=True
        """
        step_id = step.id

        # Get pre_delay and post_delay from step config
        pre_delay_sec = step.get_pre_delay()
        post_delay_sec = step.get_post_delay()

        # Get pre_messages and post_messages
        pre_messages_config: list[Any] = getattr(step, "pre_messages", []) or []
        post_messages_config: list[Any] = step.post_messages or []

        logger.info(
            f"Starting post_messages_only animation for step '{step_id}': "
            f"pre_delay={pre_delay_sec:.2f}s, post_delay={post_delay_sec:.2f}s, "
            f"pre_messages_count={len(pre_messages_config)}, "
            f"post_messages_count={len(post_messages_config)}"
        )

        # Build pre_messages list with cumulative delays
        pre_messages_list: list[dict[str, Any]] = []
        pre_message_delays_ms: list[float] = []
        cumulative_pre_delay = 0.0

        for msg in pre_messages_config:
            if isinstance(msg, MessageConfig):
                pre_messages_list.append(self._message_config_to_dict(msg, step.role))
                pre_message_delays_ms.append(cumulative_pre_delay)
                msg_delay = msg.get_delay()
                cumulative_pre_delay += msg_delay * 1000
            elif isinstance(msg, ParallelGroup):
                for item in msg.items:
                    pre_messages_list.append(
                        self._message_config_to_dict(item, step.role)
                    )
                    pre_message_delays_ms.append(cumulative_pre_delay)
                if msg.items:
                    msg_delay = msg.items[0].get_delay()
                    cumulative_pre_delay += msg_delay * 1000

        # Build post_messages list with cumulative delays
        post_messages_list: list[dict[str, Any]] = []
        post_message_delays_ms: list[float] = []
        cumulative_post_delay = 0.0

        for msg in post_messages_config:
            if isinstance(msg, MessageConfig):
                post_messages_list.append(self._message_config_to_dict(msg, step.role))
                post_message_delays_ms.append(cumulative_post_delay)
                msg_delay = msg.get_delay()
                cumulative_post_delay += msg_delay * 1000
            elif isinstance(msg, ParallelGroup):
                for item in msg.items:
                    post_messages_list.append(
                        self._message_config_to_dict(item, step.role)
                    )
                    post_message_delays_ms.append(cumulative_post_delay)
                if msg.items:
                    msg_delay = msg.items[0].get_delay()
                    cumulative_post_delay += msg_delay * 1000

        # Calculate total duration
        total_duration_ms = (
            pre_delay_sec * 1000
            + cumulative_pre_delay
            + post_delay_sec * 1000
            + cumulative_post_delay
        )

        current_time_ms = time.time() * 1000

        # Determine starting phase based on delays and messages
        starting_phase = "pre_delay"
        if pre_delay_sec <= 0:
            if pre_messages_list:
                starting_phase = "pre_messages"
            else:
                starting_phase = "post_delay"
                if post_delay_sec <= 0:
                    starting_phase = "post_messages"

        self._progress_state = ProgressState(
            sections=[],  # No progress sections
            revealed_counts={},
            is_animating=True,
            current_step_id=step_id,
            start_time_ms=current_time_ms,
            cumulative_delays_ms={},
            total_duration_ms=total_duration_ms,
            is_complete=False,
            # Phase-based fields
            phase=starting_phase,
            pre_messages=pre_messages_list,
            post_messages=post_messages_list,
            pre_delay_ms=pre_delay_sec * 1000,
            post_delay_ms=post_delay_sec * 1000,
            pre_message_delays_ms=pre_message_delays_ms,
            post_message_delays_ms=post_message_delays_ms,
            current_pre_message_index=-1,
            current_post_message_count=0,
            phase_start_time_ms=current_time_ms,
            keep_progress_sections=False,
        )

        logger.info(
            f"Started post_messages_only animation for step '{step_id}', "
            f"total_duration={total_duration_ms:.0f}ms, starting_phase={starting_phase}, "
            f"pre_messages={len(pre_messages_list)}, post_messages={len(post_messages_list)}"
        )

    def _add_message_from_config(self, msg: MessageConfig, role: str) -> None:
        """Add a message from MessageConfig."""
        content = msg.content

        # Read file content if specified
        file_path = None
        if msg.file and self._engine:
            file_path = msg.file
            try:
                # Verify file exists by attempting to read it
                self._engine.read_file(msg.file)
                # For files, we might want to show a preview or just the path
                if msg.type == "text":
                    content = f"{content}\n\n📄 **File:** {msg.file}"
            except Exception as e:
                logger.warning(f"Could not read file {msg.file}: {e}")

        self._messages.append(
            ChatMessage(
                role=role,
                content=content,
                file_path=file_path,
                message_type=msg.type,
            )
        )

    def _parse_delay(self, delay_config: Any) -> float:
        """Parse a delay config which can be a number or {min, max} dict.

        Args:
            delay_config: Either a float/int or a dict with 'min' and 'max' keys.

        Returns:
            A random delay value in seconds.
        """
        import random

        if isinstance(delay_config, dict):
            min_val = delay_config.get("min", 0.0)
            max_val = delay_config.get("max", min_val)
            return random.uniform(min_val, max_val)
        return float(delay_config) if delay_config else 0.0

    def _progress_header_to_dict(self, header: ProgressHeaderConfig) -> dict[str, Any]:
        """Convert ProgressHeaderConfig to dict for JSON serialization."""
        result: dict[str, Any] = {
            "content": header.content,
        }

        # Include input_field if present
        if header.input_field:
            result["input_field"] = {
                "variable_name": header.input_field.variable_name,
                "placeholder": header.input_field.placeholder,
                "multiline": header.input_field.multiline,
                "optional": header.input_field.optional,
                "default_value": header.input_field.default_value,
                "collapsible": header.input_field.collapsible,
                "initially_collapsed": header.input_field.initially_collapsed,
                "mode_selector": header.input_field.mode_selector,
            }

        # Include continue_button
        result["continue_button"] = {
            "label": header.continue_button.label,
            "style": header.continue_button.style,
        }

        return result

    def continue_from_progress_header(
        self, user_input: dict[str, str] | None = None
    ) -> None:
        """Continue the animation from progress_header phase.

        Called when user clicks the continue button in the progress header.
        Stores any user input and transitions to the progress phase.

        CRITICAL FIX: This method resets start_time_ms to account for time spent
        waiting in progress_header phase. Without this, get_progress_state() would
        calculate elapsed_ms from the original animation start, causing the phase
        calculation to skip past "progress" phase immediately.

        Args:
            user_input: Optional dict of user input values (variable_name -> value)
        """
        # Check waiting_for_progress_header flag instead of phase
        # because phase is calculated dynamically in get_progress_state()
        # and self._progress_state.phase may not be updated
        if not self._progress_state.waiting_for_progress_header:
            logger.warning(
                f"continue_from_progress_header called but not waiting for header. "
                f"waiting_for_progress_header={self._progress_state.waiting_for_progress_header}, "
                f"phase={self._progress_state.phase}"
            )
            return

        # Store user input if provided
        if user_input:
            # Store in progress_state or elsewhere as needed
            logger.info(f"User input from progress_header: {user_input}")

        current_time_ms = time.time() * 1000

        # Calculate time spent before progress_header phase (pre_delay + pre_messages)
        # This is the elapsed time that get_progress_state() expects at the START of progress phase
        pre_delay_ms = self._progress_state.pre_delay_ms
        pre_message_delays = self._progress_state.pre_message_delays_ms
        pre_messages_duration_ms = pre_message_delays[-1] if pre_message_delays else 0.0

        # The time "before" progress phase is pre_delay + pre_messages duration
        time_before_progress_ms = pre_delay_ms + pre_messages_duration_ms

        # Reset start_time_ms so that elapsed_ms will equal time_before_progress_ms
        # when get_progress_state() is called immediately after this method returns.
        # This ensures the progress phase timing works correctly regardless of
        # how long the user waited in the progress_header phase.
        old_start_time_ms = self._progress_state.start_time_ms
        self._progress_state.start_time_ms = current_time_ms - time_before_progress_ms

        # Transition to progress phase
        self._progress_state.waiting_for_progress_header = False
        self._progress_state.phase = "progress"
        self._progress_state.phase_start_time_ms = current_time_ms

        logger.info(
            f"Continuing from progress_header to progress phase. "
            f"Reset timing: old_start={old_start_time_ms:.0f}ms, "
            f"new_start={self._progress_state.start_time_ms:.0f}ms, "
            f"time_before_progress={time_before_progress_ms:.0f}ms"
        )

    def start_progress_animation(self, sections: list[ProgressSection]) -> None:
        """Start time-based progress animation with phases.

        Animation phases:
        1. pre_delay: Wait before showing pre_messages
        2. pre_messages: Show messages sequentially, each REPLACES previous
        3. progress_header: Show progress header with input field, wait for user
        4. progress: Progress sections animation
        5. post_delay: Wait after progress completes
        6. post_messages: Show messages sequentially, each APPENDS (not replaces)
        7. complete: Animation finished

        Args:
            sections: List of ProgressSection configs from the current step.
        """
        # Filter out disabled sections
        enabled_sections = [s for s in sections if s.enabled]

        current_step = self.get_current_step()
        step_id = current_step.id if current_step else ""

        # Get pre_delay and post_delay from step config
        pre_delay_sec = 0.0
        post_delay_sec = 0.0
        pre_messages_config: list[Any] = []
        post_messages_config: list[Any] = []
        progress_header_config: ProgressHeaderConfig | None = None

        if current_step:
            # Use the step's get_pre_delay() and get_post_delay() methods
            # which return random values between min and max
            pre_delay_sec = current_step.get_pre_delay()
            post_delay_sec = current_step.get_post_delay()

            # Get pre_messages and post_messages
            pre_messages_config = getattr(current_step, "pre_messages", []) or []
            post_messages_config = getattr(current_step, "post_messages", []) or []

            # Get progress_header config
            progress_header_config = getattr(current_step, "progress_header", None)

        logger.info(
            f"Animation timing: pre_delay={pre_delay_sec:.2f}s, post_delay={post_delay_sec:.2f}s, "
            f"pre_messages_count={len(pre_messages_config)}, post_messages_count={len(post_messages_config)}, "
            f"has_progress_header={progress_header_config is not None}"
        )

        # Build pre_messages list with cumulative delays
        pre_messages_list: list[dict[str, Any]] = []
        pre_message_delays_ms: list[float] = []
        cumulative_pre_delay = 0.0

        for msg in pre_messages_config:
            if isinstance(msg, MessageConfig):
                pre_messages_list.append(self._message_config_to_dict(msg, "assistant"))
                # Each pre_message appears at cumulative_pre_delay, then stays for its delay duration
                pre_message_delays_ms.append(cumulative_pre_delay)
                # Use msg.get_delay() which returns random value between delay_min and delay_max
                # Apply delay_multiplier for speedup
                msg_delay = self._scale_delay(msg.get_delay())
                cumulative_pre_delay += msg_delay * 1000  # Convert to ms
            elif isinstance(msg, ParallelGroup):
                # Handle parallel group - show all items together
                for item in msg.items:
                    pre_messages_list.append(
                        self._message_config_to_dict(item, "assistant")
                    )
                    pre_message_delays_ms.append(cumulative_pre_delay)
                # For parallel groups, use get_delay() from first item
                if msg.items:
                    msg_delay = self._scale_delay(msg.items[0].get_delay())
                    cumulative_pre_delay += msg_delay * 1000

        # Build post_messages list with cumulative delays
        post_messages_list: list[dict[str, Any]] = []
        post_message_delays_ms: list[float] = []
        cumulative_post_delay = 0.0

        for msg in post_messages_config:
            if isinstance(msg, MessageConfig):
                post_messages_list.append(
                    self._message_config_to_dict(msg, "assistant")
                )
                post_message_delays_ms.append(cumulative_post_delay)
                # Use msg.get_delay() which returns random value between delay_min and delay_max
                # Apply delay_multiplier for speedup
                msg_delay = self._scale_delay(msg.get_delay())
                cumulative_post_delay += msg_delay * 1000
            elif isinstance(msg, ParallelGroup):
                for item in msg.items:
                    post_messages_list.append(
                        self._message_config_to_dict(item, "assistant")
                    )
                    post_message_delays_ms.append(cumulative_post_delay)
                # For parallel groups, use get_delay() from first item
                if msg.items:
                    msg_delay = self._scale_delay(msg.items[0].get_delay())
                    cumulative_post_delay += msg_delay * 1000

        # Pre-compute cumulative delays for progress sections
        cumulative_delays_ms: dict[str, list[float]] = {}
        total_progress_duration_ms = 0.0

        section_dicts = []
        for section in enabled_sections:
            slot = section.slot

            # Get message_delay_multiplier from section config (default 1.0)
            message_delay_multiplier = getattr(section, "message_delay_multiplier", 1.0)
            if message_delay_multiplier is None:
                message_delay_multiplier = 1.0

            # Calculate appearance_delay_ms (randomized between min and max)
            appearance_delay_min = getattr(section, "appearance_delay_min", 0.0) or 0.0
            appearance_delay_max = getattr(section, "appearance_delay_max", 0.0) or 0.0
            if appearance_delay_max < appearance_delay_min:
                appearance_delay_max = appearance_delay_min
            import random

            appearance_delay_sec = random.uniform(
                appearance_delay_min, appearance_delay_max
            )
            appearance_delay_ms = appearance_delay_sec * 1000

            # Check if this is a task_progress section
            section_type = getattr(section, "section_type", None)
            tasks = getattr(section, "tasks", None)

            if section_type == "task_progress" and tasks:
                # Handle task_progress section - process each benchmark individually
                task_dicts = []
                for task in tasks:
                    task_slot = f"{slot}_{task.id}"
                    bm_messages = task.messages

                    # Build cumulative delays for this benchmark
                    delays = [0.0]  # First message visible immediately
                    cumulative = 0.0
                    for msg in bm_messages[:-1]:
                        delay_sec = (
                            msg.delay * message_delay_multiplier * self.delay_multiplier
                        )
                        cumulative += delay_sec * 1000
                        delays.append(cumulative)

                    cumulative_delays_ms[task_slot] = delays

                    # Track total duration (longest benchmark)
                    if bm_messages:
                        last_delay = (
                            bm_messages[-1].delay * message_delay_multiplier * 1000
                        )
                        task_duration = cumulative + last_delay
                        total_progress_duration_ms = max(
                            total_progress_duration_ms, task_duration
                        )

                    task_dicts.append(
                        {
                            "id": task.id,
                            "title": task.title,
                            "messages": [
                                {
                                    "content": msg.content,
                                    "delay": msg.delay * message_delay_multiplier,
                                    "status": "pending",
                                }
                                for msg in bm_messages
                            ],
                            "result": task.result,
                        }
                    )

                section_dicts.append(
                    {
                        "slot": section.slot,
                        "title": section.title,
                        "type": section_type,
                        "tasks": task_dicts,
                        "collapsible": section.collapsible,
                        "initial_state": section.initial_state,
                        "appearance_delay_ms": appearance_delay_ms,
                        "prompt_file": section.prompt_file,
                    }
                )
                logger.info(
                    f"[DEBUG] Task progress section '{section.slot}' built with {len(tasks)} tasks"
                )
            else:
                # Handle regular progress section
                messages = section.messages

                # Build cumulative delays: message N visible at sum of delays 0..N-1
                delays = [0.0]  # First message visible immediately
                cumulative = 0.0
                for msg in messages[:-1]:
                    # Apply both message_delay_multiplier and the service-level delay_multiplier
                    delay_sec = (
                        msg.delay * message_delay_multiplier * self.delay_multiplier
                    )
                    cumulative += delay_sec * 1000
                    delays.append(cumulative)

                cumulative_delays_ms[slot] = delays

                # Track total duration (including appearance delay)
                if messages:
                    last_delay = messages[-1].delay * message_delay_multiplier * 1000
                    section_duration = appearance_delay_ms + cumulative + last_delay
                    total_progress_duration_ms = max(
                        total_progress_duration_ms, section_duration
                    )

                section_dicts.append(
                    {
                        "slot": section.slot,
                        "title": section.title,
                        "messages": [
                            {
                                "content": msg.content,
                                "delay": msg.delay * message_delay_multiplier,
                                "status": "pending",
                            }
                            for msg in section.messages
                        ],
                        "collapsible": section.collapsible,
                        "initial_state": section.initial_state,
                        "appearance_delay_ms": appearance_delay_ms,
                        "prompt_file": section.prompt_file,
                    }
                )
                # Debug: Log what we're sending to frontend
                logger.info(
                    f"[DEBUG] Section '{section.slot}' built with prompt_file='{section.prompt_file}'"
                )

        # Initialize revealed counts
        revealed_counts = {}
        for section in enabled_sections:
            section_type = getattr(section, "section_type", None)
            tasks = getattr(section, "tasks", None)
            if section_type == "task_progress" and tasks:
                # For task_progress, create revealed_counts for each benchmark
                for task in tasks:
                    task_slot = f"{section.slot}_{task.id}"
                    revealed_counts[task_slot] = 0
            else:
                revealed_counts[section.slot] = (
                    0  # Start at 0, will reveal first when progress phase starts
                )

        # Calculate total duration including all phases
        total_duration_ms = (
            pre_delay_sec * 1000
            + cumulative_pre_delay
            + total_progress_duration_ms
            + post_delay_sec * 1000
            + cumulative_post_delay
        )

        # Determine starting phase
        starting_phase = "pre_delay"
        if pre_delay_sec <= 0:
            if pre_messages_list:
                starting_phase = "pre_messages"
            else:
                starting_phase = "progress"

        current_time_ms = time.time() * 1000

        # Convert progress_header config to dict if present
        progress_header_dict = None
        has_progress_header = progress_header_config is not None
        if progress_header_config:
            progress_header_dict = self._progress_header_to_dict(progress_header_config)

        self._progress_state = ProgressState(
            sections=section_dicts,
            revealed_counts=revealed_counts,
            is_animating=True,
            current_step_id=step_id,
            start_time_ms=current_time_ms,
            cumulative_delays_ms=cumulative_delays_ms,
            total_duration_ms=total_duration_ms,
            is_complete=False,
            # Phase-based fields
            phase=starting_phase,
            pre_messages=pre_messages_list,
            post_messages=post_messages_list,
            pre_delay_ms=pre_delay_sec * 1000,
            post_delay_ms=post_delay_sec * 1000,
            pre_message_delays_ms=pre_message_delays_ms,
            post_message_delays_ms=post_message_delays_ms,
            current_pre_message_index=-1,
            current_post_message_count=0,
            phase_start_time_ms=current_time_ms,
            # Progress header fields
            progress_header=progress_header_dict,
            waiting_for_progress_header=has_progress_header,
        )

        logger.info(
            f"Started progress animation for step '{step_id}' with {len(sections)} sections, "
            f"pre_delay={pre_delay_sec:.1f}s, post_delay={post_delay_sec:.1f}s, "
            f"pre_messages={len(pre_messages_list)}, post_messages={len(post_messages_list)}, "
            f"total_duration={total_duration_ms:.0f}ms"
        )

    def get_progress_state(self) -> ProgressState:
        """Calculate current animation phase and state based on elapsed time.

        This is a STATELESS calculation - it computes the current phase and what
        should be visible based purely on (current_time - start_time).

        Animation phases:
        1. pre_delay: Wait before showing pre_messages
        2. pre_messages: Show messages sequentially, each REPLACES previous
        3. progress: Progress sections animation
        4. post_delay: Wait after progress completes
        5. post_messages: Show messages sequentially, each APPENDS (not replaces)
        6. complete: Animation finished

        Returns:
            ProgressState with calculated phase, revealed_counts, and message indices.
        """
        if not self._progress_state.is_animating:
            return self._progress_state

        start_time_ms = self._progress_state.start_time_ms
        current_time_ms = time.time() * 1000
        elapsed_ms = current_time_ms - start_time_ms

        # Calculate phase transitions based on elapsed time
        pre_delay_ms = self._progress_state.pre_delay_ms
        pre_message_delays = self._progress_state.pre_message_delays_ms
        post_delay_ms = self._progress_state.post_delay_ms
        post_message_delays = self._progress_state.post_message_delays_ms
        cumulative_delays_ms = self._progress_state.cumulative_delays_ms

        # Calculate total pre_messages duration (last delay + its duration)
        pre_messages_duration_ms = pre_message_delays[-1] if pre_message_delays else 0.0
        # Add the delay of the last pre_message to get total duration
        if self._progress_state.pre_messages:
            last_pre_msg = self._progress_state.pre_messages[-1]
            # Parse the delay for the last message (already parsed during start)
            pre_messages_duration_ms = (
                pre_message_delays[-1] if pre_message_delays else 0.0
            )

        # Calculate total progress duration
        progress_duration_ms = 0.0
        for section_dict in self._progress_state.sections:
            section_type = section_dict.get("type")

            if section_type == "task_progress":
                # For task_progress sections, calculate duration from each benchmark
                for bm in section_dict.get("tasks", []):
                    task_slot = f"{section_dict['slot']}_{bm['id']}"
                    delays = cumulative_delays_ms.get(task_slot, [0.0])
                    bm_messages = bm.get("messages", [])
                    if bm_messages and delays:
                        # Last message delay + its own delay
                        last_msg_delay = bm_messages[-1].get("delay", 0) * 1000
                        task_duration = (
                            delays[-1] + last_msg_delay
                            if len(delays) > 0
                            else last_msg_delay
                        )
                        progress_duration_ms = max(
                            progress_duration_ms, task_duration
                        )
            else:
                # Regular progress section
                slot = section_dict["slot"]
                delays = cumulative_delays_ms.get(slot, [0.0])
                messages = section_dict.get("messages", [])
                if messages and delays:
                    # Last message delay + its own delay
                    last_msg_delay = messages[-1].get("delay", 0) * 1000
                    section_duration = (
                        delays[-1] + last_msg_delay
                        if len(delays) > 0
                        else last_msg_delay
                    )
                    progress_duration_ms = max(progress_duration_ms, section_duration)

        # Calculate total post_messages duration
        post_messages_duration_ms = (
            post_message_delays[-1] if post_message_delays else 0.0
        )

        # Phase boundaries (cumulative)
        pre_delay_end = pre_delay_ms
        pre_messages_end = pre_delay_end + pre_messages_duration_ms
        progress_end = pre_messages_end + progress_duration_ms
        post_delay_end = progress_end + post_delay_ms
        post_messages_end = post_delay_end + post_messages_duration_ms

        # Debug logging for timing issues (only log occasionally to avoid spam)
        if elapsed_ms % 1000 < 200:  # Log roughly once per second
            logger.debug(
                f"get_progress_state timing: elapsed={elapsed_ms:.0f}ms, "
                f"boundaries=[pre_delay_end={pre_delay_end:.0f}, pre_msgs_end={pre_messages_end:.0f}, "
                f"progress_end={progress_end:.0f}, post_delay_end={post_delay_end:.0f}], "
                f"waiting_for_header={self._progress_state.waiting_for_progress_header}"
            )

        # Determine current phase and calculate state
        phase = "complete"
        current_pre_message_index = -1
        current_post_message_count = 0
        revealed_counts = {}

        if elapsed_ms < pre_delay_end:
            # Phase 1: pre_delay - waiting before pre_messages
            phase = "pre_delay"
            # No messages shown yet
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = 0
                else:
                    revealed_counts[section_dict["slot"]] = 0

        elif elapsed_ms < pre_messages_end:
            # Phase 2: pre_messages - showing pre_messages sequentially (replacing)
            phase = "pre_messages"
            phase_elapsed = elapsed_ms - pre_delay_end

            # Find which pre_message should be showing (each replaces previous)
            current_pre_message_index = -1
            for i, delay in enumerate(pre_message_delays):
                if phase_elapsed >= delay:
                    current_pre_message_index = i
                else:
                    break

            # No progress messages shown yet
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    # For task_progress, set revealed_counts for each benchmark
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = 0
                else:
                    revealed_counts[section_dict["slot"]] = 0

        elif self._progress_state.waiting_for_progress_header:
            # Phase 3: progress_header - waiting for user to click continue
            phase = "progress_header"

            # Show the last pre_message during progress_header phase
            if pre_message_delays:
                current_pre_message_index = len(pre_message_delays) - 1

            # No progress messages shown yet
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = 0
                else:
                    revealed_counts[section_dict["slot"]] = 0

        elif elapsed_ms < progress_end:
            # Phase 3: progress - showing progress sections
            phase = "progress"
            phase_elapsed = elapsed_ms - pre_messages_end

            # Show the last pre_message during progress phase
            if pre_message_delays:
                current_pre_message_index = len(pre_message_delays) - 1

            # Calculate revealed counts for progress sections
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")

                if section_type == "task_progress":
                    # For task_progress, calculate revealed counts for each benchmark
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        delays = cumulative_delays_ms.get(task_slot, [0.0])
                        total_messages = len(bm.get("messages", []))

                        # Count how many messages should be visible
                        count = sum(1 for delay in delays if phase_elapsed >= delay)
                        revealed_counts[task_slot] = min(count, total_messages)
                else:
                    # Regular progress section
                    slot = section_dict["slot"]
                    delays = cumulative_delays_ms.get(slot, [0.0])
                    total_messages = len(section_dict.get("messages", []))

                    # Get appearance_delay_ms for this section (default 0)
                    appearance_delay_ms = section_dict.get("appearance_delay_ms", 0.0)

                    # Adjust elapsed time by subtracting appearance delay
                    # Messages only start appearing after the section appears
                    section_elapsed = phase_elapsed - appearance_delay_ms

                    if section_elapsed < 0:
                        # Section hasn't appeared yet
                        revealed_counts[slot] = 0
                    else:
                        # Count how many messages should be visible at this elapsed time
                        count = sum(1 for delay in delays if section_elapsed >= delay)
                        revealed_counts[slot] = min(count, total_messages)

        elif elapsed_ms < post_delay_end:
            # Phase 4: post_delay - waiting after progress completes
            phase = "post_delay"

            # Show the last pre_message during post_delay
            if pre_message_delays:
                current_pre_message_index = len(pre_message_delays) - 1

            # All progress messages revealed
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = len(bm.get("messages", []))
                else:
                    slot = section_dict["slot"]
                    revealed_counts[slot] = len(section_dict.get("messages", []))

        elif elapsed_ms < post_messages_end:
            # Phase 5: post_messages - showing post_messages sequentially (appending)
            phase = "post_messages"
            phase_elapsed = elapsed_ms - post_delay_end

            # Pre_messages are replaced by post_messages
            current_pre_message_index = -1  # No pre_messages shown

            # Calculate how many post_messages to show (appending)
            current_post_message_count = 0
            for i, delay in enumerate(post_message_delays):
                if phase_elapsed >= delay:
                    current_post_message_count = i + 1
                else:
                    break

            # All progress messages revealed
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = len(bm.get("messages", []))
                else:
                    slot = section_dict["slot"]
                    revealed_counts[slot] = len(section_dict.get("messages", []))

        else:
            # Phase 6: complete
            phase = "complete"
            current_pre_message_index = -1  # No pre_messages
            current_post_message_count = len(self._progress_state.post_messages)

            # All progress messages revealed
            for section_dict in self._progress_state.sections:
                section_type = section_dict.get("type")
                if section_type == "task_progress":
                    for bm in section_dict.get("tasks", []):
                        task_slot = f"{section_dict['slot']}_{bm['id']}"
                        revealed_counts[task_slot] = len(bm.get("messages", []))
                else:
                    slot = section_dict["slot"]
                    revealed_counts[slot] = len(section_dict.get("messages", []))

        is_complete = phase == "complete"

        # Get keep_progress_sections from current step config
        keep_progress_sections = False
        current_step = self.get_current_step()
        if current_step:
            keep_progress_sections = getattr(
                current_step, "keep_progress_sections", False
            )

        # Get group_sections_by_step from flow config metadata (default: True)
        group_sections_by_step = True
        if self._config and hasattr(self._config, "metadata"):
            metadata = self._config.metadata
            if isinstance(metadata, dict):
                group_sections_by_step = metadata.get("group_sections_by_step", True)

        # Return updated state
        return ProgressState(
            sections=self._progress_state.sections,
            revealed_counts=revealed_counts,
            is_animating=not is_complete,
            current_step_id=self._progress_state.current_step_id,
            start_time_ms=start_time_ms,
            cumulative_delays_ms=cumulative_delays_ms,
            total_duration_ms=self._progress_state.total_duration_ms,
            is_complete=is_complete,
            # Phase-based fields
            phase=phase,
            pre_messages=self._progress_state.pre_messages,
            post_messages=self._progress_state.post_messages,
            pre_delay_ms=pre_delay_ms,
            post_delay_ms=post_delay_ms,
            pre_message_delays_ms=pre_message_delays,
            post_message_delays_ms=post_message_delays,
            current_pre_message_index=current_pre_message_index,
            current_post_message_count=current_post_message_count,
            phase_start_time_ms=self._progress_state.phase_start_time_ms,
            keep_progress_sections=keep_progress_sections,
            group_sections_by_step=group_sections_by_step,
            # Progress header fields
            progress_header=self._progress_state.progress_header,
            waiting_for_progress_header=self._progress_state.waiting_for_progress_header,
        )

    def complete_progress_animation(self) -> None:
        """Mark progress animation as complete and reset state.

        NOTE: This resets the progress state. Only call this AFTER the frontend
        has processed the 'complete' phase (e.g., saved sections, displayed post_messages).
        """
        self._progress_state = ProgressState(
            sections=[],
            revealed_counts={},
            is_animating=False,
            current_step_id="",
            start_time_ms=0.0,
            cumulative_delays_ms={},
            total_duration_ms=0.0,
            is_complete=True,
        )
        self._pending_step_messages = True
        # Clear auto-advance flag when completing
        self._auto_advance_pending = False
        logger.info("Progress animation completed")

    def set_auto_advance_pending(self, pending: bool) -> None:
        """Set flag to auto-advance on next poll after frontend processes completion."""
        self._auto_advance_pending = pending
        logger.info(f"Auto-advance pending set to: {pending}")

    def is_auto_advance_pending(self) -> bool:
        """Check if auto-advance is pending."""
        return getattr(self, "_auto_advance_pending", False)

    def get_current_step_messages(self) -> dict[str, Any]:
        """Get current step's messages and suggested actions without advancing.

        This is called when progress animation completes to get the step's
        content to display. The step is NOT advanced - that happens when
        the user clicks a suggested action.

        For steps with progress_sections:
        - Returns post_messages (messages to show AFTER animation completes)
        - post_messages REPLACE pre_messages in the chat

        For steps without progress_sections:
        - Returns regular messages

        Returns:
            Dict with 'messages', 'suggested_actions', 'wait_for_user'.
            'suggested_actions' has format: {'message': str|None, 'actions': [...]}.
        """
        step = self.get_current_step()
        if not step:
            return {
                "messages": [],
                "suggested_actions": {"message": None, "actions": []},
                "wait_for_user": True,
            }

        messages = []

        # If step has progress_sections, use post_messages (if available)
        # Otherwise fall back to regular messages
        messages_to_process = []
        if step.progress_sections and step.post_messages:
            messages_to_process = step.post_messages
        else:
            messages_to_process = step.messages

        for msg in messages_to_process:
            if isinstance(msg, ParallelGroup):
                for item in msg.items:
                    messages.append(self._message_config_to_dict(item, step.role))
            elif isinstance(msg, MessageConfig):
                messages.append(self._message_config_to_dict(msg, step.role))

        # Build suggested_actions with message and actions
        suggested_actions_list = []
        if step.suggested_actions:
            suggested_actions_list = [
                {
                    "label": action.label,
                    "action_type": action.action_type,
                    "style": action.style,
                    "prefix": action.prefix,
                    "user_message": action.user_message,
                    "target_step": action.target_step,
                    "input_config": action.input_config,
                }
                for action in step.suggested_actions
            ]

        # Return suggested_actions in new format: {message, actions}
        suggested_actions = {
            "message": step.suggested_actions_message,
            "actions": suggested_actions_list,
        }

        return {
            "messages": messages,
            "suggested_actions": suggested_actions,
            "wait_for_user": step.wait_for_user,
        }

    def get_pre_messages(self) -> list[dict[str, Any]]:
        """Get pre_messages for the current step (shown before progress animation).

        Returns:
            List of message dicts to display before animation starts.
        """
        step = self.get_current_step()
        if not step or not step.pre_messages:
            return []

        messages = []
        for msg in step.pre_messages:
            if isinstance(msg, ParallelGroup):
                for item in msg.items:
                    messages.append(self._message_config_to_dict(item, step.role))
            elif isinstance(msg, MessageConfig):
                messages.append(self._message_config_to_dict(msg, step.role))

        return messages

    def _message_config_to_dict(self, msg: MessageConfig, role: str) -> dict[str, Any]:
        """Convert MessageConfig to dict for JSON serialization."""
        result = {
            "role": role,
            "content": msg.content,
            "file_path": msg.file if msg.file else None,
            "message_type": msg.type,
        }

        # Include input_field if present
        if msg.input_field:
            result["input_field"] = {
                "variable_name": msg.input_field.variable_name,
                "placeholder": msg.input_field.placeholder,
                "multiline": msg.input_field.multiline,
                "optional": msg.input_field.optional,
                "default_value": msg.input_field.default_value,
                "collapsible": msg.input_field.collapsible,
                "initially_collapsed": msg.input_field.initially_collapsed,
                "mode_selector": msg.input_field.mode_selector,
            }

        # Include editable_list if present (pass through as-is for frontend)
        if msg.editable_list:
            print(
                f"[DEBUG BACKEND] _message_config_to_dict: editable_list FOUND for message"
            )
            print(f"[DEBUG BACKEND] editable_list content: {msg.editable_list}")
            result["editable_list"] = msg.editable_list
        else:
            print(
                f"[DEBUG BACKEND] _message_config_to_dict: NO editable_list for message (content starts with: {msg.content[:50] if msg.content else 'N/A'}...)"
            )

        return result

    async def _animate_progress_sections(self, sections: list[ProgressSection]) -> None:
        """Start time-based progress animation (NON-BLOCKING).

        This just starts the animation and returns IMMEDIATELY.
        The WebSocket will poll get_progress_state() to get current visibility.

        IMPORTANT: Unlike the old blocking implementation, this does NOT wait
        for the animation to complete. The WebSocket route handles:
        1. Polling get_progress_state() every 200ms
        2. Detecting is_complete and calling complete_progress_animation()
        3. Sending step messages when animation finishes
        """
        self.start_progress_animation(sections)
        # Return immediately - let WebSocket handle the animation updates
        # The step messages will be added when WebSocket detects completion

    def read_file(self, relative_path: str) -> str:
        """Read a file from the experiment flow directory."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        return self._engine.read_file(relative_path)

    def get_file_path(self, relative_path: str) -> str:
        """Get the full path to a file in the experiment flow directory."""
        return os.path.join(self._base_path, relative_path)

    # Navigation methods for iteration/step jumping

    def get_iterations(self) -> list[dict[str, Any]]:
        """Get all available iterations with their metadata.

        Returns:
            List of iteration dicts with id, name, description, start/end indices, and steps.
        """
        if not self._engine:
            return []

        iterations = self._engine.get_iterations()
        return [
            {
                "id": it.id,
                "name": it.name,
                "description": it.description,
                "start_step_index": it.start_step_index,
                "end_step_index": it.end_step_index,
                "steps": [{"index": s.index, "name": s.name} for s in it.steps],
            }
            for it in iterations
        ]

    def get_flow_structure(self) -> dict[str, Any]:
        """Get complete flow structure for navigation UI.

        Returns:
            Dict with flow name, description, total steps, and iterations.
        """
        if not self._config or not self._engine:
            return {
                "name": "",
                "description": "",
                "total_steps": 0,
                "iterations": [],
            }

        return {
            "name": self._config.name,
            "description": self._config.description,
            "total_steps": len(self._config.steps),
            "iterations": self.get_iterations(),
        }

    def jump_to_step(self, step_index: int) -> ExperimentState:
        """Jump to a specific step by index.

        Args:
            step_index: The 0-based index of the step to jump to.

        Returns:
            Updated experiment state after the jump.

        Raises:
            ValueError: If step_index is out of range.
        """
        if not self._engine:
            raise RuntimeError("Engine not initialized")

        if not self._engine.jump_to_step(step_index):
            raise ValueError(
                f"Step index {step_index} is out of range "
                f"(0-{len(self._config.steps) - 1 if self._config else 0})"
            )

        # Reset progress state when jumping
        self._progress_state = ProgressState()
        self._messages = []

        logger.info(f"Jumped to step index {step_index}")
        return self.get_state()

    def jump_to_step_by_id(self, step_id: str) -> ExperimentState:
        """Jump to a specific step by ID.

        Args:
            step_id: The ID of the step (e.g., "step_5").

        Returns:
            Updated experiment state after the jump.

        Raises:
            ValueError: If step_id is not found.
        """
        if not self._engine:
            raise RuntimeError("Engine not initialized")

        if not self._engine.jump_to_step_by_id(step_id):
            raise ValueError(f"Step '{step_id}' not found")

        # Reset progress state when jumping
        self._progress_state = ProgressState()
        self._messages = []

        logger.info(f"Jumped to step '{step_id}'")
        return self.get_state()

    def jump_to_iteration(
        self, iteration_id: str, step_offset: int = 0
    ) -> ExperimentState:
        """Jump to a specific step within an iteration.

        Args:
            iteration_id: The iteration ID (e.g., "iteration_1" or "1").
            step_offset: The step offset within the iteration (0-based, default 0).

        Returns:
            Updated experiment state after the jump.

        Raises:
            ValueError: If iteration is not found.
        """
        if not self._engine:
            raise RuntimeError("Engine not initialized")

        if not self._engine.jump_to_iteration(iteration_id, step_offset):
            raise ValueError(f"Iteration '{iteration_id}' not found")

        # Reset progress state when jumping
        self._progress_state = ProgressState()
        self._messages = []

        logger.info(
            f"Jumped to iteration '{iteration_id}' with step offset {step_offset}"
        )
        return self.get_state()

    def get_current_iteration(self) -> dict[str, Any] | None:
        """Get the iteration containing the current step.

        Returns:
            Iteration dict or None if not in any iteration.
        """
        if not self._engine:
            return None

        iteration = self._engine.get_current_iteration()
        if not iteration:
            return None

        return {
            "id": iteration.id,
            "name": iteration.name,
            "description": iteration.description,
            "start_step_index": iteration.start_step_index,
            "end_step_index": iteration.end_step_index,
            "steps": [{"index": s.index, "name": s.name} for s in iteration.steps],
        }
