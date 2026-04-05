# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
Chat API Routes for the chatbot backend.

Endpoints:
- POST /api/chat/send - Send user message and get response
- GET /api/chat/messages - Get all messages for current session
- POST /api/chat/action - Handle suggested action clicks
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class SendMessageRequest(BaseModel):
    """Request body for sending a message."""

    message: str


class ActionRequest(BaseModel):
    """Request body for handling an action."""

    index: int


class BranchActionRequest(BaseModel):
    """Request body for handling a branch action with user input."""

    index: int
    input_value: str
    target_step: str | None = None


class InputFieldResponse(BaseModel):
    """Response model for an input field configuration."""

    variable_name: str
    placeholder: str = "Enter your input..."
    multiline: bool = False
    optional: bool = True
    default_value: str = ""
    collapsible: bool = False
    initially_collapsed: bool = False
    mode_selector: dict[str, Any] | None = None


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""

    role: str
    content: str
    file_path: str | None = None
    message_type: str = "text"
    input_field: InputFieldResponse | None = None
    editable_list: dict[str, Any] | None = None


class ProgressSectionResponse(BaseModel):
    """Response model for a progress section."""

    slot: str
    title: str
    messages: list[dict[str, Any]]
    collapsible: bool = True
    initial_state: str = "expanded"
    appearance_delay_ms: float = 0.0  # Delay before this section appears
    # Fields for task_progress support
    type: str | None = None  # "task_progress" for task progress panels
    tasks: list[dict[str, Any]] | None = None  # Task data
    prompt_file: str | None = None


class ExperimentStateResponse(BaseModel):
    """Response model for experiment state."""

    current_step_index: int
    current_step_id: str
    is_complete: bool
    messages: list[ChatMessageResponse]
    suggested_actions: dict[
        str, Any
    ]  # Format: {message: str | None, actions: list[dict]}
    is_waiting_for_user: bool
    is_animating: bool = False  # True if progress animation is in progress
    # Progress animation state (only present when is_animating=True)
    progress_sections: list[ProgressSectionResponse] | None = None
    revealed_counts: dict[str, int] | None = None


def _get_experiment_service(request: Request) -> Any:
    """Get experiment service from app state."""
    return request.app.state.get_experiment_service()


@router.post("/send", response_model=ExperimentStateResponse)
async def send_message(
    request: Request, body: SendMessageRequest
) -> ExperimentStateResponse:
    """Send a user message and process the experiment step.

    This triggers the experiment to process the user input and
    advance to the next step (if applicable).

    IMPORTANT: This endpoint returns immediately after starting the step.
    If the step has progress animation, the animation runs in background
    and updates are pushed via WebSocket. The frontend should:
    1. Show the user message immediately (optimistic UI)
    2. Show pre_messages from response
    3. Poll /api/chat/progress for progress updates
    4. Replace pre_messages with post_messages when animation completes
    """
    try:
        service = _get_experiment_service(request)
        state = await service.process_user_input(body.message)

        # Check if animation is currently running
        progress_state = service.get_progress_state()
        is_animating = progress_state.is_animating

        # Build progress sections for response if animating
        # NOTE: We return sections info but frontend should NOT display them
        # until the polling endpoint says we're in 'progress' phase
        progress_sections = None
        revealed_counts = None
        if is_animating and progress_state.sections:
            progress_sections = [
                ProgressSectionResponse(
                    slot=section["slot"],
                    title=section["title"],
                    messages=section.get("messages", []),
                    collapsible=section.get("collapsible", True),
                    initial_state=section.get("initial_state", "expanded"),
                    appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                    type=section.get("type"),
                    tasks=section.get("tasks"),
                    prompt_file=section.get("prompt_file"),
                )
                for section in progress_state.sections
            ]
            revealed_counts = progress_state.revealed_counts

        # DON'T add pre_messages to the response - let the polling handle them
        # based on the current animation phase (pre_delay, pre_messages, etc.)

        # Build messages list - only include messages already processed (not animation messages)
        messages = [
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                file_path=msg.file_path,
                message_type=msg.message_type,
            )
            for msg in state.messages
        ]

        return ExperimentStateResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            messages=messages,
            suggested_actions=state.suggested_actions,
            is_waiting_for_user=state.is_waiting_for_user,
            is_animating=is_animating,
            progress_sections=progress_sections,
            revealed_counts=revealed_counts,
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages", response_model=list[ChatMessageResponse])
async def get_messages(request: Request) -> list[ChatMessageResponse]:
    """Get all messages in the current session."""
    try:
        service = _get_experiment_service(request)
        state = service.get_state()

        return [
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                file_path=msg.file_path,
                message_type=msg.message_type,
            )
            for msg in state.messages
        ]
    except Exception as e:
        logger.error(f"Error getting messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/action", response_model=ExperimentStateResponse)
async def handle_action(
    request: Request, body: ActionRequest
) -> ExperimentStateResponse:
    """Handle a suggested action button click.

    This processes the action (e.g., "continue", "input_prefix")
    and advances the experiment accordingly.

    Like /send, if the next step has progress animation, is_animating=True
    and the frontend should start polling /progress for updates.
    """
    try:
        service = _get_experiment_service(request)
        state = await service.handle_action(body.index)

        # Check if animation is currently running (same as /send endpoint)
        progress_state = service.get_progress_state()
        is_animating = progress_state.is_animating

        # Build progress sections for response if animating
        progress_sections = None
        revealed_counts = None
        if is_animating and progress_state.sections:
            progress_sections = [
                ProgressSectionResponse(
                    slot=section["slot"],
                    title=section["title"],
                    messages=section.get("messages", []),
                    collapsible=section.get("collapsible", True),
                    initial_state=section.get("initial_state", "expanded"),
                    appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                    type=section.get("type"),
                    tasks=section.get("tasks"),
                    prompt_file=section.get("prompt_file"),
                )
                for section in progress_state.sections
            ]
            revealed_counts = progress_state.revealed_counts

        # Build messages list
        messages = [
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                file_path=msg.file_path,
                message_type=msg.message_type,
            )
            for msg in state.messages
        ]

        return ExperimentStateResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            messages=messages,
            suggested_actions=state.suggested_actions,
            is_waiting_for_user=state.is_waiting_for_user,
            is_animating=is_animating,
            progress_sections=progress_sections,
            revealed_counts=revealed_counts,
        )
    except Exception as e:
        logger.error(f"Error handling action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/branch_action", response_model=ExperimentStateResponse)
async def handle_branch_action(
    request: Request, body: BranchActionRequest
) -> ExperimentStateResponse:
    """Handle a branch action with user input.

    This processes a branch action that includes user input (e.g., "Deep dive on a topic")
    and navigates to a specific branch step.

    Like /action, if the target step has progress animation, is_animating=True
    and the frontend should start polling /progress for updates.
    """
    try:
        service = _get_experiment_service(request)
        state = await service.handle_branch_action(
            body.index, body.input_value, body.target_step
        )

        # Check if animation is currently running (same as /action endpoint)
        progress_state = service.get_progress_state()
        is_animating = progress_state.is_animating

        # Build progress sections for response if animating
        progress_sections = None
        revealed_counts = None
        if is_animating and progress_state.sections:
            progress_sections = [
                ProgressSectionResponse(
                    slot=section["slot"],
                    title=section["title"],
                    messages=section.get("messages", []),
                    collapsible=section.get("collapsible", True),
                    initial_state=section.get("initial_state", "expanded"),
                    appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                    type=section.get("type"),
                    tasks=section.get("tasks"),
                    prompt_file=section.get("prompt_file"),
                )
                for section in progress_state.sections
            ]
            revealed_counts = progress_state.revealed_counts

        # Build messages list
        messages = [
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                file_path=msg.file_path,
                message_type=msg.message_type,
            )
            for msg in state.messages
        ]

        return ExperimentStateResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            messages=messages,
            suggested_actions=state.suggested_actions,
            is_waiting_for_user=state.is_waiting_for_user,
            is_animating=is_animating,
            progress_sections=progress_sections,
            revealed_counts=revealed_counts,
        )
    except Exception as e:
        logger.error(f"Error handling branch action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CompletedStepData(BaseModel):
    """Data for a completed step that was auto-advanced.

    When a step completes with wait_for_user: false, we capture its completion
    data here so the frontend can process it before showing the next animation.
    """

    step_id: str
    post_messages: list[ChatMessageResponse]
    sections: list[ProgressSectionResponse]
    revealed_counts: dict[str, int]
    keep_progress_sections: bool = False


class ContinueButtonResponse(BaseModel):
    """Response model for a continue button in progress header."""

    label: str = "🚀 Continue"
    style: str = "primary"


class ProgressHeaderResponse(BaseModel):
    """Response model for progress header (shown before progress sections)."""

    content: str
    input_field: InputFieldResponse | None = None
    continue_button: ContinueButtonResponse = ContinueButtonResponse()


class ProgressStateResponse(BaseModel):
    """Response model for progress animation state (used for polling).

    Animation phases:
    1. pre_delay: Waiting before pre_messages
    2. pre_messages: Showing pre_messages sequentially (each replaces previous)
    3. progress_header: Showing progress header with input field, waiting for user to click continue
    4. progress: Showing progress sections
    5. post_delay: Waiting after progress completes
    6. post_messages: Showing post_messages sequentially (each appends)
    7. complete: Animation finished

    Auto-advance behavior:
    When a step completes with wait_for_user: false, the backend executes the
    next step immediately. The completed step's data is stored in completed_steps
    so the frontend can process it before showing the next animation.
    """

    sections: list[ProgressSectionResponse]
    revealed_counts: dict[str, int]
    is_animating: bool
    is_complete: bool
    current_step_id: str
    # Phase-based animation fields
    phase: str  # idle, pre_delay, pre_messages, progress_header, progress, post_delay, post_messages, complete
    pre_messages: list[ChatMessageResponse] | None = None
    post_messages: list[ChatMessageResponse] | None = None
    current_pre_message_index: int = (
        -1
    )  # Which pre_message is currently showing (-1 = none)
    current_post_message_count: int = 0  # How many post_messages are revealed
    # Only present when animation completes
    step_messages: list[ChatMessageResponse] | None = None
    suggested_actions: dict[str, Any] | None = (
        None  # Format: {message: str | None, actions: list[dict]}
    )
    # Option to keep progress sections visible after animation completes
    keep_progress_sections: bool = False
    # Step ID for identifying which step these sections belong to
    step_id: str = ""
    # UI config from flow.json metadata
    group_sections_by_step: bool = True  # Default: group sections by step
    # Completed steps from auto-advance (frontend processes these FIRST)
    completed_steps: list[CompletedStepData] | None = None
    # Time-based animation: start time in ms for frontend to calculate elapsed time
    start_time_ms: float = 0.0
    # Progress header fields
    progress_header: ProgressHeaderResponse | None = None
    waiting_for_progress_header: bool = False


@router.get("/progress", response_model=ProgressStateResponse)
async def get_progress(request: Request) -> ProgressStateResponse:
    """Get current progress animation state (for HTTP polling).

    This endpoint allows the frontend to poll for progress updates
    instead of using WebSocket. Poll every 200ms during animation.

    Animation phases:
    1. pre_delay: Waiting before pre_messages
    2. pre_messages: Showing pre_messages sequentially (each replaces previous)
    3. progress: Showing progress sections
    4. post_delay: Waiting after progress completes
    5. post_messages: Showing post_messages sequentially (each appends)
    6. complete: Animation finished

    Auto-advance behavior:
    When a step completes with wait_for_user: false, we:
    1. Capture the completed step's data (sections, post_messages)
    2. Advance to the next step and start its animation
    3. Return BOTH the completed step data AND the new animation state
    This allows the frontend to process completions in order, then show the next animation.

    Returns the current progress state including:
    - phase: Current animation phase
    - sections: List of progress sections with their messages
    - revealed_counts: How many messages are revealed per section
    - pre_messages: List of pre_messages (for phases before post_messages)
    - post_messages: List of post_messages (for post_messages phase)
    - current_pre_message_index: Which pre_message to show (-1 = none)
    - current_post_message_count: How many post_messages to show
    - step_messages: Messages to display after completion (only when complete)
    - suggested_actions: Actions to show after completion (only when complete)
    - completed_steps: List of completed step data (for auto-advance chains)
    """
    try:
        service = _get_experiment_service(request)

        # Collect completion data from auto-advancing steps
        completed_steps_data: list[CompletedStepData] = []
        # Track if we exited because of wait_for_user: true (need to show suggested_actions)
        stopped_for_user_wait = False

        # Loop: process steps until we hit one that's animating or has wait_for_user: true
        while True:
            progress_state = service.get_progress_state()

            if progress_state.is_complete:
                # Step completed - capture its data
                # FIX 1: Guard against capturing when no valid step exists
                current_step_id = progress_state.current_step_id
                if not current_step_id:
                    logger.info("No current step to capture, breaking loop")
                    break

                step_data = service.get_current_step_messages()
                wait_for_user = step_data.get("wait_for_user", True)

                # Build sections for this completed step
                completed_sections = [
                    ProgressSectionResponse(
                        slot=section["slot"],
                        title=section["title"],
                        messages=section.get("messages", []),
                        collapsible=section.get("collapsible", True),
                        initial_state=section.get("initial_state", "expanded"),
                        appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                        type=section.get("type"),
                        tasks=section.get("tasks"),
                        prompt_file=section.get("prompt_file"),
                    )
                    for section in progress_state.sections
                ]

                # FIX 2: Capture post_messages from progress_state BEFORE reset
                # This is critical for post_messages_only steps where animated
                # post_messages are in progress_state, not in step_data["messages"]
                post_messages_from_animation = progress_state.post_messages or []

                # Build post_messages - prefer animated post_messages, fallback to step_data
                if post_messages_from_animation:
                    # Use post_messages from the animation (for post_messages_only steps)
                    completed_post_messages = [
                        ChatMessageResponse(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content", ""),
                            file_path=msg.get("file_path"),
                            message_type=msg.get("message_type", "text"),
                            input_field=InputFieldResponse(**msg["input_field"])
                            if msg.get("input_field")
                            else None,
                            editable_list=msg.get("editable_list"),
                        )
                        for msg in post_messages_from_animation
                    ]
                else:
                    # Fallback to step_data messages (for regular steps)
                    completed_post_messages = [
                        ChatMessageResponse(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content", ""),
                            file_path=msg.get("file_path"),
                            message_type=msg.get("message_type", "text"),
                            input_field=InputFieldResponse(**msg["input_field"])
                            if msg.get("input_field")
                            else None,
                            editable_list=msg.get("editable_list"),
                        )
                        for msg in step_data.get("messages", [])
                    ]

                # Capture completion data
                completion_data = CompletedStepData(
                    step_id=current_step_id,
                    post_messages=completed_post_messages,
                    sections=completed_sections,
                    revealed_counts=progress_state.revealed_counts.copy(),
                    keep_progress_sections=progress_state.keep_progress_sections,
                )
                completed_steps_data.append(completion_data)

                logger.info(
                    f"Step '{progress_state.current_step_id}' completed, "
                    f"wait_for_user={wait_for_user}, "
                    f"sections={len(completed_sections)}, "
                    f"post_messages={len(completed_post_messages)}"
                )

                # Complete this step's animation (resets progress state)
                service.complete_progress_animation()

                if not wait_for_user and service._engine:
                    # Auto-advance to next step
                    logger.info(
                        f"Auto-advancing from step '{completion_data.step_id}' (wait_for_user=false)"
                    )
                    service._engine.advance_step()
                    await service._process_current_step()

                    # Check if next step is animating
                    new_state = service.get_progress_state()
                    if new_state.is_animating:
                        # Next step is animating - exit loop and return
                        logger.info(
                            f"Next step '{new_state.current_step_id}' is animating"
                        )
                        break
                    # Else: next step completed instantly (e.g., no delays)
                    # Continue loop to capture its completion data
                    logger.info(f"Next step completed instantly, continuing loop")
                else:
                    # wait_for_user: true OR no more steps - return completion
                    logger.info(
                        f"Step requires user action or flow complete, returning"
                    )
                    stopped_for_user_wait = wait_for_user
                    break
            else:
                # Still animating - exit loop
                break

        # Get final progress state after all auto-advances
        progress_state = service.get_progress_state()

        # Build sections for response
        sections = [
            ProgressSectionResponse(
                slot=section["slot"],
                title=section["title"],
                messages=section.get("messages", []),
                collapsible=section.get("collapsible", True),
                initial_state=section.get("initial_state", "expanded"),
                appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                type=section.get("type"),
                tasks=section.get("tasks"),
                prompt_file=section.get("prompt_file"),
            )
            for section in progress_state.sections
        ]

        # Build pre_messages and post_messages as ChatMessageResponse
        pre_messages_response = (
            [
                ChatMessageResponse(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    file_path=msg.get("file_path"),
                    message_type=msg.get("message_type", "text"),
                    input_field=InputFieldResponse(**msg["input_field"])
                    if msg.get("input_field")
                    else None,
                    editable_list=msg.get("editable_list"),
                )
                for msg in progress_state.pre_messages
            ]
            if progress_state.pre_messages
            else None
        )

        post_messages_response = (
            [
                ChatMessageResponse(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    file_path=msg.get("file_path"),
                    message_type=msg.get("message_type", "text"),
                    input_field=InputFieldResponse(**msg["input_field"])
                    if msg.get("input_field")
                    else None,
                    editable_list=msg.get("editable_list"),
                )
                for msg in progress_state.post_messages
            ]
            if progress_state.post_messages
            else None
        )

        # Build progress_header response if present
        progress_header_response = None
        if progress_state.progress_header:
            header_data = progress_state.progress_header
            input_field_response = None
            if header_data.get("input_field"):
                input_field_response = InputFieldResponse(**header_data["input_field"])
            continue_button_response = ContinueButtonResponse(
                **header_data.get("continue_button", {})
            )
            progress_header_response = ProgressHeaderResponse(
                content=header_data.get("content", ""),
                input_field=input_field_response,
                continue_button=continue_button_response,
            )

        response = ProgressStateResponse(
            sections=sections,
            revealed_counts=progress_state.revealed_counts,
            is_animating=progress_state.is_animating,
            is_complete=progress_state.is_complete,
            current_step_id=progress_state.current_step_id,
            # Phase-based fields
            phase=progress_state.phase,
            pre_messages=pre_messages_response,
            post_messages=post_messages_response,
            current_pre_message_index=progress_state.current_pre_message_index,
            current_post_message_count=progress_state.current_post_message_count,
            keep_progress_sections=progress_state.keep_progress_sections,
            step_id=progress_state.current_step_id,
            group_sections_by_step=progress_state.group_sections_by_step,
            # Include completed steps from auto-advance chain
            completed_steps=completed_steps_data if completed_steps_data else None,
            # Time-based animation: send start_time_ms for frontend to calculate elapsed time
            start_time_ms=progress_state.start_time_ms,
            # Progress header fields
            progress_header=progress_header_response,
            waiting_for_progress_header=progress_state.waiting_for_progress_header,
        )

        # If current state is also complete (final step), include its data too
        if progress_state.is_complete:
            step_data = service.get_current_step_messages()
            response.step_messages = [
                ChatMessageResponse(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    file_path=msg.get("file_path"),
                    message_type=msg.get("message_type", "text"),
                    input_field=InputFieldResponse(**msg["input_field"])
                    if msg.get("input_field")
                    else None,
                    editable_list=msg.get("editable_list"),
                )
                for msg in step_data.get("messages", [])
            ]
            response.suggested_actions = step_data.get(
                "suggested_actions", {"message": None, "actions": []}
            )

        # IMPORTANT: Also include suggested_actions when we stopped for wait_for_user
        # This happens when a step completes with wait_for_user: true
        if stopped_for_user_wait and completed_steps_data:
            # Get suggested_actions from the step that required user wait
            step_data = service.get_current_step_messages()
            response.suggested_actions = step_data.get(
                "suggested_actions", {"message": None, "actions": []}
            )
            logger.info(
                f"Including suggested_actions for wait_for_user step: "
                f"{len(response.suggested_actions.get('actions', []))} actions"
            )

        return response
    except Exception as e:
        logger.error(f"Error getting progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ContinueFromProgressHeaderRequest(BaseModel):
    """Request body for continuing from progress_header phase."""

    user_input: dict[str, str] | None = None


@router.post("/progress/continue", response_model=ProgressStateResponse)
async def continue_from_progress_header(
    request: Request, body: ContinueFromProgressHeaderRequest
) -> ProgressStateResponse:
    """Continue animation from progress_header phase after user clicks continue button.

    This endpoint is called when the user clicks the continue button in the
    progress_header. It stores any user input and transitions the animation
    from the progress_header phase to the progress phase.

    Request body:
        user_input: Optional dict of user input values (variable_name -> value)

    Returns:
        The updated progress state after transitioning to progress phase.
    """
    try:
        service = _get_experiment_service(request)

        # Call the service method to continue from progress_header
        service.continue_from_progress_header(body.user_input)

        # Get the updated progress state
        progress_state = service.get_progress_state()

        # Build sections for response
        sections = [
            ProgressSectionResponse(
                slot=section["slot"],
                title=section["title"],
                messages=section.get("messages", []),
                collapsible=section.get("collapsible", True),
                initial_state=section.get("initial_state", "expanded"),
                appearance_delay_ms=section.get("appearance_delay_ms", 0.0),
                type=section.get("type"),
                tasks=section.get("tasks"),
                prompt_file=section.get("prompt_file"),
            )
            for section in progress_state.sections
        ]

        # Build pre_messages and post_messages as ChatMessageResponse
        pre_messages_response = (
            [
                ChatMessageResponse(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    file_path=msg.get("file_path"),
                    message_type=msg.get("message_type", "text"),
                    input_field=InputFieldResponse(**msg["input_field"])
                    if msg.get("input_field")
                    else None,
                    editable_list=msg.get("editable_list"),
                )
                for msg in progress_state.pre_messages
            ]
            if progress_state.pre_messages
            else None
        )

        post_messages_response = (
            [
                ChatMessageResponse(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    file_path=msg.get("file_path"),
                    message_type=msg.get("message_type", "text"),
                    input_field=InputFieldResponse(**msg["input_field"])
                    if msg.get("input_field")
                    else None,
                    editable_list=msg.get("editable_list"),
                )
                for msg in progress_state.post_messages
            ]
            if progress_state.post_messages
            else None
        )

        # Build progress_header response if present
        progress_header_response = None
        if progress_state.progress_header:
            header_data = progress_state.progress_header
            input_field_response = None
            if header_data.get("input_field"):
                input_field_response = InputFieldResponse(**header_data["input_field"])
            continue_button_response = ContinueButtonResponse(
                **header_data.get("continue_button", {})
            )
            progress_header_response = ProgressHeaderResponse(
                content=header_data.get("content", ""),
                input_field=input_field_response,
                continue_button=continue_button_response,
            )

        return ProgressStateResponse(
            sections=sections,
            revealed_counts=progress_state.revealed_counts,
            is_animating=progress_state.is_animating,
            is_complete=progress_state.is_complete,
            current_step_id=progress_state.current_step_id,
            phase=progress_state.phase,
            pre_messages=pre_messages_response,
            post_messages=post_messages_response,
            current_pre_message_index=progress_state.current_pre_message_index,
            current_post_message_count=progress_state.current_post_message_count,
            keep_progress_sections=progress_state.keep_progress_sections,
            step_id=progress_state.current_step_id,
            group_sections_by_step=progress_state.group_sections_by_step,
            start_time_ms=progress_state.start_time_ms,
            progress_header=progress_header_response,
            waiting_for_progress_header=progress_state.waiting_for_progress_header,
        )
    except Exception as e:
        logger.error(f"Error continuing from progress_header: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", response_model=ExperimentStateResponse)
async def reset_experiment(request: Request) -> ExperimentStateResponse:
    """Reset the experiment to the beginning."""
    try:
        service = _get_experiment_service(request)
        service.reset()
        state = service.get_state()

        return ExperimentStateResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            messages=[
                ChatMessageResponse(
                    role=msg.role,
                    content=msg.content,
                    file_path=msg.file_path,
                    message_type=msg.message_type,
                )
                for msg in state.messages
            ],
            suggested_actions=state.suggested_actions,
            is_waiting_for_user=state.is_waiting_for_user,
        )
    except Exception as e:
        logger.error(f"Error resetting experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
