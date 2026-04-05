# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
Experiment API Routes for the chatbot backend.

Endpoints:
- GET /api/experiment/status - Get current experiment state
- GET /api/experiment/progress - Get current progress animation state (polling fallback)
- POST /api/experiment/complete-step - Complete animation and get step messages
- POST /api/experiment/reset - Reset experiment to start
- GET /api/experiment/files/{path} - Get file content for viewer
- GET /api/experiment/flow-structure - Get complete flow structure with iterations
- POST /api/experiment/jump - Jump to a specific iteration/step
"""

import logging
from typing import Any, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ProgressSectionResponse(BaseModel):
    """Response model for a progress section."""

    slot: str
    title: str
    messages: list[dict[str, Any]]
    collapsible: bool
    initial_state: str
    # Fields for task_progress support
    type: str | None = None  # "task_progress" for task progress panels
    tasks: list[dict[str, Any]] | None = None  # Task data
    appearance_delay_ms: float = 0.0  # Appearance delay
    prompt_file: str | None = None


class ProgressStateResponse(BaseModel):
    """Response model for progress animation state."""

    sections: list[ProgressSectionResponse]
    revealed_counts: dict[str, int]
    is_animating: bool
    is_complete: bool
    current_step_id: str
    # Fields for phase-based animation
    phase: str = "idle"  # Current animation phase
    pre_messages: list[dict[str, Any]] = []  # Pre-messages
    post_messages: list[dict[str, Any]] = []  # Post-messages
    current_pre_message_index: int = -1  # Which pre_message is showing
    progress_header: dict[str, Any] | None = None  # Progress header config
    keep_progress_sections: bool = False  # Keep sections after completion


class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status."""

    current_step_index: int
    current_step_id: str
    is_complete: bool
    is_waiting_for_user: bool
    progress_state: ProgressStateResponse
    suggested_actions: list[dict[str, Any]]


class StepMessagesResponse(BaseModel):
    """Response model for step messages after animation completes."""

    messages: list[dict[str, Any]]
    suggested_actions: list[dict[str, Any]]
    wait_for_user: bool
    file_references: list[dict[str, Any]]


def _get_experiment_service(request: Request) -> Any:
    """Get experiment service from app state."""
    return request.app.state.get_experiment_service()


@router.get("/status", response_model=ExperimentStatusResponse)
async def get_status(request: Request) -> ExperimentStatusResponse:
    """Get the current experiment status and progress state.

    This is useful for the frontend to check the current state
    without triggering any actions.
    """
    try:
        service = _get_experiment_service(request)
        state = service.get_state()

        progress = state.progress_state
        progress_response = ProgressStateResponse(
            sections=[
                ProgressSectionResponse(
                    slot=s.get("slot", ""),
                    title=s.get("title", ""),
                    messages=s.get("messages", []),
                    collapsible=s.get("collapsible", True),
                    initial_state=s.get("initial_state", "expanded"),
                    type=s.get("type"),
                    tasks=s.get("tasks"),
                    appearance_delay_ms=s.get("appearance_delay_ms", 0.0),
                    prompt_file=s.get("prompt_file"),
                )
                for s in progress.sections
            ],
            revealed_counts=progress.revealed_counts,
            is_animating=progress.is_animating,
            is_complete=progress.is_complete,
            current_step_id=progress.current_step_id,
            phase=progress.phase,
            pre_messages=progress.pre_messages,
            post_messages=progress.post_messages,
            current_pre_message_index=progress.current_pre_message_index,
            progress_header=progress.progress_header,
            keep_progress_sections=progress.keep_progress_sections,
        )

        return ExperimentStatusResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            is_waiting_for_user=state.is_waiting_for_user,
            progress_state=progress_response,
            suggested_actions=state.suggested_actions,
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress", response_model=ProgressStateResponse)
async def get_progress(request: Request) -> ProgressStateResponse:
    """Get the current progress animation state.

    This endpoint provides a polling fallback for the WebSocket progress updates.
    It returns the same time-based calculation as the WebSocket endpoint.

    The frontend can poll this endpoint at 200ms intervals if WebSocket
    connection fails or is not available.
    """
    try:
        service = _get_experiment_service(request)
        # Time-based calculation: get current progress state
        progress = service.get_progress_state()

        return ProgressStateResponse(
            sections=[
                ProgressSectionResponse(
                    slot=s.get("slot", ""),
                    title=s.get("title", ""),
                    messages=s.get("messages", []),
                    collapsible=s.get("collapsible", True),
                    initial_state=s.get("initial_state", "expanded"),
                    type=s.get("type"),
                    tasks=s.get("tasks"),
                    appearance_delay_ms=s.get("appearance_delay_ms", 0.0),
                    prompt_file=s.get("prompt_file"),
                )
                for s in progress.sections
            ],
            revealed_counts=progress.revealed_counts,
            is_animating=progress.is_animating,
            is_complete=progress.is_complete,
            current_step_id=progress.current_step_id,
            phase=progress.phase,
            pre_messages=progress.pre_messages,
            post_messages=progress.post_messages,
            current_pre_message_index=progress.current_pre_message_index,
            progress_header=progress.progress_header,
            keep_progress_sections=progress.keep_progress_sections,
        )
    except Exception as e:
        logger.error(f"Error getting progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete-step", response_model=StepMessagesResponse)
async def complete_step(request: Request) -> StepMessagesResponse:
    """Complete the current animation and get step messages.

    This endpoint is called when the progress animation completes.
    It returns:
    - The step messages to display (with file paths for "📄 View File" buttons)
    - Suggested actions (e.g., "Continue" button)
    - Whether to wait for user input before advancing

    Note: This does NOT auto-advance the step. The step advances when
    the user clicks a suggested action (handled by /api/chat/action).
    """
    try:
        service = _get_experiment_service(request)

        # Mark animation as complete
        service.complete_progress_animation()

        # Get step messages and actions
        step_data = service.get_current_step_messages()

        # Extract file references for "📄 View File" buttons
        file_references = []
        for msg in step_data.get("messages", []):
            if msg.get("file_path"):
                file_references.append(
                    {
                        "file_path": msg["file_path"],
                        "label": f"📄 View: {msg['file_path'].split('/')[-1]}",
                    }
                )

        logger.info(
            f"Step completed with {len(step_data.get('messages', []))} messages, "
            f"{len(file_references)} file references"
        )

        return StepMessagesResponse(
            messages=step_data.get("messages", []),
            suggested_actions=step_data.get("suggested_actions", []),
            wait_for_user=step_data.get("wait_for_user", True),
            file_references=file_references,
        )
    except Exception as e:
        logger.error(f"Error completing step: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_path:path}")
async def get_file(request: Request, file_path: str) -> PlainTextResponse:
    """Get file content for the file viewer.

    The file_path should be URL-encoded and relative to the
    experiment flow directory.
    """
    try:
        service = _get_experiment_service(request)
        # Decode URL-encoded path
        decoded_path = unquote(file_path)
        content = service.read_file(decoded_path)

        return PlainTextResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/static-html/{file_path:path}")
async def get_static_html_file(request: Request, file_path: str) -> Any:
    """Serve static files from HTML documentation directories.

    This endpoint serves files with proper MIME types, allowing HTML files
    to load their CSS, JS, and other assets correctly. It's designed
    specifically for Sphinx-generated HTML documentation that has relative
    paths to static assets.

    The file_path should be URL-encoded and relative to the
    experiment flow directory.

    Example:
        /api/experiment/static-html/files/context/codebase_documentation/_build/html/index.html
        /api/experiment/static-html/files/context/codebase_documentation/_build/html/_static/css/theme.css
    """
    import mimetypes
    import os

    from starlette.responses import Response

    logger.info(f"=== STATIC-HTML ENDPOINT HIT === file_path: {file_path}")

    try:
        service = _get_experiment_service(request)
        # Decode URL-encoded path
        decoded_path = unquote(file_path)
        logger.info(f"Decoded path: {decoded_path}")

        # Get the full file path
        full_path = service.get_file_path(decoded_path)
        logger.info(f"Full path: {full_path}")

        # Security check: ensure the path is within the experiment directory
        base_path = service._base_path
        real_full_path = os.path.realpath(full_path)
        real_base_path = os.path.realpath(base_path)

        if not real_full_path.startswith(real_base_path):
            raise HTTPException(
                status_code=403,
                detail="Access denied: path outside experiment directory",
            )

        # Check if file exists
        if not os.path.isfile(full_path):
            logger.error(f"File not found: {full_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Read the file content
        with open(full_path, "rb") as f:
            content = f.read()

        # Determine content type based on file extension
        if full_path.endswith(".html") or full_path.endswith(".htm"):
            content_type = "text/html; charset=utf-8"
        elif full_path.endswith(".css"):
            content_type = "text/css; charset=utf-8"
        elif full_path.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"
        elif full_path.endswith(".json"):
            content_type = "application/json; charset=utf-8"
        elif full_path.endswith(".png"):
            content_type = "image/png"
        elif full_path.endswith(".jpg") or full_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif full_path.endswith(".svg"):
            content_type = "image/svg+xml"
        elif full_path.endswith(".woff"):
            content_type = "font/woff"
        elif full_path.endswith(".woff2"):
            content_type = "font/woff2"
        elif full_path.endswith(".ttf"):
            content_type = "font/ttf"
        elif full_path.endswith(".eot"):
            content_type = "application/vnd.ms-fontobject"
        else:
            mime_type, _ = mimetypes.guess_type(full_path)
            content_type = mime_type or "application/octet-stream"

        logger.info(f"Serving file: {full_path} with Content-Type: {content_type}")

        # Return with explicit headers to ensure browser displays (not downloads)
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache",
            },
        )
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error serving static file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flow-info")
async def get_flow_info(request: Request) -> dict[str, Any]:
    """Get information about the current experiment flow.

    Returns metadata about the flow such as name, description,
    and total number of steps.
    """
    try:
        service = _get_experiment_service(request)
        if service._config is None:
            raise HTTPException(status_code=500, detail="Flow not initialized")

        config = service._config
        return {
            "name": config.name,
            "description": config.description,
            "total_steps": len(config.steps),
            "code_entry_point": config.code_entry_point,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === New Navigation API Endpoints ===


class JumpRequest(BaseModel):
    """Request model for jumping to a specific position."""

    step_index: Optional[int] = None
    step_id: Optional[str] = None
    iteration_id: Optional[str] = None
    step_offset: int = 0  # Step offset within iteration (0-based)


class IterationStepInfo(BaseModel):
    """Information about a step within an iteration."""

    index: int
    name: str


class IterationInfo(BaseModel):
    """Information about an iteration."""

    id: str
    name: str
    description: str
    start_step_index: int
    end_step_index: int
    steps: list[IterationStepInfo]


class FlowStructureResponse(BaseModel):
    """Response model for flow structure."""

    name: str
    description: str
    total_steps: int
    iterations: list[IterationInfo]


@router.get("/flow-structure", response_model=FlowStructureResponse)
async def get_flow_structure(request: Request) -> FlowStructureResponse:
    """Get complete flow structure with iterations and steps for navigation UI.

    This endpoint provides the full structure of the experiment flow, including
    all iterations and their steps. Useful for building navigation UIs.

    Returns:
        FlowStructureResponse with name, description, total_steps, and iterations.
    """
    try:
        service = _get_experiment_service(request)
        structure = service.get_flow_structure()

        return FlowStructureResponse(
            name=structure["name"],
            description=structure["description"],
            total_steps=structure["total_steps"],
            iterations=[
                IterationInfo(
                    id=it["id"],
                    name=it["name"],
                    description=it["description"],
                    start_step_index=it["start_step_index"],
                    end_step_index=it["end_step_index"],
                    steps=[
                        IterationStepInfo(index=s["index"], name=s["name"])
                        for s in it["steps"]
                    ],
                )
                for it in structure["iterations"]
            ],
        )
    except Exception as e:
        logger.error(f"Error getting flow structure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jump")
async def jump_to_position(
    request: Request, jump_request: JumpRequest
) -> ExperimentStatusResponse:
    """Jump to a specific step or iteration.

    Provide one of:
    - step_index: Jump to step by index (0-based)
    - step_id: Jump to step by ID (e.g., "step_5")
    - iteration_id: Jump to iteration (e.g., "iteration_1" or "1")
      with optional step_offset within that iteration

    Examples:
        {"step_index": 5}  # Jump to step 5
        {"step_id": "step_5"}  # Jump to step with ID "step_5"
        {"iteration_id": "iteration_2"}  # Jump to start of iteration 2
        {"iteration_id": "2", "step_offset": 3}  # Jump to step 3 of iteration 2
    """
    try:
        service = _get_experiment_service(request)

        # Determine which jump to perform
        if jump_request.step_index is not None:
            state = service.jump_to_step(jump_request.step_index)
        elif jump_request.step_id is not None:
            state = service.jump_to_step_by_id(jump_request.step_id)
        elif jump_request.iteration_id is not None:
            state = service.jump_to_iteration(
                jump_request.iteration_id, jump_request.step_offset
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide one of: step_index, step_id, or iteration_id",
            )

        # Build response
        progress = state.progress_state
        progress_response = ProgressStateResponse(
            sections=[
                ProgressSectionResponse(
                    slot=s.get("slot", ""),
                    title=s.get("title", ""),
                    messages=s.get("messages", []),
                    collapsible=s.get("collapsible", True),
                    initial_state=s.get("initial_state", "expanded"),
                    type=s.get("type"),
                    tasks=s.get("tasks"),
                    appearance_delay_ms=s.get("appearance_delay_ms", 0.0),
                    prompt_file=s.get("prompt_file"),
                )
                for s in progress.sections
            ],
            revealed_counts=progress.revealed_counts,
            is_animating=progress.is_animating,
            is_complete=progress.is_complete,
            current_step_id=progress.current_step_id,
            phase=progress.phase,
            pre_messages=progress.pre_messages,
            post_messages=progress.post_messages,
            current_pre_message_index=progress.current_pre_message_index,
            progress_header=progress.progress_header,
            keep_progress_sections=progress.keep_progress_sections,
        )

        return ExperimentStatusResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            is_waiting_for_user=state.is_waiting_for_user,
            progress_state=progress_response,
            suggested_actions=state.suggested_actions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error jumping to position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_experiment(
    request: Request,
    start_step_index: int = Query(default=0, description="Starting step index"),
    iteration_id: Optional[str] = Query(
        default=None, description="Start from iteration (e.g., 'iteration_1' or '1')"
    ),
    step_offset: int = Query(
        default=0, description="Step offset within iteration (0-based)"
    ),
) -> ExperimentStatusResponse:
    """Reset experiment with optional starting position.

    By default resets to step 0. Can specify:
    - start_step_index: Start from a specific step index
    - iteration_id + step_offset: Start from a specific step within an iteration

    Examples:
        POST /api/experiment/reset  # Reset to beginning
        POST /api/experiment/reset?start_step_index=5  # Reset to step 5
        POST /api/experiment/reset?iteration_id=iteration_2  # Reset to start of iteration 2
        POST /api/experiment/reset?iteration_id=2&step_offset=3  # Reset to step 3 of iteration 2
    """
    try:
        service = _get_experiment_service(request)

        # Reset the experiment
        service.reset()

        # Jump to specified position if provided
        if iteration_id:
            state = service.jump_to_iteration(iteration_id, step_offset)
        elif start_step_index > 0:
            state = service.jump_to_step(start_step_index)
        else:
            state = service.get_state()

        # Build response
        progress = state.progress_state
        progress_response = ProgressStateResponse(
            sections=[
                ProgressSectionResponse(
                    slot=s.get("slot", ""),
                    title=s.get("title", ""),
                    messages=s.get("messages", []),
                    collapsible=s.get("collapsible", True),
                    initial_state=s.get("initial_state", "expanded"),
                    type=s.get("type"),
                    tasks=s.get("tasks"),
                    appearance_delay_ms=s.get("appearance_delay_ms", 0.0),
                    prompt_file=s.get("prompt_file"),
                )
                for s in progress.sections
            ],
            revealed_counts=progress.revealed_counts,
            is_animating=progress.is_animating,
            is_complete=progress.is_complete,
            current_step_id=progress.current_step_id,
            phase=progress.phase,
            pre_messages=progress.pre_messages,
            post_messages=progress.post_messages,
            current_pre_message_index=progress.current_pre_message_index,
            progress_header=progress.progress_header,
            keep_progress_sections=progress.keep_progress_sections,
        )

        return ExperimentStatusResponse(
            current_step_index=state.current_step_index,
            current_step_id=state.current_step_id,
            is_complete=state.is_complete,
            is_waiting_for_user=state.is_waiting_for_user,
            progress_state=progress_response,
            suggested_actions=state.suggested_actions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error resetting experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
