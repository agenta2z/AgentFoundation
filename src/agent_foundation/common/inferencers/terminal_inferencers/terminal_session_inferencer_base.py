"""Abstract base class for terminal-based inferencers with session support."""

import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from attr import attrib, attrs

from science_modeling_tools.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
)


@attrs
class TerminalSessionInferencerBase(TerminalInferencerBase):
    """
    Abstract base class for terminal-based inferencers with session management.

    This class extends TerminalInferencerBase to provide session tracking capabilities,
    allowing multi-turn conversations with CLI tools that support session continuation.

    Default Behavior:
        - First call: Creates a new session (since active_session_id is None)
        - Subsequent calls: Automatically resumes the active session
        - Use new_session=True to force a new session

    Subclasses should implement:
        - _build_session_args(): Build CLI args for session/resume functionality

    Attributes:
        session_arg_name (str): CLI argument name for session ID (e.g., '--session-id').
        resume_arg_name (str): CLI argument name for resume flag (e.g., '--resume').
        active_session_id (str): Currently active session ID for continuation.

    Session Storage:
        _sessions: Dict mapping session_id -> list of historical turns.
        Each turn has:
            - from: 'user' or 'system'
            - content: The message content
            - timestamp: When the turn occurred

    Example:
        >>> inferencer = MySessionInferencer()
        >>> # First call starts a new session (active_session_id is None)
        >>> result1 = inferencer.infer("Hello")
        >>> print(inferencer.active_session_id)  # e.g., 'abc-123-def'
        >>> # Second call automatically resumes (active_session_id exists)
        >>> result2 = inferencer.infer("What was my last message?")
        >>> print(inferencer.get_session_history())  # Shows both turns
        >>> # Force a new session
        >>> result3 = inferencer.infer("Start fresh", new_session=True)
    """

    # CLI argument names for session management
    session_arg_name: str = attrib(default="--session-id")
    resume_arg_name: str = attrib(default="--resume")

    # Currently active session ID
    active_session_id: Optional[str] = attrib(default=None)

    # Internal session storage (not an init parameter)
    _sessions: Dict[str, List[Dict[str, Any]]] = attrib(factory=dict, init=False)

    # === Session Management Methods ===

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new session.

        Args:
            session_id: Optional session ID. If None, generates a new UUID.

        Returns:
            The session ID (either provided or generated).
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self.active_session_id = session_id
        self.log_debug(f"Started session: {session_id}", "Session")
        return session_id

    def end_session(self, session_id: Optional[str] = None) -> None:
        """
        End a session (clears active_session_id if it matches).

        Args:
            session_id: Session ID to end. If None, ends the active session.
        """
        target_id = session_id or self.active_session_id

        if target_id and self.active_session_id == target_id:
            self.active_session_id = None
            self.log_debug(f"Ended session: {target_id}", "Session")

    def get_session_history(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the history of turns for a session.

        Args:
            session_id: Session ID to get history for. If None, uses active session.

        Returns:
            List of turn dictionaries with 'from', 'content', 'timestamp'.
        """
        target_id = session_id or self.active_session_id
        if target_id and target_id in self._sessions:
            return self._sessions[target_id].copy()
        return []

    def clear_session(self, session_id: Optional[str] = None) -> None:
        """
        Clear the history for a session.

        Args:
            session_id: Session ID to clear. If None, clears active session.
        """
        target_id = session_id or self.active_session_id
        if target_id and target_id in self._sessions:
            self._sessions[target_id] = []
            self.log_debug(f"Cleared session history: {target_id}", "Session")

    def delete_session(self, session_id: Optional[str] = None) -> None:
        """
        Delete a session entirely.

        Args:
            session_id: Session ID to delete. If None, deletes active session.
        """
        target_id = session_id or self.active_session_id
        if target_id:
            if target_id in self._sessions:
                del self._sessions[target_id]
            if self.active_session_id == target_id:
                self.active_session_id = None
            self.log_debug(f"Deleted session: {target_id}", "Session")

    def list_sessions(self) -> List[str]:
        """
        List all session IDs.

        Returns:
            List of session IDs.
        """
        return list(self._sessions.keys())

    def get_session_turn_count(self, session_id: Optional[str] = None) -> int:
        """
        Get the number of turns in a session.

        Args:
            session_id: Session ID. If None, uses active session.

        Returns:
            Number of turns in the session.
        """
        target_id = session_id or self.active_session_id
        if target_id and target_id in self._sessions:
            return len(self._sessions[target_id])
        return 0

    # === Turn Management Methods ===

    def _add_turn(
        self,
        session_id: str,
        from_: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a turn to a session's history.

        Args:
            session_id: The session ID to add the turn to.
            from_: Who the turn is from ('user' or 'system').
            content: The content of the turn.
            metadata: Optional additional metadata for the turn.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        turn = {
            "from": from_,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if metadata:
            turn["metadata"] = metadata

        self._sessions[session_id].append(turn)
        self.log_debug(f"Added {from_} turn to session {session_id[:8]}...", "Session")

    def get_last_turn(
        self, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last turn from a session.

        Args:
            session_id: Session ID. If None, uses active session.

        Returns:
            The last turn dictionary, or None if no turns.
        """
        history = self.get_session_history(session_id)
        return history[-1] if history else None

    def get_last_user_turn(
        self, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last user turn from a session.

        Args:
            session_id: Session ID. If None, uses active session.

        Returns:
            The last user turn dictionary, or None if no user turns.
        """
        history = self.get_session_history(session_id)
        for turn in reversed(history):
            if turn.get("from") == "user":
                return turn
        return None

    def get_last_system_turn(
        self, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last system turn from a session.

        Args:
            session_id: Session ID. If None, uses active session.

        Returns:
            The last system turn dictionary, or None if no system turns.
        """
        history = self.get_session_history(session_id)
        for turn in reversed(history):
            if turn.get("from") == "system":
                return turn
        return None

    # === Abstract Method for Subclasses ===

    @abstractmethod
    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """
        Build CLI arguments for session management.

        This method should be implemented by subclasses to construct the
        appropriate CLI arguments for session ID and resume functionality.

        Args:
            session_id: The session ID to use.
            is_resume: Whether this is resuming an existing session.

        Returns:
            String containing the CLI arguments for session (e.g., '--resume --session-id abc123').

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    # === Override _infer to Track Sessions ===

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs
    ) -> Any:
        """
        Execute inference with session tracking.

        Default behavior:
        - resume=True by default
        - If active_session_id is None, starts a new session (acts as resume=False)
        - If active_session_id exists, resumes that session

        This method extends the base _infer to:
        1. Handle session continuation automatically
        2. Track user input as a turn
        3. Track system response as a turn
        4. Extract and store session ID from response

        Args:
            inference_input: Input for the inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments.
                - session_id: Override session ID
                - resume: Whether to resume (default True, but ignored if no session)
                - new_session: If True, forces a new session (clears active_session_id)

        Returns:
            Parsed output from parse_output() with session tracking.
        """
        # Check if user wants to force a new session
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None

        # Determine session context
        session_id = kwargs.get("session_id", self.active_session_id)

        # Default resume=True, but if session_id is None, treat as new session
        is_resume = kwargs.get("resume", True)
        if session_id is None:
            # No session to resume, this will be a new session
            is_resume = False

        # Update kwargs with session info for construct_command
        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume

        if is_resume:
            self.log_debug(f"Resuming session: {session_id[:8]}...", "Session")
        else:
            self.log_debug("Starting new session", "Session")

        # Extract the prompt content for turn tracking
        if isinstance(inference_input, dict):
            prompt_content = inference_input.get("prompt", str(inference_input))
        else:
            prompt_content = str(inference_input)

        # Call parent's _infer
        result = super()._infer(inference_input, inference_config, **kwargs)

        # Extract session ID from result if available
        result_session_id = (
            result.get("session_id") if isinstance(result, dict) else None
        )

        # Determine the final session ID (from result or from input)
        final_session_id = result_session_id or session_id

        # If we got a new session ID, update active session
        if final_session_id:
            if final_session_id != self.active_session_id:
                self.active_session_id = final_session_id
                self.log_debug(
                    f"Updated active session to: {final_session_id[:8]}...", "Session"
                )

            # Track the user turn
            self._add_turn(
                final_session_id,
                "user",
                prompt_content,
                metadata={"resume": is_resume},
            )

            # Track the system turn
            if isinstance(result, dict):
                response_content = result.get("output", "")
                self._add_turn(
                    final_session_id,
                    "system",
                    response_content,
                    metadata={
                        "success": result.get("success", False),
                        "session_id": final_session_id,
                    },
                )

        return result
