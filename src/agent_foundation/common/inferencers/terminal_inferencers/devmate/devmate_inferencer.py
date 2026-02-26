"""DevMate terminal inferencer for executing DevMate CLI commands."""

import os
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO

from attr import attrib, attrs

from science_modeling_tools.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
)


@attrs
class DevmateInferencer(TerminalSessionInferencerBase):
    """
    DevMate-specific inferencer for executing DevMate CLI commands.

    This inferencer wraps the `devmate` CLI tool to execute freeform prompts
    and other DevMate commands programmatically.

    The inferencer automatically sets up a pre-execution script to change
    to the repo directory before executing devmate commands, which is required
    for proper devmate operation.

    Session Support:
        DevMate supports session continuation via --resume and --session-id flags.
        This inferencer inherits from TerminalSessionInferencerBase to provide:
        - Automatic session tracking
        - Multi-turn conversation support
        - Session history management

    Attributes:
        repo_path (str): Path to the repository for DevMate context.
            Defaults to ~/fbsource if not specified.
        model_name (str): Model to use for inference (e.g., 'claude-sonnet-4.5').
            Defaults to 'claude-sonnet-4.5'.
        max_tokens (int): Maximum tokens for response. Defaults to 32768.
        no_create_commit (bool): If True, prevents DevMate from creating commits.
            Defaults to True.
        context_files (List[str]): Optional list of files to include as context.

    Inherited Session Attributes:
        session_arg_name (str): CLI arg for session ID. Defaults to '--session-id'.
        resume_arg_name (str): CLI arg for resume flag. Defaults to '--resume'.
        active_session_id (str): Currently active session ID.
        auto_resume (bool): If True, automatically resume on subsequent calls.

    Example:
        >>> # Single-turn usage
        >>> inferencer = DevmateInferencer(
        ...     repo_path="/path/to/repo",
        ...     model_name="claude-sonnet-4.5",
        ... )
        >>> result = inferencer.infer("Help me understand this code")
        >>> print(result["output"])

        >>> # Multi-turn session usage
        >>> inferencer = DevmateInferencer(auto_resume=True)
        >>> result1 = inferencer.infer("What files are in this directory?")
        >>> session_id = result1.get("session_id")
        >>> # Automatically resumes with same session
        >>> result2 = inferencer.infer("Now show me the content of the first file")
        >>> print(inferencer.get_session_history())  # Shows both turns
    """

    repo_path: Optional[str] = attrib(default=None)
    model_name: str = attrib(default="claude-sonnet-4.5")
    max_tokens: int = attrib(default=32768)
    no_create_commit: bool = attrib(default=True)
    context_files: Optional[List[str]] = attrib(default=None)

    def __attrs_post_init__(self):
        """Set repo_path and configure pre-execution script to cd to repo."""
        # Default repo_path to ~/fbsource if not specified
        if self.repo_path is None:
            self.repo_path = os.path.expanduser("~/fbsource")

        # Set working_dir to repo_path for command execution
        if self.working_dir is None:
            self.working_dir = self.repo_path

        # Set up pre-execution script to cd to repo directory
        # This is required because devmate needs to be run from within the repo
        cd_script = f'cd "{self.repo_path}" || exit 1'

        if self.pre_exec_scripts is None:
            self.pre_exec_scripts = [cd_script]
        elif cd_script not in self.pre_exec_scripts:
            # Insert cd command at the beginning if not already present
            self.pre_exec_scripts.insert(0, cd_script)

        super().__attrs_post_init__()

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """
        Build CLI arguments for DevMate session management.

        DevMate uses --resume and --session-id flags for session continuation:
        - --resume: Flag to indicate resuming a previous session
        - --session-id 'uuid': The session ID to resume

        Args:
            session_id: The session ID to use.
            is_resume: Whether this is resuming an existing session.

        Returns:
            String containing the CLI arguments (e.g., "--resume --session-id 'abc123'").
        """
        if is_resume and session_id:
            return f"{self.resume_arg_name} {self.session_arg_name} '{session_id}'"
        elif session_id:
            # Just pass session ID without resume (for tracking purposes)
            return f"{self.session_arg_name} '{session_id}'"
        return ""

    def construct_command(self, inference_input: Any, **kwargs) -> str:
        """
        Construct the DevMate CLI command as a shell command string.

        This returns a properly quoted shell command string that matches
        the working bash script format:
        devmate run freeform "prompt=$PROMPT" "model_name=$MODEL" "max_tokens=32768" --no-create-commit

        For session continuation (resume), the format is:
        devmate run --resume --session-id 'uuid' freeform "prompt=$PROMPT" ...

        Args:
            inference_input: The prompt string or dict with 'prompt' key.
            **kwargs: Additional arguments:
                - model_name: Override model name
                - max_tokens: Override max tokens
                - context_files: Override context files
                - session_id: Session ID for continuation
                - resume: Whether to resume a previous session

        Returns:
            Shell command string with properly quoted arguments.
        """
        # Extract prompt
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        # Get model and token settings (allow overrides via kwargs)
        model = kwargs.get("model_name", self.model_name)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        context_files = kwargs.get("context_files", self.context_files)

        # Get session settings from kwargs (set by _infer in parent class)
        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)

        # Escape any double quotes in the prompt for shell safety
        escaped_prompt = prompt.replace('"', '\\"')

        # Build the shell command string
        # Base: devmate run
        command_parts = ["devmate", "run"]

        # Add session args if resuming (before 'freeform')
        if is_resume and session_id:
            session_args = self._build_session_args(session_id, is_resume)
            command_parts.append(session_args)

        # Add freeform and parameters
        command_parts.extend(
            [
                "freeform",
                f'"prompt={escaped_prompt}"',
                f'"model_name={model}"',
                f'"max_tokens={max_tokens}"',
            ]
        )

        # Add context files if specified (also quoted)
        if context_files:
            for file_path in context_files:
                command_parts.append(f'"context_file={file_path}"')

        # Add no_create_commit flag at the END (like the working bash script)
        if self.no_create_commit:
            command_parts.append("--no-create-commit")

        # Return as a single shell command string
        return " ".join(command_parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """
        Parse the DevMate command output.

        This method cleans up the raw DevMate output by:
        1. Removing session header/footer blocks (Session ID, Trajectory, etc.)
        2. Extracting the actual response content
        3. Handling duplicate responses (takes the first one)

        Args:
            stdout: Standard output from DevMate command.
            stderr: Standard error from DevMate command.
            return_code: Process return code.

        Returns:
            Dictionary with:
                - output (str): The cleaned main response content
                - raw_output (str): The original unprocessed output
                - stderr (str): Any error output
                - return_code (int): Process return code
                - success (bool): True if command succeeded
                - session_id (str, optional): Extracted session ID
                - trajectory_url (str, optional): Extracted trajectory URL
                - error (str, optional): Error message if failed
        """
        # Store raw output
        raw_output = stdout.strip() if stdout else ""

        # Extract session info before cleaning
        session_id = self._extract_session_id(raw_output)
        trajectory_url = self._extract_trajectory_url(raw_output)

        # Clean up the output
        cleaned_output = self._clean_devmate_output(raw_output)

        result = {
            "output": cleaned_output,
            "raw_output": raw_output,
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }

        # Add session info if extracted
        if session_id:
            result["session_id"] = session_id
        if trajectory_url:
            result["trajectory_url"] = trajectory_url

        # Add error message if failed
        if return_code != 0:
            if stderr:
                result["error"] = stderr.strip()
            else:
                result["error"] = (
                    f"DevMate command failed with return code {return_code}"
                )

        return result

    def _extract_session_id(self, output: str) -> Optional[str]:
        """Extract session ID from DevMate output."""
        import re

        match = re.search(r"Session ID:\s*([a-f0-9-]+)", output)
        if match:
            return match.group(1)
        return None

    def _extract_trajectory_url(self, output: str) -> Optional[str]:
        """Extract trajectory URL from DevMate output."""
        import re

        match = re.search(r"Trajectory:\s*(https?://\S+)", output)
        if match:
            return match.group(1)
        return None

    def _clean_devmate_output(self, output: str) -> str:
        """
        Clean DevMate output by removing session headers/footers.

        DevMate output typically has this structure:
        - Header: "Starting Devmate server...", session info block
        - Content: The actual response
        - Footer: "Finished session...", session info block repeated

        This method extracts just the actual response content.
        """
        if not output:
            return ""

        import re

        # Split into lines for processing
        lines = output.split("\n")
        cleaned_lines = []

        # Patterns to identify session-related lines (case insensitive for robustness)
        session_patterns = [
            r"^Starting Devmate server",
            r"^Finished starting Devmate server",
            r"^Started session",
            r"^Finished session",
            r"^Session ID:",
            r"^Trajectory:",
            r"^Server logs available at:",
            r"^=+$",  # Separator lines (================)
        ]

        # Compile patterns (case insensitive)
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in session_patterns]

        for line in lines:
            stripped_line = line.strip()

            # Skip empty lines at the beginning
            if not cleaned_lines and not stripped_line:
                continue

            # Check if this line matches any session pattern
            is_session_line = any(
                pattern.match(stripped_line) for pattern in compiled_patterns
            )

            if is_session_line:
                # Skip session-related lines
                continue
            else:
                # This is content
                cleaned_lines.append(line)

        # Join cleaned lines and strip trailing whitespace/empty lines
        cleaned_output = "\n".join(cleaned_lines).strip()

        # Remove any trailing session info that might be on the same line
        # e.g., "response text Finished session abc123"
        cleaned_output = re.sub(
            r"\s*Finished session\s+[a-f0-9-]+\s*$",
            "",
            cleaned_output,
            flags=re.IGNORECASE,
        )

        # Handle duplicate responses (DevMate sometimes outputs response twice)
        # Look for `---` separator that might indicate duplication
        if "---\n" in cleaned_output:
            parts = cleaned_output.split("---\n")
            if len(parts) >= 2:
                # Check if first and second parts are similar (duplicate)
                first_part = parts[0].strip()
                second_part = parts[1].strip() if len(parts) > 1 else ""

                # If they look similar (both start/end similarly), take just the first
                if first_part and second_part:
                    # Simple heuristic: if second part starts like first part
                    first_lines = first_part.split("\n")[:3]
                    second_lines = second_part.split("\n")[:3]

                    if first_lines == second_lines:
                        # It's a duplicate, return just the first part
                        cleaned_output = first_part

        return cleaned_output

    def get_response_text(self, result: Dict[str, Any]) -> str:
        """
        Extract just the response text from a result dictionary.

        This is a convenience method for getting the main output.

        Args:
            result: The result dictionary from infer().

        Returns:
            The main response text, or error message if failed.
        """
        if result.get("success"):
            return result.get("output", "")
        else:
            return result.get("error", "Unknown error occurred")

    # === Session Convenience Methods ===

    def resume_session(
        self, prompt: str, session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Resume a previous session with a new prompt.

        This is a convenience method that explicitly resumes a session.

        Args:
            prompt: The new prompt to send.
            session_id: Session ID to resume. If None, uses active_session_id.
            **kwargs: Additional arguments passed to infer().

        Returns:
            Result dictionary from infer().

        Raises:
            ValueError: If no session_id is provided and no active session exists.
        """
        target_session = session_id or self.active_session_id
        if not target_session:
            raise ValueError(
                "No session_id provided and no active session. "
                "Please provide a session_id or start a session first."
            )

        return self.infer(prompt, session_id=target_session, resume=True, **kwargs)

    def new_session(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Start a new session with a prompt.

        This explicitly starts a new session, ignoring any active session.

        Args:
            prompt: The initial prompt for the new session.
            **kwargs: Additional arguments passed to infer().

        Returns:
            Result dictionary from infer().
        """
        # Clear any active session to ensure a fresh start
        self.active_session_id = None
        return self.infer(prompt, resume=False, **kwargs)

    # === Streaming Methods ===

    def _is_session_info_line(self, line: str) -> bool:
        """
        Check if a line is session-related info that should be filtered.

        Args:
            line: The line to check.

        Returns:
            True if the line should be filtered out (session header/footer).
        """
        import re

        stripped = line.strip()

        # Empty lines pass through (will be filtered later if needed)
        if not stripped:
            return False

        # Patterns for session info lines to filter
        session_patterns = [
            r"^Starting Devmate server",
            r"^Finished starting Devmate server",
            r"^Started session",
            r"^Finished session",
            r"^Session ID:",
            r"^Trajectory:",
            r"^Server logs available at:",
            r"^=+$",  # Separator lines (================)
        ]

        for pattern in session_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        return False

    def infer_streaming(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        output_stream: Optional[TextIO] = None,
        filter_session_info: bool = True,
        **kwargs,
    ) -> Iterator[str]:
        """
        Execute DevMate inference with streaming output.

        This method yields output lines as they become available from the
        DevMate CLI, allowing real-time display of responses.

        Args:
            prompt: The prompt to send to DevMate.
            stream_callback: Optional callback invoked for each output line.
            output_stream: Optional TextIO stream to write output to.
            filter_session_info: If True (default), filters out session ID,
                trajectory URL, and other session header/footer info from
                the streaming output. The session info is still available
                via get_streaming_result() after streaming completes.
            **kwargs: Additional arguments:
                - model_name: Override model name
                - max_tokens: Override max tokens
                - context_files: Override context files
                - session_id: Session ID for continuation
                - resume: Whether to resume a previous session
                - new_session: If True, forces a new session

        Yields:
            Lines of DevMate output as they become available.

        Note:
            After iteration completes, call get_streaming_result() to get
            the parsed output with session ID and other metadata.

        Example:
            >>> inferencer = DevmateInferencer()
            >>> for line in inferencer.infer_streaming("Explain this code"):
            ...     print(line, end="")  # Print as it streams
            >>> result = inferencer.get_streaming_result()
            >>> print(f"Session ID: {result.get('session_id')}")
        """
        # Handle new_session flag
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None

        # Determine session context (same logic as parent's _infer)
        session_id = kwargs.get("session_id", self.active_session_id)
        is_resume = kwargs.get("resume", True)

        # If no session to resume, this will be a new session
        if session_id is None:
            is_resume = False

        # Update kwargs with session info for construct_command
        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume

        if is_resume and session_id:
            self.log_debug(
                f"Streaming with session resume: {session_id[:8]}...", "Stream"
            )
        else:
            self.log_debug("Streaming new session", "Stream")

        # Track state for filtering
        content_started = False
        pending_empty_lines = []

        # Use the parent's _infer_streaming method with filtering
        for line in self._infer_streaming(
            {"prompt": prompt},
            stream_callback=None,  # We handle callback ourselves after filtering
            output_stream=None,  # We handle output stream ourselves after filtering
            **kwargs,
        ):
            # Always accumulate for get_streaming_result()
            # (parent already does this via _last_streaming_output)

            if filter_session_info:
                # Check if this is a session info line
                if self._is_session_info_line(line):
                    # Skip session info lines
                    continue

                # Handle empty lines
                stripped = line.strip()
                if not stripped:
                    # Buffer empty lines - only output them if content follows
                    if content_started:
                        pending_empty_lines.append(line)
                    continue

                # This is actual content
                content_started = True

                # Output any pending empty lines first
                for empty_line in pending_empty_lines:
                    if stream_callback:
                        stream_callback(empty_line)
                    if output_stream:
                        output_stream.write(empty_line)
                        output_stream.flush()
                    yield empty_line
                pending_empty_lines = []

                # Output the content line
                if stream_callback:
                    stream_callback(line)
                if output_stream:
                    output_stream.write(line)
                    output_stream.flush()
                yield line
            else:
                # No filtering - pass through everything
                if stream_callback:
                    stream_callback(line)
                if output_stream:
                    output_stream.write(line)
                    output_stream.flush()
                yield line

    def get_streaming_result(self) -> Dict[str, Any]:
        """
        Get the final parsed result after streaming completes.

        Call this method after exhausting the iterator from infer_streaming()
        to get the parsed output with session ID, trajectory URL, and other
        metadata extracted.

        Returns:
            Dictionary with:
                - output (str): The cleaned main response content
                - raw_output (str): The original unprocessed output
                - stderr (str): Any error output (empty for streaming)
                - return_code (int): Process return code
                - success (bool): True if command succeeded
                - session_id (str, optional): Extracted session ID
                - trajectory_url (str, optional): Extracted trajectory URL

        Example:
            >>> inferencer = DevmateInferencer()
            >>> # Consume the streaming iterator
            >>> output = "".join(inferencer.infer_streaming("Hello"))
            >>> # Now get the parsed result
            >>> result = inferencer.get_streaming_result()
            >>> session_id = result.get("session_id")
        """
        stdout = getattr(self, "_last_streaming_output", "")
        return_code = getattr(self, "_last_streaming_return_code", 0)

        # Use the existing parse_output method
        result = self.parse_output(stdout, "", return_code)

        # Update active session if we got a new session ID
        session_id = result.get("session_id")
        if session_id and session_id != self.active_session_id:
            self.active_session_id = session_id
            self.log_debug(f"Updated active session to: {session_id[:8]}...", "Stream")

        return result
