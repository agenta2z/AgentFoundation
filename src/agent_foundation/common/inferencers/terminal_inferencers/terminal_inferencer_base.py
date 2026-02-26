"""Abstract base class for terminal-based inferencers."""

import os
import subprocess
import tempfile
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO, Union

from attr import attrib, attrs

from science_modeling_tools.common.inferencers.inferencer_base import InferencerBase


@attrs
class TerminalInferencerBase(InferencerBase):
    """
    Abstract base class for executing terminal commands as inference.

    This class extends InferencerBase to provide a framework for executing
    CLI commands with built-in retry logic, timeout handling, and output capture.

    Subclasses should implement:
        - construct_command(): Build the CLI command from inference input
        - parse_output(): Parse the command output into desired format

    Attributes:
        working_dir (str): Directory to execute commands in. Defaults to current dir.
        timeout (int): Command execution timeout in seconds. Defaults to 300 (5 min).
        output_file (str): Optional file path to save output. If None, uses temp file.
        capture_stderr (bool): Whether to capture stderr. Defaults to True.
        env_vars (dict): Additional environment variables for command execution.
        pre_exec_scripts (List[str]): Shell scripts to run BEFORE the main command.
            Useful for setup like 'cd /path/to/repo', 'export VAR=value', etc.
        post_exec_scripts (List[str]): Shell scripts to run AFTER the main command.
            Useful for cleanup or post-processing.
        fail_on_pre_script_error (bool): If True, abort if pre-script fails. Default True.
        fail_on_post_script_error (bool): If True, fail if post-script fails. Default False.
    """

    working_dir: str = attrib(default=None)
    timeout: int = attrib(default=300)
    output_file: Optional[str] = attrib(default=None)
    capture_stderr: bool = attrib(default=True)
    env_vars: Dict[str, str] = attrib(factory=dict)

    # Pre/post execution scripts
    pre_exec_scripts: Optional[List[str]] = attrib(default=None)
    post_exec_scripts: Optional[List[str]] = attrib(default=None)
    fail_on_pre_script_error: bool = attrib(default=True)
    fail_on_post_script_error: bool = attrib(default=False)

    def __attrs_post_init__(self):
        """Initialize working directory if not set."""
        if self.working_dir is None:
            self.working_dir = os.getcwd()
        super().__attrs_post_init__()

    @abstractmethod
    def construct_command(self, inference_input: Any, **kwargs) -> "List[str] | str":
        """
        Construct the CLI command to execute.

        Args:
            inference_input: The input data for inference (e.g., prompt, file path).
            **kwargs: Additional arguments for command construction.

        Returns:
            Either a list of command parts (for subprocess with shell=False)
            or a shell command string (for subprocess with shell=True).
            When a string is returned, shell=True is used to support shell syntax.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str, return_code: int) -> Any:
        """
        Parse the command output into the desired format.

        Args:
            stdout: Standard output from the command.
            stderr: Standard error from the command.
            return_code: Process return code.

        Returns:
            Parsed output in the subclass-specific format.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _execute_script(
        self, script: str, cwd: Optional[str] = None, env: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a single shell script.

        Args:
            script: Shell script/command to execute.
            cwd: Working directory for execution.
            env: Environment variables.

        Returns:
            Dictionary with stdout, stderr, return_code, success.
        """
        if env is None:
            env = os.environ.copy()
            if self.env_vars:
                env.update(self.env_vars)

        if cwd is None:
            cwd = self.working_dir

        self.log_debug(f"Executing script: {script}", "Script")

        try:
            result = subprocess.run(
                script,
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr if self.capture_stderr else "",
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "script": script,
            }

        except subprocess.TimeoutExpired as e:
            self.log_debug(f"Script timed out after {self.timeout}s", "Timeout")
            return {
                "stdout": e.stdout or "" if e.stdout else "",
                "stderr": str(e) if self.capture_stderr else "",
                "return_code": -1,
                "success": False,
                "script": script,
                "error": f"Timeout after {self.timeout} seconds",
            }

        except Exception as e:
            self.log_debug(f"Script execution error: {e}", "Error")
            return {
                "stdout": "",
                "stderr": str(e) if self.capture_stderr else "",
                "return_code": -1,
                "success": False,
                "script": script,
                "error": str(e),
            }

    def _execute_scripts(
        self,
        scripts: List[str],
        script_type: str = "pre",
        cwd: Optional[str] = None,
        env: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute a list of shell scripts sequentially.

        Args:
            scripts: List of shell scripts/commands to execute.
            script_type: Type of scripts ("pre" or "post") for logging.
            cwd: Working directory for execution.
            env: Environment variables.

        Returns:
            Dictionary with:
                - success (bool): True if all scripts succeeded
                - results (List[Dict]): Individual results for each script
                - failed_script (str, optional): The script that failed
                - error (str, optional): Error message if any script failed
        """
        results = []
        for i, script in enumerate(scripts):
            self.log_debug(
                f"Running {script_type}-script {i + 1}/{len(scripts)}: {script}",
                f"{script_type.title()}Script",
            )

            result = self._execute_script(script, cwd=cwd, env=env)
            results.append(result)

            if not result["success"]:
                return {
                    "success": False,
                    "results": results,
                    "failed_script": script,
                    "error": result.get("error", result.get("stderr", "Script failed")),
                }

        return {
            "success": True,
            "results": results,
        }

    def _execute_command(self, command: "List[str] | str", **kwargs) -> Dict[str, Any]:
        """
        Execute a subprocess command with timeout and output capture.

        Args:
            command: Either a list of command parts (shell=False) or a shell
                command string (shell=True). When a string is passed, shell=True
                is used automatically to support shell syntax like quoting.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Returns:
            Dictionary with:
                - stdout (str): Standard output
                - stderr (str): Standard error
                - return_code (int): Process return code
                - success (bool): True if return_code is 0
                - command: The executed command (list or string)

        Raises:
            subprocess.TimeoutExpired: If command exceeds timeout.
            Exception: Any other subprocess execution errors.
        """
        # Build environment with custom vars
        env = os.environ.copy()
        if self.env_vars:
            env.update(self.env_vars)

        # Determine if we should use shell mode
        # If command is a string, use shell=True to support shell syntax
        use_shell = isinstance(command, str)

        if use_shell:
            self.log_debug(f"Executing shell command: {command}", "Command")
        else:
            self.log_debug(f"Executing command: {' '.join(command)}", "Command")
        self.log_debug(f"Working directory: {self.working_dir}", "WorkingDir")

        try:
            result = subprocess.run(
                command,
                shell=use_shell,
                cwd=self.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr if self.capture_stderr else "",
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "command": command,
            }

        except subprocess.TimeoutExpired as e:
            self.log_debug(f"Command timed out after {self.timeout}s", "Timeout")
            return {
                "stdout": e.stdout or "",
                "stderr": str(e) if self.capture_stderr else "",
                "return_code": -1,
                "success": False,
                "command": command,
                "error": f"Timeout after {self.timeout} seconds",
            }

        except Exception as e:
            self.log_debug(f"Command execution error: {e}", "Error")
            return {
                "stdout": "",
                "stderr": str(e) if self.capture_stderr else "",
                "return_code": -1,
                "success": False,
                "command": command,
                "error": str(e),
            }

    def _execute_command_streaming(
        self,
        command: Union[List[str], str],
        stream_callback: Optional[Callable[[str], None]] = None,
        output_stream: Optional[TextIO] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Execute a subprocess command with streaming output.

        Uses subprocess.Popen to stream output line-by-line as it becomes
        available, rather than waiting for the command to complete.

        Args:
            command: Either a list of command parts (shell=False) or a shell
                command string (shell=True).
            stream_callback: Optional callback invoked for each output line.
            output_stream: Optional TextIO stream to write output to.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Yields:
            Lines of output as they become available.

        Note:
            After iteration completes, the following attributes are set:
            - _last_streaming_output: All accumulated output as a string
            - _last_streaming_return_code: The process return code
        """
        # Build environment with custom vars
        env = os.environ.copy()
        if self.env_vars:
            env.update(self.env_vars)

        # Determine if we should use shell mode
        use_shell = isinstance(command, str)

        if use_shell:
            self.log_debug(f"Streaming shell command: {command}", "StreamCommand")
        else:
            self.log_debug(f"Streaming command: {' '.join(command)}", "StreamCommand")

        accumulated_output: List[str] = []

        try:
            process = subprocess.Popen(
                command,
                shell=use_shell,
                cwd=self.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout for streaming
                text=True,
                bufsize=1,  # Line buffered
            )

            # Read output line by line
            for line in iter(process.stdout.readline, ""):
                accumulated_output.append(line)

                # Write to custom stream if provided
                if output_stream:
                    output_stream.write(line)
                    output_stream.flush()

                # Call callback if provided
                if stream_callback:
                    stream_callback(line)

                yield line

            # Close stdout and wait for process to complete
            process.stdout.close()
            process.wait()

            # Store final result for later retrieval
            self._last_streaming_output = "".join(accumulated_output)
            self._last_streaming_return_code = process.returncode

        except Exception as e:
            self.log_debug(f"Streaming command error: {e}", "StreamError")
            self._last_streaming_output = "".join(accumulated_output)
            self._last_streaming_return_code = -1
            yield f"Error: {str(e)}\n"

    def _infer_streaming(
        self,
        inference_input: Any,
        inference_config: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        output_stream: Optional[TextIO] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Execute inference with streaming output.

        This method is similar to _infer() but yields output lines as they
        become available instead of waiting for completion.

        Args:
            inference_input: Input for the inference (passed to construct_command).
            inference_config: Optional configuration (currently unused).
            stream_callback: Optional callback invoked for each output line.
            output_stream: Optional TextIO stream to write output to.
            **kwargs: Additional arguments passed to construct_command.

        Yields:
            Lines of output as they become available from the command.

        Note:
            - Pre-execution scripts run synchronously before streaming starts
            - Post-execution scripts are NOT run in streaming mode
            - After iteration, call get_streaming_result() for parsed output
        """
        # Get pre scripts (allow override via kwargs)
        pre_scripts = kwargs.pop("pre_exec_scripts", self.pre_exec_scripts)

        # 1. Run pre-execution scripts (synchronous, blocking)
        if pre_scripts:
            self.log_debug(f"Running {len(pre_scripts)} pre-execution script(s)", "Pre")
            pre_result = self._execute_scripts(pre_scripts, script_type="pre")

            if not pre_result["success"] and self.fail_on_pre_script_error:
                yield f"Pre-execution script failed: {pre_result.get('error', '')}\n"
                self._last_streaming_output = ""
                self._last_streaming_return_code = -1
                return

        # 2. Construct the command
        command = self.construct_command(inference_input, **kwargs)

        # 3. Execute with streaming
        yield from self._execute_command_streaming(
            command,
            stream_callback=stream_callback,
            output_stream=output_stream,
            **kwargs,
        )

    def _save_output(self, content: str, filename: Optional[str] = None) -> str:
        """
        Save output content to a file.

        Args:
            content: Content to save.
            filename: Optional filename. If None, creates a temp file.

        Returns:
            Path to the saved file.
        """
        if filename:
            filepath = filename
        elif self.output_file:
            filepath = self.output_file
        else:
            # Create a temp file
            fd, filepath = tempfile.mkstemp(suffix=".txt", prefix="terminal_output_")
            os.close(fd)

        with open(filepath, "w") as f:
            f.write(content)

        self.log_debug(f"Output saved to: {filepath}", "OutputFile")
        return filepath

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs
    ) -> Any:
        """
        Execute the terminal command and return parsed output.

        This method:
        1. Runs pre-execution scripts (if any)
        2. Constructs the command using construct_command()
        3. Executes the command using _execute_command()
        4. Runs post-execution scripts (if any)
        5. Optionally saves output to file
        6. Parses output using parse_output()

        Args:
            inference_input: Input for the inference (passed to construct_command).
            inference_config: Optional configuration (currently unused).
            **kwargs: Additional arguments passed to construct_command.
                - pre_exec_scripts: Override pre-execution scripts
                - post_exec_scripts: Override post-execution scripts

        Returns:
            Parsed output from parse_output().
        """
        # Get pre/post scripts (allow override via kwargs)
        pre_scripts = kwargs.pop("pre_exec_scripts", self.pre_exec_scripts)
        post_scripts = kwargs.pop("post_exec_scripts", self.post_exec_scripts)

        # 1. Run pre-execution scripts
        if pre_scripts:
            self.log_debug(f"Running {len(pre_scripts)} pre-execution script(s)", "Pre")
            pre_result = self._execute_scripts(pre_scripts, script_type="pre")

            if not pre_result["success"] and self.fail_on_pre_script_error:
                # Return early with error if pre-script fails
                return self.parse_output(
                    stdout="",
                    stderr=f"Pre-execution script failed: {pre_result.get('error', '')}",
                    return_code=-1,
                )

        # 2. Construct the command
        command = self.construct_command(inference_input, **kwargs)

        # 3. Execute the main command
        result = self._execute_command(command, **kwargs)

        # 4. Run post-execution scripts
        post_script_failed = False
        if post_scripts:
            self.log_debug(
                f"Running {len(post_scripts)} post-execution script(s)", "Post"
            )
            post_result = self._execute_scripts(post_scripts, script_type="post")

            if not post_result["success"] and self.fail_on_post_script_error:
                # Append post-script error to stderr
                result["stderr"] = (
                    result.get("stderr", "")
                    + f"\nPost-execution script failed: {post_result.get('error', '')}"
                )
                result["success"] = False
                post_script_failed = True

        # 5. Optionally save output
        if self.output_file or kwargs.get("save_output"):
            output_path = kwargs.get("output_path", self.output_file)
            if result.get("stdout"):
                self._save_output(result["stdout"], output_path)

        # 6. Parse and return
        parsed_result = self.parse_output(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            return_code=result.get("return_code", -1),
        )

        # Override success if post-script failed and fail_on_post_script_error is True
        if post_script_failed:
            parsed_result["success"] = False

        return parsed_result
