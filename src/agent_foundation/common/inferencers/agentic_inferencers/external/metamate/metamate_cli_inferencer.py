# pyre-strict
"""MetaMate CLI Inferencer.

Wraps the ``query_metamate`` buck target as a subprocess-based
TerminalSessionInferencerBase implementation.

Subprocess Dependency:
    ``buck run fbcode//agent_foundation.common.inferencers.agentic_inferencers.external.metamate:query_metamate``
    No Python import from ``agent_foundation.common.inferencers.agentic_inferencers.external.metamate/``.
"""

import logging
import shlex
from typing import Any, List, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalInferencerResponse,
    TerminalSessionInferencerBase,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    MetamateAgent,
)

logger: logging.Logger = logging.getLogger(__name__)

_BUCK_TARGET: str = "fbcode//agent_foundation/common/inferencers/agentic_inferencers/external/metamate:query_metamate"


@attrs
class MetamateCliInferencer(TerminalSessionInferencerBase):
    """MetaMate CLI inferencer via subprocess execution.

    Builds and executes ``buck run <target> -- --query ...`` commands
    against the ``query_metamate`` buck target.

    Inherits from TerminalSessionInferencerBase which provides:
    - ``_ainfer_streaming()`` → streams subprocess stdout line-by-line
    - ``ainfer_streaming()`` / ``infer_streaming()`` with idle timeout
    - ``_infer()`` via ``subprocess.run()`` (already implemented)
    - Session management (inherited from StreamingInferencerBase)

    This class implements the three abstract methods:
    - ``construct_command()`` — builds the shell command
    - ``parse_output()`` — parses stdout/stderr into response
    - ``_build_session_args()`` — returns empty string (CLI is single-turn)

    Usage Patterns:
        # Simple query:
        inferencer = MetamateCliInferencer()
        result = inferencer("What is MetaMate?")
        print(result)

        # Deep research:
        inferencer = MetamateCliInferencer(deep_research=True, timeout_seconds=600)
        result = inferencer("Research how ranking models work")

        # With specific agent:
        inferencer = MetamateCliInferencer(agent_name="METAMATE_GENERAL_AGENT")
        result = inferencer("Search for auth docs")

    Attributes:
        api_key: MetaMate API key.
        agent_name: Optional agent name override.
        deep_research: If True, enables deep research mode.
        timeout_seconds: Timeout for the CLI request.
        extra_cli_args: Additional CLI arguments to pass.
    """

    api_key: str = attrib(default=DEFAULT_API_KEY)
    agent_name: Optional[str] = attrib(default=None)
    deep_research: bool = attrib(default=False)
    timeout_seconds: int = attrib(default=DEFAULT_TIMEOUT)
    extra_cli_args: Optional[List[str]] = attrib(default=None)

    # === Abstract Method Implementations ===

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the ``buck run`` command for query_metamate.

        Args:
            inference_input: The prompt string or dict with "prompt" key.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Shell command string.
        """
        prompt = self._extract_prompt(inference_input)

        parts = [
            f"buck run {_BUCK_TARGET} --",
            f"--query {shlex.quote(prompt)}",
            f"--api-key {self.api_key}",
        ]

        agent = self.agent_name
        if self.deep_research and agent is None:
            agent = MetamateAgent.DEEP_RESEARCH.value

        if agent:
            parts.append(f"--agent-name {agent}")

        if self.deep_research:
            parts.append("--deep-research")

        if self.timeout_seconds:
            parts.append(f"--timeout {self.timeout_seconds}")

        if self.extra_cli_args:
            parts.extend(self.extra_cli_args)

        return " ".join(parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> TerminalInferencerResponse:
        """Parse CLI output into a response object.

        Extracts the response text from between the RESPONSE delimiters
        in the structured stdout format of ``query_metamate``.

        Args:
            stdout: Standard output from the command.
            stderr: Standard error from the command.
            return_code: Process return code.

        Returns:
            TerminalInferencerResponse with parsed output.
        """
        output_text = stdout.strip()

        # Extract response text from query_metamate's structured output:
        #   --------   ← delimiter
        #   RESPONSE   ← marker
        #   --------   ← delimiter
        #   <text>     ← response content
        #   --------   ← delimiter (metadata section follows)
        #
        # Phase 1: find the "RESPONSE" marker.
        # Phase 2: skip the delimiter immediately after it, then collect
        #          lines until the next delimiter.
        response_marker = "RESPONSE"
        delimiter = "-" * 72
        lines = output_text.split("\n")
        response_lines: list[str] = []
        found_response_marker = False
        collecting = False

        for line in lines:
            stripped = line.strip()

            if not found_response_marker:
                if stripped == response_marker:
                    found_response_marker = True
                continue

            if stripped == delimiter:
                if collecting:
                    break
                collecting = True
                continue

            if collecting:
                response_lines.append(line)

        parsed_text = (
            "\n".join(response_lines).strip() if response_lines else output_text
        )

        return TerminalInferencerResponse(
            output=parsed_text,
            raw_output=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build CLI session arguments.

        The MetaMate CLI (``query_metamate``) is single-turn and does not
        support session continuation via CLI flags. Returns empty string.

        Args:
            session_id: The session ID (unused).
            is_resume: Whether this is a resume operation (unused).

        Returns:
            Empty string.
        """
        if session_id:
            logger.warning(
                "[%s] MetaMate CLI is single-turn; session_id=%s will be ignored. "
                "Use MetamateSDKInferencer for multi-turn conversations.",
                self.__class__.__name__,
                session_id[:8] if session_id else None,
            )
        return ""
