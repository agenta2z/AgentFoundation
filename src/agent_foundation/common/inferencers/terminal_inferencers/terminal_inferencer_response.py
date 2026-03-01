"""Response type for terminal/CLI-based inferencers.

Provides a structured response with a clean ``__str__`` so that
``DualInferencer`` (which does ``str(await inferencer.ainfer(...))``)
gets the meaningful output text rather than a raw dict repr.

Follows the same pattern as ``SDKInferencerResponse`` used by
``ClaudeCodeInferencer`` and ``DevmateSDKInferencer``.
"""

from typing import Any, Dict, Optional

from attr import attrib, attrs


@attrs
class TerminalInferencerResponse:
    """Standard response type for terminal/CLI-based inferencers.

    Wraps the dict returned by ``parse_output()`` into a typed object
    whose ``__str__`` returns the clean output text.

    Attributes:
        output: The cleaned main response content.
        raw_output: The original unprocessed CLI output.
        stderr: Standard error output from the command.
        return_code: Process return code.
        success: True if the command succeeded (return_code == 0).
        session_id: Optional session identifier for multi-turn conversations.
        trajectory_url: Optional URL for inspection/debugging.
        dump_data: Optional full dump data (when dump_output is enabled).
        error: Optional error message if the command failed.
    """

    output: str = attrib(default="")
    raw_output: str = attrib(default="")
    stderr: str = attrib(default="")
    return_code: int = attrib(default=0)
    success: bool = attrib(default=True)
    session_id: Optional[str] = attrib(default=None)
    trajectory_url: Optional[str] = attrib(default=None)
    dump_data: Optional[Dict[str, Any]] = attrib(default=None)
    error: Optional[str] = attrib(default=None)

    def __str__(self) -> str:
        """Return the clean output text.

        This is critical for DualInferencer compatibility — it calls
        ``str(result)`` to extract text for the consensus loop.
        """
        return self.output or self.raw_output

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TerminalInferencerResponse":
        """Create from a parse_output() dict.

        Unknown keys are silently ignored so subclasses of
        TerminalSessionInferencerBase can add custom fields to their
        parse_output() dicts without breaking this constructor.

        Args:
            d: Dictionary returned by ``parse_output()``.

        Returns:
            A TerminalInferencerResponse instance.
        """
        return cls(
            output=d.get("output", ""),
            raw_output=d.get("raw_output", ""),
            stderr=d.get("stderr", ""),
            return_code=d.get("return_code", 0),
            success=d.get("success", True),
            session_id=d.get("session_id"),
            trajectory_url=d.get("trajectory_url"),
            dump_data=d.get("dump_data"),
            error=d.get("error"),
        )
