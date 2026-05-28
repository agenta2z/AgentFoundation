from enum import StrEnum
from typing import Any, Callable, List, Optional, Sequence, Type, Union

from attr import attrib, attrs
from rich_python_utils.common_objects.input_and_response import InputAndResponse
from rich_python_utils.string_utils.xml_helpers import mapping_to_xml


class ReflectionStyles(StrEnum):
    """
    Enum representing styles of reflections.

    Attributes:
        NoReflection: No reflections are performed.
        Separate: Reflections are done separately for each response.
        Sequential: Each reflection builds upon previous reflections.
        IntegrateAll: Reflections are integrated across all responses.
    """

    NoReflection = "no_reflection"  # No reflections are performed
    Separate = "separate"  # Reflections are done separately for each response
    Sequential = "sequential"  # Each reflection builds upon previous reflections
    IntegrateAll = "integrate_all"  # Reflections are integrated across all responses.


class ResponseSelectors(StrEnum):
    BaseResponse = "base_response"
    FirstReflection = "first_reflection"
    LastReflection = "last_reflection"


DEFAULT_TAG_INFERENCER_RESPONSE = "InferencerResponse"
DEFAULT_TAG_BASE_RESPONSE = "BaseResponse"
DEFAULT_TAG_REFLECTION_RESPONSES = "ReflectionResponses"
DEFAULT_TAG_REFLECTION_RESPONSE = "ReflectionResponse"
DEFAULT_TAG_REFLECTION_STYLE = "ReflectionStyle"


@attrs
class InferencerResponse:
    base_response: Any = attrib()
    reflection_response: Optional[Union[Sequence, Any]] = attrib(default=None)
    reflection_style: ReflectionStyles = attrib(default=ReflectionStyles.NoReflection)
    response_selector: Union[
        Callable[["InferencerResponse"], Any], ResponseSelectors
    ] = attrib(default=ResponseSelectors.LastReflection)

    def select_response(
        self,
        response_selector: Union[
            Callable[["InferencerResponse"], Any], ResponseSelectors
        ] = None,
        response_types: Type = None,
    ):
        if response_selector is None:
            response_selector = self.response_selector

        if isinstance(response_selector, ResponseSelectors):
            if response_selector == ResponseSelectors.BaseResponse:
                return self.base_response
            elif response_selector == ResponseSelectors.FirstReflection:
                if isinstance(
                    self.reflection_response, (response_types or DEFAULT_RESPONSE_TYPES)
                ):
                    return self.reflection_response
                elif isinstance(self.reflection_response, Sequence):
                    return self.reflection_response[0]
                else:
                    raise ValueError(
                        f"Expect 'reflection_response' be a Sequence, "
                        f"or a single string response or InferencerResponse object; "
                        f"got  {self.reflection_response}"
                    )
            elif response_selector == ResponseSelectors.LastReflection:
                if isinstance(
                    self.reflection_response, (response_types or DEFAULT_RESPONSE_TYPES)
                ):
                    return self.reflection_response
                elif isinstance(self.reflection_response, Sequence):
                    return self.reflection_response[-1]
                else:
                    raise ValueError(
                        f"Expect 'reflection_response' be a Sequence, "
                        f"or a single string response or InferencerResponse object; "
                        f"got  {self.reflection_response}"
                    )
            else:
                raise ValueError(f"Unexpected response selector: {response_selector}")
        elif callable(response_selector):
            return response_selector(self)
        else:
            raise ValueError(
                f"'response_selector' must be either a callable or a ResponseSelectors enum; got {response_selector}"
            )

    def to_string(
        self,
        root_tag: str = DEFAULT_TAG_INFERENCER_RESPONSE,
        base_response_tag: str = DEFAULT_TAG_BASE_RESPONSE,
        reflection_responses_tag: str = DEFAULT_TAG_REFLECTION_RESPONSES,
        reflection_response_tag: str = DEFAULT_TAG_REFLECTION_RESPONSE,
        reflection_style_tag: str = DEFAULT_TAG_REFLECTION_STYLE,
        include_reflection_style: bool = False,
        unpack_single_reflection: bool = True,
        indent: str = "  ",
    ) -> str:
        """
        Convert the `InferencerResponse` object to an XML string representation with configurable tags.

        This method formats the response as an XML document, allowing customization of element tags
        and inclusion of optional metadata such as the reflection style. Reflections can be presented
        as individual elements or grouped, depending on the configuration.

        Args:
            root_tag (str): The tag name for the root XML element. Defaults to 'InferencerResponse'.
            base_response_tag (str): The tag name for the base response element. Defaults to 'BaseResponse'.
            reflection_responses_tag (str): The tag name for the container of reflection responses.
                Defaults to 'ReflectionResponses'.
            reflection_response_tag (str): The tag name for each individual reflection response.
                Defaults to 'ReflectionResponse'.
            reflection_style_tag (str): The tag name for the reflection style metadata.
                Defaults to 'ReflectionStyle'.
            include_reflection_style (bool): If True, include the reflection style as a separate element. Defaults to False.
            unpack_single_reflection (bool): If True and the reflection contains only one item, it will
                be presented as a single element instead of a nested structure. Defaults to True.
            indent (str): The string to use for indentation in the XML output. Defaults to a single space ('  ').

        Returns:
            str: Formatted XML string representation of the response.

        Example:
            >>> response = InferencerResponse(
            ...     base_response="Output",
            ...     reflection_response=["Reflection1", "Reflection2"],
            ...     reflection_style=ReflectionStyles.IntegrateAll,
            ... )
            >>> print(response.to_string(include_reflection_style=True))
            <InferencerResponse>
              <BaseResponse>Output</BaseResponse>
              <ReflectionResponses>
                <ReflectionResponse>Reflection1</ReflectionResponse>
                <ReflectionResponse>Reflection2</ReflectionResponse>
              </ReflectionResponses>
              <ReflectionStyle>integrate_all</ReflectionStyle>
            </InferencerResponse>

            >>> response = InferencerResponse(
            ...     base_response="Output",
            ...     reflection_response="Reflection",
            ...     reflection_style=ReflectionStyles.IntegrateAll,
            ... )
            >>> print(response.to_string())
            <InferencerResponse>
              <BaseResponse>Output</BaseResponse>
              <ReflectionResponse>Reflection</ReflectionResponse>
            </InferencerResponse>

            >>> response = InferencerResponse(
            ...     base_response="Output",
            ...     reflection_response=["Reflection"],
            ...     reflection_style=ReflectionStyles.IntegrateAll,
            ... )
            >>> print(response.to_string())
            <InferencerResponse>
              <BaseResponse>Output</BaseResponse>
              <ReflectionResponse>Reflection</ReflectionResponse>
            </InferencerResponse>
        """

        if self.response_selector is not None:
            return str(self.select_response())
        else:
            mapping = {base_response_tag: self.base_response}

            if isinstance(self.reflection_response, DEFAULT_RESPONSE_TYPES):
                mapping[reflection_response_tag] = self.reflection_response
            elif not isinstance(self.reflection_response, Sequence):
                raise ValueError(
                    f"'reflection' must be an Sequence; got {self.reflection_response}"
                )
            elif unpack_single_reflection and len(self.reflection_response) == 1:
                mapping[reflection_response_tag] = self.reflection_response[0]
            else:
                mapping[reflection_responses_tag] = self.reflection_response

            if include_reflection_style:
                mapping[reflection_style_tag] = self.reflection_style

            return mapping_to_xml(
                mapping,
                root_tag=root_tag,
                item_tag={reflection_responses_tag: reflection_response_tag},
                indent=indent,
                unescape=True,
            )

    def __str__(self) -> str:
        """Default string representation using default tags."""
        return self.to_string()


class Severity(StrEnum):
    """Severity levels for review issues in consensus workflows.

    Ordered from least to most severe. Used by DualInferencer to determine
    consensus threshold — issues at or below the threshold level are acceptable.
    """

    NONE = "NONE"
    COSMETIC = "COSMETIC"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


# Ordering for severity comparison (lower index = less severe)
_SEVERITY_ORDER = {s: i for i, s in enumerate(Severity)}


def severity_at_most(severity, threshold, severity_levels=None) -> bool:
    """Check if severity is at or below the threshold level.

    Args:
        severity: The severity to check.
        threshold: The maximum acceptable severity.
        severity_levels: Optional ordered sequence of severity values
            (least to most severe). When provided, ordering is derived
            from this sequence instead of the default Severity enum.

    Returns:
        True if severity <= threshold in the ordering.
        False if either value is not found in the ordering.
    """
    if severity_levels is not None:
        order = {s: i for i, s in enumerate(severity_levels)}
    else:
        order = _SEVERITY_ORDER
    sev_idx = order.get(severity)
    thr_idx = order.get(threshold)
    if sev_idx is None or thr_idx is None:
        return False
    return sev_idx <= thr_idx


_DEFAULT_SEVERITY_LEVELS = ('NONE', 'COSMETIC', 'MINOR', 'MAJOR', 'CRITICAL')


@attrs
class ConsensusConfig:
    """Configuration for consensus loop in DualInferencer.

    Attributes:
        max_iterations: Maximum propose-review-fix cycles per attempt.
        max_consensus_attempts: Maximum fresh-start attempts if consensus fails.
        severity_levels: Ordered severity values from least to most severe.
            Allows custom severity scales (e.g. numeric ``(0, 1, 2, 3)`` or
            domain-specific labels). Defaults to the standard 5-level scale.
        consensus_threshold: Maximum acceptable severity level for consensus.
            Must be a value present in ``severity_levels``.
        enable_counter_feedback: Whether fixer can reject reviewer issues.
    """

    max_iterations: int = attrib(default=5)
    max_consensus_attempts: int = attrib(default=1)
    severity_levels: tuple = attrib(default=_DEFAULT_SEVERITY_LEVELS)
    consensus_threshold = attrib(default='COSMETIC')
    enable_counter_feedback: bool = attrib(default=True)

    def __attrs_post_init__(self):
        if self.consensus_threshold not in self.severity_levels:
            raise ValueError(
                f"consensus_threshold={self.consensus_threshold!r} is not in "
                f"severity_levels={self.severity_levels!r}"
            )

    def acceptable_severities(self) -> tuple:
        """Severity levels at or below threshold, e.g. ``('NONE', 'COSMETIC')``."""
        idx = self.severity_levels.index(self.consensus_threshold)
        return self.severity_levels[: idx + 1]

    def blocking_severities(self) -> tuple:
        """Severity levels above threshold, e.g. ``('MINOR', 'MAJOR', 'CRITICAL')``."""
        idx = self.severity_levels.index(self.consensus_threshold)
        return self.severity_levels[idx + 1 :]

    def approve_hint(self) -> str:
        """Short hint for JSON schema ``"approve"`` field.

        Picks whichever list (acceptable vs blocking) is shorter::

            COSMETIC → 'true if only observing NONE/COSMETIC issues'
            MINOR    → 'true if no MAJOR/CRITICAL issues'
            MAJOR    → 'true if no CRITICAL issues'
        """
        acceptable = self.acceptable_severities()
        blocking = self.blocking_severities()
        if not blocking:
            return "true (all severity levels are acceptable)"
        if len(acceptable) <= len(blocking):
            return f"true if only observing {'/'.join(acceptable)} issues"
        return f"true if no {'/'.join(blocking)} issues"

    def approval_guidance(self) -> str:
        """Prose guidance for when to set ``approve: true``."""
        acceptable = self.acceptable_severities()
        blocking = self.blocking_severities()
        if not blocking:
            return "there are no severity restrictions — all levels are acceptable."
        return (
            f"all issues are at most {self.consensus_threshold} severity "
            f"(i.e., {' or '.join(acceptable)}). "
            f"Any issue rated {', '.join(blocking)} means you must NOT approve."
        )


@attrs
class ConsensusIterationRecord:
    """Record of a single propose-review-fix iteration.

    Attributes:
        iteration: 1-based iteration number within the attempt.
        base_output: The proposal text for this iteration.
        review_input: The rendered review prompt sent to the reviewer.
        review_output: Raw review response from the reviewer.
        review_feedback: Parsed review feedback dict (approved, severity, issues).
        counter_feedback: Parsed counter-feedback dict from the fixer, if any.
        consensus_reached: Whether consensus was reached in this iteration.
    """

    iteration: int = attrib()
    base_output: str = attrib()
    review_input: str = attrib()
    review_output: str = attrib()
    review_feedback: Any = attrib(default=None)
    counter_feedback: Any = attrib(default=None)
    consensus_reached: bool = attrib(default=False)


@attrs
class ConsensusAttemptRecord:
    """Record of a full consensus attempt (one or more iterations).

    Attributes:
        attempt: 1-based attempt number.
        iterations: List of ConsensusIterationRecord for this attempt.
        consensus_reached: Whether consensus was achieved in this attempt.
        final_output: The final proposal text from this attempt.
        final_feedback: The final parsed review feedback from this attempt.
    """

    attempt: int = attrib()
    iterations: List[ConsensusIterationRecord] = attrib(factory=list)
    consensus_reached: bool = attrib(default=False)
    final_output: str = attrib(default="")
    final_feedback: Any = attrib(default=None)


@attrs
class DualInferencerResponse(InferencerResponse):
    """Response from DualInferencer with consensus history.

    Extends InferencerResponse with consensus-specific metadata.
    base_response holds the final proposal text.
    reflection_response holds the final review InputAndResponse.

    Attributes:
        consensus_history: List of ConsensusAttemptRecord documenting the loop.
        total_iterations: Total propose-review-fix cycles across all attempts.
        consensus_achieved: Whether consensus was ultimately reached.
        phase: Label for the consensus phase (e.g., "planning", "execution").
    """

    consensus_history: List[ConsensusAttemptRecord] = attrib(factory=list)
    total_iterations: int = attrib(default=0)
    consensus_achieved: bool = attrib(default=False)
    phase: str = attrib(default="")


DEFAULT_RESPONSE_TYPES = Union[str, InferencerResponse, InputAndResponse]


class InferencerExecutionError(RuntimeError):
    """Raised when a terminal inferencer returns a structured failure result."""

    def __init__(
        self,
        tool: str = "<terminal-inferencer>",
        return_code: Optional[int] = None,
        stderr: str = "",
        error: str = "",
    ) -> None:
        self.tool = tool
        self.return_code = return_code
        self.stderr = stderr
        self.error = error
        msg = f"{tool} failed (return_code={return_code}): {error or stderr or '<no error message>'}"
        super().__init__(msg)


class MaxIterationsExhaustedError(InferencerExecutionError):
    """Raised when a terminal inferencer's per-session iteration counter is
    exhausted. Subclass of ``InferencerExecutionError`` for backward compatibility.
    """

    def __init__(
        self,
        *args,
        max_iterations: Optional[int] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.session_id = session_id


def extract_response_text(result):
    """Extract text from various inferencer result types.

    Type dispatch (order matters — most specific first):
    - DualInferencerResponse → str(result.base_response)
    - InferencerResponse → str(result.select_response())
    - AgenticResult → result.text
    - Terminal-inferencer result dict (has ``success`` key) → ``output`` on
      success; raise ``InferencerExecutionError`` on failure.
    - Plain dict without ``success`` key → returns the first matching string
      value among ``output``/``raw_output``/``content`` keys; raises
      ``InferencerExecutionError`` for unrecognized dict shapes.
    - Other → str(result)

    IMPORTANT: DualInferencerResponse check must come BEFORE InferencerResponse
    because DualInferencerResponse inherits from InferencerResponse.
    """
    if isinstance(result, DualInferencerResponse):
        return str(result.base_response)
    if isinstance(result, InferencerResponse):
        return str(result.select_response())
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
            AgenticResult,
        )
        if isinstance(result, AgenticResult):
            return result.text
    except ImportError:
        pass
    if isinstance(result, dict) and "success" in result:
        if not result.get("success", False):
            error_text = result.get("error", "") or ""
            if (
                "MAX_ITERATIONS" in error_text
                and "Max iterations of" in error_text
                and "reached" in error_text
            ):
                import re as _re

                _m = _re.search(r"Max iterations of (\d+) reached", error_text)
                raise MaxIterationsExhaustedError(
                    tool=result.get("tool") or "<terminal-inferencer>",
                    return_code=result.get("return_code"),
                    stderr=result.get("stderr", "") or "",
                    error=error_text,
                    max_iterations=int(_m.group(1)) if _m else None,
                    session_id=result.get("session_id"),
                )
            raise InferencerExecutionError(
                tool=result.get("tool") or "<terminal-inferencer>",
                return_code=result.get("return_code"),
                stderr=result.get("stderr", "") or "",
                error=error_text,
            )
        return result.get("output", "") or ""
    if isinstance(result, dict):
        for key in ("output", "raw_output", "content"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        raise InferencerExecutionError(
            tool="<unknown_dict_shape>",
            error=(
                "extract_response_text received a dict without a recognized "
                f"shape; keys={sorted(result.keys())[:8]!r}"
            ),
        )
    return str(result)
