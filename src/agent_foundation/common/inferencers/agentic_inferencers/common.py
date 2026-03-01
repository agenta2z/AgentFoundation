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


def severity_at_most(severity: Severity, threshold: Severity) -> bool:
    """Check if severity is at or below the threshold level.

    Args:
        severity: The severity to check.
        threshold: The maximum acceptable severity.

    Returns:
        True if severity <= threshold in the ordering
        (NONE < COSMETIC < MINOR < MAJOR < CRITICAL).
    """
    return _SEVERITY_ORDER[severity] <= _SEVERITY_ORDER[threshold]


@attrs
class ConsensusConfig:
    """Configuration for consensus loop in DualInferencer.

    Attributes:
        max_iterations: Maximum propose-review-fix cycles per attempt.
        max_consensus_attempts: Maximum fresh-start attempts if consensus fails.
        consensus_threshold: Maximum acceptable severity level for consensus.
        enable_counter_feedback: Whether fixer can reject reviewer issues.
    """

    max_iterations: int = attrib(default=5)
    max_consensus_attempts: int = attrib(default=1)
    consensus_threshold: Severity = attrib(default=Severity.COSMETIC)
    enable_counter_feedback: bool = attrib(default=True)


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
