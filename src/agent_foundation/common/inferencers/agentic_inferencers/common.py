from enum import StrEnum
from typing import Any, Optional, Sequence, Union, Callable, Type

from attr import attrs, attrib

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


DEFAULT_TAG_INFERENCER_RESPONSE = 'InferencerResponse'
DEFAULT_TAG_BASE_RESPONSE = 'BaseResponse'
DEFAULT_TAG_REFLECTION_RESPONSES = 'ReflectionResponses'
DEFAULT_TAG_REFLECTION_RESPONSE = 'ReflectionResponse'
DEFAULT_TAG_REFLECTION_STYLE = 'ReflectionStyle'


@attrs
class InferencerResponse:
    base_response: Any = attrib()
    reflection_response: Optional[Union[Sequence, Any]] = attrib(default=None)
    reflection_style: ReflectionStyles = attrib(default=ReflectionStyles.NoReflection)
    response_selector: Union[Callable[['InferencerResponse'], Any], ResponseSelectors] = attrib(
        default=ResponseSelectors.LastReflection
    )

    def select_response(
            self,
            response_selector: Union[Callable[['InferencerResponse'], Any], ResponseSelectors] = None,
            response_types: Type = None
    ):
        if response_selector is None:
            response_selector = self.response_selector

        if isinstance(response_selector, ResponseSelectors):
            if response_selector == ResponseSelectors.BaseResponse:
                return self.base_response
            elif response_selector == ResponseSelectors.FirstReflection:
                if isinstance(self.reflection_response, (response_types or DEFAULT_RESPONSE_TYPES)):
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
                if isinstance(self.reflection_response, (response_types or DEFAULT_RESPONSE_TYPES)):
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

    def to_string(self,
                  root_tag: str = DEFAULT_TAG_INFERENCER_RESPONSE,
                  base_response_tag: str = DEFAULT_TAG_BASE_RESPONSE,
                  reflection_responses_tag: str = DEFAULT_TAG_REFLECTION_RESPONSES,
                  reflection_response_tag: str = DEFAULT_TAG_REFLECTION_RESPONSE,
                  reflection_style_tag: str = DEFAULT_TAG_REFLECTION_STYLE,
                  include_reflection_style: bool = False,
                  unpack_single_reflection: bool = True,
                  indent: str = '  ') -> str:
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
            mapping = {
                base_response_tag: self.base_response
            }

            if isinstance(self.reflection_response, DEFAULT_RESPONSE_TYPES):
                mapping[reflection_response_tag] = self.reflection_response
            elif not isinstance(self.reflection_response, Sequence):
                raise ValueError(f"'reflection' must be an Sequence; got {self.reflection_response}")
            elif unpack_single_reflection and len(self.reflection_response) == 1:
                mapping[reflection_response_tag] = self.reflection_response[0]
            else:
                mapping[reflection_responses_tag] = self.reflection_response

            if include_reflection_style:
                mapping[reflection_style_tag] = self.reflection_style

            return mapping_to_xml(
                mapping,
                root_tag=root_tag,
                item_tag={
                    reflection_responses_tag: reflection_response_tag
                },
                indent=indent,
                unescape=True
            )

    def __str__(self) -> str:
        """Default string representation using default tags."""
        return self.to_string()


DEFAULT_RESPONSE_TYPES = Union[str, InferencerResponse, InputAndResponse]
