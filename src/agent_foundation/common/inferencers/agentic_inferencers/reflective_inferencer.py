import logging
from functools import partial
from typing import Callable, Mapping, Union, Sequence, Any

from attr import attrs, attrib

from science_modeling_tools.common.inferencers.agentic_inferencers.constants import (
    DEFAULT_PLACEHOLDER_INFERENCE_PROMPT,
    DEFAULT_PLACEHOLDER_INFERENCE_RESPONSE,
    DEFAULT_SELF_REFLECTION_PROMPT_TEMPLATE
)
from science_modeling_tools.common.inferencers.inferencer_base import InferencerBase
from science_modeling_tools.common.inferencers.agentic_inferencers.common import InferencerResponse, ReflectionStyles, \
    ResponseSelectors
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.input_and_response import InputAndResponse
from rich_python_utils.string_utils import join_
from rich_python_utils.string_utils.formatting.template_manager import TemplateManager
from rich_python_utils.string_utils.xml_helpers import unescape_xml


@attrs
class ReflectiveInferencer(InferencerBase):
    base_inferencer: InferencerBase = attrib(default=None)
    base_response_concat: Callable = attrib(default=None)
    reflection_prompt_template: Union[str, Mapping[ReflectionStyles, str]] = attrib(
        default=DEFAULT_SELF_REFLECTION_PROMPT_TEMPLATE
    )
    reflection_prompt_formatter: Callable = attrib(default=None)
    reflection_inferencer: InferencerBase = attrib(default=None)

    num_reflections: int = attrib(default=1)
    reflection_style: ReflectionStyles = attrib(default=ReflectionStyles.Separate)
    unpack_single_reflection: bool = attrib(default=True)
    unpack_single_response: bool = attrib(default=True)
    response_selector: Union[Callable[['InferencerResponse'], Any], ResponseSelectors] = attrib(default=None)

    reflection_prompt_placeholder_inferencer_input: str = attrib(default=DEFAULT_PLACEHOLDER_INFERENCE_PROMPT)
    reflection_prompt_placeholder_inferencer_response: str = attrib(default=DEFAULT_PLACEHOLDER_INFERENCE_RESPONSE)

    def __attrs_post_init__(self):
        super(ReflectiveInferencer, self).__attrs_post_init__()
        if (not self.response_types) or self.response_types == (str,):
            self.response_types = (str, InferencerResponse)

        if not isinstance(self.reflection_prompt_formatter, TemplateManager):
            self.reflection_prompt_formatter = TemplateManager(
                templates=self.reflection_prompt_template,
                template_formatter=self.reflection_prompt_formatter
            )

        # Set parent debuggable for nested inferencer components
        if self.base_inferencer is not None and isinstance(self.base_inferencer, Debuggable):
            self.base_inferencer.set_parent_debuggable(self)
        # Only set parent for reflection_inferencer if it's a different object than base_inferencer
        if (
                self.reflection_inferencer is not None and
                isinstance(self.reflection_inferencer, Debuggable) and
                self.reflection_inferencer is not self.base_inferencer
        ):
            self.reflection_inferencer.set_parent_debuggable(self)

    def _concat_base_responses(self, base_responses):
        if self.base_response_concat:
            return self.base_response_concat(base_responses)
        else:
            return join_(base_responses, sep='\n\n')

    def _process_reflection_input(self, inference_input, reflection_input, inference_config):
        return self.reflection_prompt_formatter(
            feed={
                self.reflection_prompt_placeholder_inferencer_input: inference_input,
                self.reflection_prompt_placeholder_inferencer_response: reflection_input
            },
            post_process=partial(unescape_xml, unescape_for_html=True),
            **inference_config
        )

    def _infer(self, inference_input: str, inference_config: Any = None, **_inference_args):
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, Mapping):
            raise ValueError("'inference_input' must be a mapping")

        if self.reflection_style == ReflectionStyles.IntegrateAll:
            reflection_input = self._concat_base_responses(
                *(self.base_inferencer.iter_infer(inference_input, **_inference_args))
            )
            processed_reflection_input = self._process_reflection_input(
                inference_input=inference_input,
                reflection_input=reflection_input,
                inference_config=inference_config
            )
            reflection_response = self.reflection_inferencer(processed_reflection_input)
            inference_response = InferencerResponse(
                base_response=inference_input,
                reflection_response=InputAndResponse(input=processed_reflection_input, response=reflection_response),
                reflection_style=self.reflection_style,
                response_selector=self.response_selector
            )
            all_inference_responses = [inference_response]
        else:
            all_inference_responses = []
            for response in self.base_inferencer.iter_infer(inference_input, **_inference_args):
                if self.reflection_style == ReflectionStyles.NoReflection:
                    inference_response = InferencerResponse(
                        base_response=response,
                        reflection_response=None,
                        reflection_style=self.reflection_style,
                        response_selector=self.response_selector
                    )
                else:
                    all_reflection_response = []
                    previous_reflection_response = None

                    for _ in range(self.num_reflections):
                        if self.reflection_style == ReflectionStyles.Separate:
                            # Each reflection considers only the original response
                            reflection_input = response
                        elif self.reflection_style == ReflectionStyles.Sequential:  # Sequential style
                            # Each reflection builds upon the previous reflection
                            reflection_input = (
                                previous_reflection_response
                                if previous_reflection_response is not None
                                else response
                            )
                        else:
                            raise ValueError(
                                f"The specified 'self_reflection_style' ({self.reflection_style}) is not supported.)"
                            )

                        processed_reflection_input = self._process_reflection_input(
                            inference_input=inference_input,
                            reflection_input=reflection_input,
                            inference_config=inference_config
                        )
                        reflection_response = self.reflection_inferencer(processed_reflection_input)
                        self.log_debug(processed_reflection_input, 'ReflectionPrompt')
                        self.log_debug(reflection_response, 'ReflectionResponse')

                        all_reflection_response.append(
                            InputAndResponse(input=processed_reflection_input, response=reflection_response)
                        )
                        previous_reflection_response = reflection_response

                    inference_response = InferencerResponse(
                        base_response=response,
                        reflection_response=(
                            all_reflection_response[0] if (self.num_reflections == 1 and self.unpack_single_reflection)
                            else all_reflection_response
                        ),
                        reflection_style=self.reflection_style,
                        response_selector=self.response_selector
                    )

                all_inference_responses.append(inference_response)

        if self.unpack_single_response and len(all_inference_responses) == 1:
            all_inference_responses = all_inference_responses[0]
        return all_inference_responses
