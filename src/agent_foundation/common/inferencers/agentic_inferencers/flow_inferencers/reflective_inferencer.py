import logging
from functools import partial
from typing import Any, Callable, Mapping, Sequence, Union

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    InferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
)
from agent_foundation.common.inferencers.agentic_inferencers.constants import (
    DEFAULT_PLACEHOLDER_INFERENCE_PROMPT,
    DEFAULT_PLACEHOLDER_INFERENCE_RESPONSE,
    DEFAULT_SELF_REFLECTION_PROMPT_TEMPLATE,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.input_and_response import InputAndResponse
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.io_utils.artifact import artifact_type
from rich_python_utils.string_utils import join_
from rich_python_utils.string_utils.formatting.template_manager import TemplateManager
from rich_python_utils.string_utils.xml_helpers import unescape_xml


@artifact_type(Workflow, type="json", group="workflows")
@attrs
class ReflectiveInferencer(LinearWorkflowInferencer):
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
    response_selector: Union[
        Callable[["InferencerResponse"], Any], ResponseSelectors
    ] = attrib(default=None)

    reflection_prompt_placeholder_inferencer_input: str = attrib(
        default=DEFAULT_PLACEHOLDER_INFERENCE_PROMPT
    )
    reflection_prompt_placeholder_inferencer_response: str = attrib(
        default=DEFAULT_PLACEHOLDER_INFERENCE_RESPONSE
    )

    @property
    def supports_prompt_rendering(self) -> bool:
        return self.reflection_prompt_formatter is not None

    def __attrs_post_init__(self):
        # --- Domain-specific init FIRST (before calling super) ---
        if (not self.response_types) or self.response_types == (str,):
            self.response_types = (str, InferencerResponse)

        if not isinstance(self.reflection_prompt_formatter, TemplateManager):
            self.reflection_prompt_formatter = TemplateManager(
                templates=self.reflection_prompt_template,
                template_formatter=self.reflection_prompt_formatter,
            )

        # --- Build step_configs based on reflection_style ---
        if self.reflection_style == ReflectionStyles.Sequential:
            self.step_configs = [
                WorkflowStepConfig(
                    name="base",
                    inferencer=self.base_inferencer,
                    output_state_key="base_response",
                    enable_result_save=False,
                ),
                WorkflowStepConfig(
                    name="reflect",
                    inferencer=self.reflection_inferencer,
                    input_builder=self._build_reflection_input_sequential,
                    output_state_key="reflection_output",
                    state_updater=self._append_reflection_record,
                    loop_back_to="reflect",
                    loop_condition=lambda s, r: True,  # REQUIRED — None means no loop
                    max_loop_iterations=self.num_reflections - 1,
                    enable_result_save=False,
                ),
            ]
        elif self.reflection_style == ReflectionStyles.Separate:
            self.step_configs = [
                WorkflowStepConfig(
                    name="base",
                    inferencer=self.base_inferencer,
                    output_state_key="base_response",
                    enable_result_save=False,
                ),
                WorkflowStepConfig(
                    name="reflect",
                    inferencer=self.reflection_inferencer,
                    input_builder=self._build_reflection_input_separate,
                    output_state_key="reflection_output",
                    state_updater=self._append_reflection_record,
                    loop_back_to="reflect",
                    loop_condition=lambda s, r: True,
                    max_loop_iterations=self.num_reflections - 1,
                    enable_result_save=False,
                ),
            ]
        elif self.reflection_style == ReflectionStyles.IntegrateAll:
            self.step_configs = [
                WorkflowStepConfig(
                    name="collect",
                    step_fn=self._collect_all_responses,
                    output_state_key="collected_responses",
                    enable_result_save=False,
                ),
                WorkflowStepConfig(
                    name="reflect",
                    inferencer=self.reflection_inferencer,
                    input_builder=self._build_reflection_input_integrate,
                    output_state_key="reflection_output",
                    enable_result_save=False,
                ),
            ]
        elif self.reflection_style == ReflectionStyles.NoReflection:
            self.step_configs = [
                WorkflowStepConfig(
                    name="base",
                    inferencer=self.base_inferencer,
                    output_state_key="base_response",
                    enable_result_save=False,
                ),
                WorkflowStepConfig(
                    name="reflect",
                    inferencer=self.reflection_inferencer,
                    enabled=False,
                ),
            ]

        # Set response_builder for constructing InferencerResponse from state
        self.response_builder = self._build_reflective_response

        # --- Call super().__attrs_post_init__() LAST ---
        super(ReflectiveInferencer, self).__attrs_post_init__()

        # Set parent debuggable for nested inferencer components
        # (super already handles step_configs inferencers, but we need to
        # handle the case where base_inferencer and reflection_inferencer
        # might be the same object — super deduplicates by id())
        if self.base_inferencer is not None and isinstance(
            self.base_inferencer, Debuggable
        ):
            self.base_inferencer.set_parent_debuggable(self)
        if (
            self.reflection_inferencer is not None
            and isinstance(self.reflection_inferencer, Debuggable)
            and self.reflection_inferencer is not self.base_inferencer
        ):
            self.reflection_inferencer.set_parent_debuggable(self)

    # ------------------------------------------------------------------
    # Input builder helpers for step_configs
    # ------------------------------------------------------------------

    def _build_reflection_input_sequential(self, state):
        """Build reflection input for Sequential mode — uses previous reflection or base response."""
        reflections = state.get("all_reflections", [])
        if reflections:
            reflection_input = reflections[-1].response
        else:
            reflection_input = state["base_response"]
        return self._process_reflection_input(
            inference_input=state["original_input"],
            reflection_input=reflection_input,
            inference_config=state.get("_inference_config", {}),
        )

    def _build_reflection_input_separate(self, state):
        """Build reflection input for Separate mode — always uses original base response."""
        return self._process_reflection_input(
            inference_input=state["original_input"],
            reflection_input=state["base_response"],
            inference_config=state.get("_inference_config", {}),
        )

    def _build_reflection_input_integrate(self, state):
        """Build reflection input for IntegrateAll mode — uses collected responses."""
        return self._process_reflection_input(
            inference_input=state["original_input"],
            reflection_input=state["collected_responses"],
            inference_config=state.get("_inference_config", {}),
        )

    def _append_reflection_record(self, state, result):
        """Append a reflection InputAndResponse to the accumulator list."""
        if "all_reflections" not in state:
            state["all_reflections"] = []
        # Get the input that was used for this reflection
        reflections = state.get("all_reflections", [])
        if self.reflection_style == ReflectionStyles.Sequential:
            if reflections:
                reflection_input = reflections[-1].response
            else:
                reflection_input = state["base_response"]
        else:
            reflection_input = state["base_response"]

        processed_input = self._process_reflection_input(
            inference_input=state["original_input"],
            reflection_input=reflection_input,
            inference_config=state.get("_inference_config", {}),
        )
        state["all_reflections"].append(
            InputAndResponse(input=processed_input, response=result)
        )

    def _collect_all_responses(self, input_val, state):
        """Collect all base responses for IntegrateAll mode."""
        return self._concat_base_responses(
            *(self.base_inferencer.iter_infer(state["original_input"]))
        )

    def _build_reflective_response(self, state):
        """Build InferencerResponse from workflow state."""
        reflections = state.get("all_reflections", [])
        base_response = state.get("base_response", state.get("collected_responses", ""))

        if self.reflection_style == ReflectionStyles.NoReflection:
            return InferencerResponse(
                base_response=base_response,
                reflection_response=None,
                reflection_style=self.reflection_style,
                response_selector=self.response_selector,
            )
        elif self.reflection_style == ReflectionStyles.IntegrateAll:
            reflection_output = state.get("reflection_output")
            processed_input = self._build_reflection_input_integrate(state)
            return InferencerResponse(
                base_response=state["original_input"],
                reflection_response=InputAndResponse(
                    input=processed_input, response=reflection_output
                ),
                reflection_style=self.reflection_style,
                response_selector=self.response_selector,
            )
        else:
            # Sequential or Separate
            reflection_response = reflections
            if self.num_reflections == 1 and self.unpack_single_reflection:
                reflection_response = reflections[0] if reflections else None
            return InferencerResponse(
                base_response=base_response,
                reflection_response=reflection_response,
                reflection_style=self.reflection_style,
                response_selector=self.response_selector,
            )

    # ------------------------------------------------------------------
    # Existing domain methods (preserved)
    # ------------------------------------------------------------------

    def _concat_base_responses(self, base_responses):
        if self.base_response_concat:
            return self.base_response_concat(base_responses)
        else:
            return join_(base_responses, sep="\n\n")

    def _process_reflection_input(
        self, inference_input, reflection_input, inference_config
    ):
        return self.reflection_prompt_formatter(
            feed={
                self.reflection_prompt_placeholder_inferencer_input: inference_input,
                self.reflection_prompt_placeholder_inferencer_response: reflection_input,
            },
            post_process=partial(unescape_xml, unescape_for_html=True),
            **inference_config,
        )

    def _infer(
        self, inference_input: str, inference_config: Any = None, **_inference_args
    ):
        """Preserved _infer for backward compatibility with iter_infer multi-response.

        Keeps the existing manual loop implementation that handles all reflection
        styles correctly. The step_configs enable ainfer to work via LWI's workflow
        engine, but _infer continues to use the proven manual implementation.
        """
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, Mapping):
            raise ValueError("'inference_input' must be a mapping")

        # Store inference_config for input_builder helpers
        self._inference_config = inference_config or {}

        if self.reflection_style == ReflectionStyles.IntegrateAll:
            reflection_input = self._concat_base_responses(
                *(self.base_inferencer.iter_infer(inference_input, **_inference_args))
            )
            processed_reflection_input = self._process_reflection_input(
                inference_input=inference_input,
                reflection_input=reflection_input,
                inference_config=inference_config,
            )
            reflection_response = self.reflection_inferencer(processed_reflection_input)
            inference_response = InferencerResponse(
                base_response=inference_input,
                reflection_response=InputAndResponse(
                    input=processed_reflection_input, response=reflection_response
                ),
                reflection_style=self.reflection_style,
                response_selector=self.response_selector,
            )
            all_inference_responses = [inference_response]
        else:
            all_inference_responses = []
            for response in self.base_inferencer.iter_infer(
                inference_input, **_inference_args
            ):
                if self.reflection_style == ReflectionStyles.NoReflection:
                    inference_response = InferencerResponse(
                        base_response=response,
                        reflection_response=None,
                        reflection_style=self.reflection_style,
                        response_selector=self.response_selector,
                    )
                else:
                    all_reflection_response = []
                    previous_reflection_response = None

                    for _ in range(self.num_reflections):
                        if self.reflection_style == ReflectionStyles.Separate:
                            # Each reflection considers only the original response
                            reflection_input = response
                        elif (
                            self.reflection_style == ReflectionStyles.Sequential
                        ):  # Sequential style
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
                            inference_config=inference_config,
                        )
                        reflection_response = self.reflection_inferencer(
                            processed_reflection_input
                        )
                        self.log_debug(processed_reflection_input, "ReflectionPrompt")
                        self.log_debug(reflection_response, "ReflectionResponse")

                        all_reflection_response.append(
                            InputAndResponse(
                                input=processed_reflection_input,
                                response=reflection_response,
                            )
                        )
                        previous_reflection_response = reflection_response

                    inference_response = InferencerResponse(
                        base_response=response,
                        reflection_response=(
                            all_reflection_response[0]
                            if (
                                self.num_reflections == 1
                                and self.unpack_single_reflection
                            )
                            else all_reflection_response
                        ),
                        reflection_style=self.reflection_style,
                        response_selector=self.response_selector,
                    )

                all_inference_responses.append(inference_response)

        if self.unpack_single_response and len(all_inference_responses) == 1:
            all_inference_responses = all_inference_responses[0]
        return all_inference_responses
