import json
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import Tuple, Any, Union, Dict, Mapping, Iterable, Optional, Sequence

from attr import attrs, attrib

from agent_foundation.agents.agent import Agent
from agent_foundation.agents.agent_response import AgentAction, AgentResponse, AgentResponseFormat
from agent_foundation.agents.agent_state import AgentTaskStatusFlags, AgentStateItem
from agent_foundation.agents.prompt_based_agents.constants import DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_TASK_INPUT, \
    DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_USER_INPUT, DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CONVERSATIONAL_INPUT, \
    DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_USER_PROFILE, DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CONTEXT, \
    DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ACTION_RESULT, DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ACTION_MEMORY, \
    DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CURRENT_STATE, DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_PREVIOUS_STATES, \
    DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ATTACHMENTS
from agent_foundation.common.inferencers.agentic_inferencers.common import InferencerResponse
from rich_python_utils.common_utils import iter_, get_relevant_named_args, get_
from rich_python_utils.string_utils import join_, extract_between, add_prefix
from rich_python_utils.string_utils.formatting.common import format_key_value, KeyValueStringFormat
from rich_python_utils.string_utils.formatting.template_manager import TemplateManager
from rich_python_utils.string_utils.xml_helpers import mapping_to_xml, xml_to_dict

DEFAULT_USER_TURN_IDENTIFYING_STRING = 'User'
DEFAULT_AGENT_TURN_IDENTIFYING_STRING = 'Agent'

DEFAULT_RESPONSE_FIELD_INSTANT_RESPONSE = 'InstantResponse'
DEFAULT_RESPONSE_FIELD_NEW_TASK_FLAG = 'NewTask'
DEFAULT_RESPONSE_FIELD_TASK_STATUS_FLAG = 'TaskStatus'
DEFAULT_RESPONSE_FIELD_TASK_STATUS_DESCRIPTION = 'TaskStatusDescription'


class FeedConflictResolution(StrEnum):
    """Strategy for resolving conflicts when knowledge dict keys overlap with prompt feed keys."""
    ATTRIBUTE_ONLY = 'attribute_only'
    FEED_ONLY = 'feed_only'
    MERGE = 'merge'


@attrs
class PromptBasedAgent(Agent):
    """
    A prompt-driven interactive agent that generates responses based on customizable prompt templates.
    This class extends `Agent` by formatting user input, profile, and context into a structured prompt
    template before passing it to the reasoning function.

    `PromptBasedAgent` is designed for interactions requiring flexible, template-based prompts,
    enabling different templates based on the agent's state and customizable formatting through an
    optional `prompt_formatter`.

    Attributes:
        default_prompt_template (str): The fallback template used for generating prompts when no specific
            template is available for the current agent state.
        prompt_templates (Dict[Any, str]): A dictionary of templates mapped to specific agent states,
            allowing different templates for each state. Defaults to `None`.
        prompt_formatter (Callable): Optional function for formatting the prompt template with the provided
            feed data. If `None`, the default Python string `format()` is used.

    Methods:
        _construct_prompt_feed(user_input, user_profile, context, action_result) -> Dict[str, str]:
            Generates a dictionary containing the user input, user profile, context, and action result.
            This dictionary is used as input data for the prompt template.

        _construct_reasoner_input(user_input, user_profile, context, action_result, agent_state) -> Any:
            Constructs the input for the reasoning function by formatting the prompt template with the
            generated feed data. If `prompt_formatter` is specified, it is used for formatting; otherwise,
            Python’s default `str.format()` is applied.
    """
    default_prompt_template: str = attrib(default='')
    prompt_templates: Union[str, Dict[Any, str]] = attrib(default=None)
    prompt_formatter: Callable = attrib(default=None)
    prompt_template_version: str = attrib(default="")
    input_string_formatter: Union[str, KeyValueStringFormat, Callable[[str], str]] = attrib(default=None)
    response_string_formatter: Union[str, KeyValueStringFormat, Callable[[str], str]] = attrib(default=None)
    direct_response_start_delimiter: str = attrib(default=None)
    direct_response_end_delimiter: str = attrib(default=None)
    raw_response_start_delimiter: str = attrib(default=None)
    raw_response_end_delimiter: str = attrib(default=None)
    raw_response_format: AgentResponseFormat = attrib(default=AgentResponseFormat.Other)
    raw_response_parsing_args: Mapping[str, Any] = attrib(default=None)

    prompt_placeholder_user_input: str = attrib(default=None)
    use_conversational_user_input: bool = attrib(default=False)
    user_turn_identifying_string: str = attrib(default=DEFAULT_USER_TURN_IDENTIFYING_STRING)
    agent_turn_identifying_string: str = attrib(default=DEFAULT_AGENT_TURN_IDENTIFYING_STRING)
    prompt_placeholder_task_input: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_TASK_INPUT)
    prompt_placeholder_user_profile: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_USER_PROFILE)
    prompt_placeholder_context: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CONTEXT)
    prompt_placeholder_action_result: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ACTION_RESULT)
    prompt_placeholder_action_memory: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ACTION_MEMORY)
    prompt_placeholder_current_state: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CURRENT_STATE)
    prompt_placeholder_previous_states: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_PREVIOUS_STATES)
    prompt_placeholder_attachments: str = attrib(default=DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_ATTACHMENTS)

    response_field_instant_response: str = attrib(default=DEFAULT_RESPONSE_FIELD_INSTANT_RESPONSE)
    response_field_new_task_flag: str = attrib(default=DEFAULT_RESPONSE_FIELD_NEW_TASK_FLAG)
    response_field_task_status_flag: str = attrib(default=DEFAULT_RESPONSE_FIELD_TASK_STATUS_FLAG)
    response_field_task_status_description: str = attrib(default=DEFAULT_RESPONSE_FIELD_TASK_STATUS_DESCRIPTION)
    response_fields_force_interpreted_as_list : Iterable[str] = attrib(default=None)
    response_fields_force_interpreted_as_string : Iterable[str] = attrib(default=None)
    feed_conflict_resolution: FeedConflictResolution = attrib(default=FeedConflictResolution.FEED_ONLY)

    def __attrs_post_init__(self):
        super(PromptBasedAgent, self).__attrs_post_init__()

        # region STEP1: processes prompt templates
        if not isinstance(self.prompt_formatter, TemplateManager):
            self.prompt_formatter = TemplateManager(
                default_template=self.default_prompt_template,
                templates=self.prompt_templates,
                template_formatter=self.prompt_formatter,
                template_version=self.prompt_template_version
            )

        # endregion

        # region STEP2: processes input/response string formatter
        if not self.input_string_formatter:
            self.input_string_formatter = partial(
                format_key_value,
                key=self.user_turn_identifying_string
            )
        elif isinstance(self.input_string_formatter, KeyValueStringFormat):
            self.input_string_formatter = partial(
                format_key_value,
                key=self.user_turn_identifying_string,
                format_type=self.input_string_formatter
            )

        if not isinstance(self.input_string_formatter, (str, Callable)):
            raise ValueError("'input_string_formatter' must be a string template or a callable")

        if not self.response_string_formatter:
            self.response_string_formatter = partial(format_key_value, key=self.agent_turn_identifying_string)
        elif isinstance(self.response_string_formatter, KeyValueStringFormat):
            self.response_string_formatter = partial(
                format_key_value,
                key=self.agent_turn_identifying_string,
                format_type=self.response_string_formatter
            )

        if not isinstance(self.response_string_formatter, (str, Callable)):
            raise ValueError("'response_string_formatter' must be a string template or a callable")
        # endregion

        # region STEP3: assigns default prompt placeholders
        if not self.prompt_placeholder_user_input:
            self.prompt_placeholder_user_input = (
                DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CONVERSATIONAL_INPUT
                if self.use_conversational_user_input
                else DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_USER_INPUT
            )

        if not self.prompt_placeholder_user_profile:
            self.prompt_placeholder_user_profile = DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_USER_PROFILE

        if not self.prompt_placeholder_context:
            self.prompt_placeholder_context = DEFAULT_PROMPT_TEMPLATE_PLACEHOLDER_CONTEXT
        # endregion

        # region STEP4: handles other arguments
        if not self.raw_response_parsing_args:
            self.raw_response_parsing_args = {}

        # endregion

    # region Prompt Construction Methods
    def _format_input_string(self, input_string: str) -> str:
        if isinstance(self.input_string_formatter, str):
            return self.input_string_formatter.format(input_string)
        else:
            return '\n'.join(
                self.input_string_formatter(_input_string)
                for _input_string in iter_(input_string)
            )

    def _format_response_string(self, response_string: str) -> str:
        if isinstance(self.response_string_formatter, str):
            return self.response_string_formatter.format(response_string)
        else:
            return '\n'.join(
                self.response_string_formatter(_response_string)
                for _response_string in iter_(response_string)
            )

    def _get_user_input_string(self, user_input: Any, conversational: bool) -> str:
        if conversational:
            conversation = []
            for agent_state in self.states:
                if isinstance(agent_state, AgentStateItem):
                    for _user_input in iter_(agent_state.user_input, non_atom_types=(list, tuple)):
                        conversation.append(self._format_input_string(_user_input))
                    agent_response = agent_state.response.instant_response
                    if agent_response:
                        conversation.append(self._format_response_string(agent_response))
            if user_input:
                conversation.append(self._format_input_string(user_input))
            return join_(conversation, sep='\n')
        elif user_input:
            return self._format_input_string(user_input)

    def _get_user_profile_string(self, user_profile: Any) -> str:
        if isinstance(user_profile, Dict):
            return json.dumps(user_profile)
        else:
            return str(user_profile)

    def _get_context_string(self, context: Any) -> str:
        if isinstance(context, Dict):
            return mapping_to_xml(context, unescape=True)
        else:
            return str(context)

    def _get_action_result_string(self, action_results: Any) -> str:
        _action_results = []
        for _action_result in iter_(action_results):
            if isinstance(_action_result, AgentAction):
                _action_results.append(
                    {
                        'ActionType': _action_result.type,
                        'ActionResult': str(_action_result.result)
                    }
                )
            else:
                _action_results.append({'ActionResult': str(_action_result)})
        return mapping_to_xml(_action_results, root_tag='ActionResults', include_root=True, unescape=True)

    def _get_action_memory_string(self, action_results: Any) -> Optional[str]:
        if isinstance(action_results, AgentAction):
            action_results = action_results.result

        action_memory = get_(get_(action_results, 'action_memory'), 'memory')
        # TODO: introduce some genric formatting util
        if action_memory:
            return action_memory

        return None

    def _get_state_strings(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.states:
            current_state_string, previous_stats_string = None, None
        else:
            i = len(self.states) - 1
            while i >= 0:
                state = self.states[i]
                # Skip None states (e.g., from PromptBasedActionPlanningAgent)
                if state is not None:
                    if str(state.task_status) != AgentTaskStatusFlags.Pending:
                        break
                i -= 1

            # Handle case where all states are None or Pending
            if i < 0:
                return None, None

            state = self.states[i]
            current_state_string = add_prefix(
                state.task_status_description_extended,
                prefix=state.task_status_description,
                sep='\n',
                avoid_repeat=True
            )

            if i == 0:
                previous_stats_string = None
            else:
                i -= 1
                previous_stats_string = []
                while i >= 0:
                    state = self.states[i]
                    # Skip None states (e.g., from PromptBasedActionPlanningAgent)
                    if state is not None:
                        if str(state.task_status) != AgentTaskStatusFlags.Pending:
                            previous_stats_string.append(
                                add_prefix(
                                    state.task_status_description_extended,
                                    prefix=state.task_status_description,
                                    sep='\n',
                                    avoid_repeat=True
                                )
                            )
                    i -= 1
                if previous_stats_string:
                    previous_stats_string = join_(*previous_stats_string, sep='\n\n', ignore_none_or_empty=True)
                else:
                    previous_stats_string = None

        return current_state_string, previous_stats_string

    def _construct_prompt_feed(
            self,
            task_input: Any,
            user_input: Any,
            user_profile: Any = None,
            context: Any = None,
            action_results: Any = None,
            attachments: Sequence[Any] = None
    ) -> Dict[str, str]:
        """
        Generates a dictionary with user input, user profile, context, and action result for prompt formatting.

        Args:
            user_input (Any): The raw input from the user.
            user_profile (Any): Data related to the user's profile, if available.
            context (Any): Contextual information relevant to the interaction.
            action_results (Any): Result from a previous action, if any.

        Returns:
            Dict[str, str]: A dictionary with keys 'user_input', 'user_profile', 'context', and 'action_result'
            containing string representations of the corresponding data.
        """
        feed = {
            self.prompt_placeholder_user_input:
                self._get_user_input_string(
                    user_input, conversational=self.use_conversational_user_input
                )
        }

        if isinstance(task_input, Mapping):
            feed.update(task_input)
        else:
            feed[self.prompt_placeholder_task_input] = task_input

        if user_profile:
            feed[self.prompt_placeholder_user_profile] = self._get_user_profile_string(user_profile)

        if context:
            feed[self.prompt_placeholder_context] = self._get_context_string(context)

        if action_results:
            try:
                feed[self.prompt_placeholder_action_result] = self._get_action_result_string(action_results)
            except Exception as e:
                print(e)
            action_memory = self._get_action_memory_string(action_results)
            if action_memory:
                feed[self.prompt_placeholder_action_memory] = action_memory

        current_state_string, previous_stats_string = self._get_state_strings()
        if current_state_string:
            feed[self.prompt_placeholder_current_state] = current_state_string
        if previous_stats_string:
            feed[self.prompt_placeholder_previous_states] = previous_stats_string

        if attachments:
            feed[self.prompt_placeholder_attachments] = '\n'.join((str(attachment) for attachment in attachments))

        return feed

    def _construct_reasoner_input(
            self,
            task_input: Any,
            user_input: Any,
            user_profile: Any = None,
            context: Any = None,
            action_results: Any = None,
            attachments: Sequence[Any] = None,
            knowledge: Dict[str, str] = None
    ) -> Any:
        """
        Constructs the formatted input for the reasoning function by applying the resolved prompt template
        to the generated prompt feed.

        This method builds the final input for the reasoning function using the resolved prompt template
        and a feed dictionary. If a `prompt_formatter` is specified, it is used to format the template with
        the feed; otherwise, `str.format()` is used.

        Args:
            task_input (Any): Task-level metadata or input context.
            user_input (Any): The raw input from the user.
            user_profile (Any): Data related to the user's profile, if available.
            context (Any): Contextual information relevant to the interaction.
            action_results (Any): Result from a previous action, if any.
            attachments (Sequence[Any]): Optional attachments to include.
            knowledge (Dict[str, str]): Resolved knowledge dict from knowledge_provider.
                Merged into the prompt feed using the feed_conflict_resolution strategy.

        Returns:
            Any: The formatted input ready for processing by the reasoning function.
        """
        feed = self._construct_prompt_feed(
            task_input=task_input,
            user_input=user_input,
            user_profile=user_profile,
            context=context,
            action_results=action_results,
            attachments=attachments
        )

        # Merge knowledge dict into feed using the configured conflict resolution strategy
        if knowledge:
            self._merge_into_feed(feed, knowledge)

        # Resolve additional_reasoner_input_feed (callables and static values)
        prompt_key = self.states.last_anchor_action_type
        resolved_extra = {}
        for k, v in self.additional_reasoner_input_feed.items():
            if callable(v):
                result = v(user_input)
                if isinstance(result, dict):
                    resolved_extra.update(result)
                else:
                    resolved_extra[k] = result
            else:
                resolved_extra[k] = v

        if resolved_extra:
            self._merge_into_feed(feed, resolved_extra)

        return self.prompt_formatter(prompt_key, feed=feed)

    def _merge_into_feed(self, feed: dict, extra: dict):
        """Merge extra dict into feed using the configured feed_conflict_resolution strategy.

        Args:
            feed: The base prompt feed dict (modified in place).
            extra: The dict to merge in (from knowledge or additional_reasoner_input_feed).
        """
        for k, v in extra.items():
            if k in feed and feed[k]:
                if self.feed_conflict_resolution == FeedConflictResolution.ATTRIBUTE_ONLY:
                    pass  # keep existing value
                elif self.feed_conflict_resolution == FeedConflictResolution.MERGE:
                    feed[k] = f"{feed[k]}\n\n{v}"
                else:  # FEED_ONLY (default)
                    feed[k] = v
            else:
                feed[k] = v

    def _construct_reasoner_inference_config(self):
        if isinstance(self.prompt_formatter, TemplateManager):
            config = {
                TemplateManager.ARG_NAME_ACTIVE_TEMPLATE_ROOT_SPACE: self.prompt_formatter.active_template_root_space
            }
            if self.states.last_anchor_action_type:
                config[TemplateManager.ARG_NAME_TEMPLATE_KEY] = self.states.last_anchor_action_type
            return config

    # endregion

    # region Response Parsing Methods
    def _extract_from_raw_response_parse(self, raw_response_parse: Mapping) -> Tuple[
        Union[str, Mapping, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        return raw_response_parse, AgentTaskStatusFlags.Completed

    def _parse_raw_response(self, raw_response: Union[str, InferencerResponse, Any]) -> Tuple[
        Union[str, Mapping, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        if isinstance(raw_response, InferencerResponse):
            raw_response_string = raw_response.select_response().response
        elif not isinstance(raw_response, str):
            raw_response_string = str(raw_response)
        else:
            raw_response_string = raw_response

        raw_response_string, matching_search1_index = extract_between(
            raw_response_string,
            search1=(self.direct_response_start_delimiter, self.raw_response_start_delimiter),
            search2=(self.direct_response_end_delimiter, self.raw_response_end_delimiter),
            keep_search1=False,
            keep_search2=False,
            allow_search1_not_found=False,
            allow_search2_not_found=False,
            return_matching_search1_index=True,
            search1_use_last_occurrence=True,
            search2_use_last_occurrence=True
        )

        if matching_search1_index == 0:
            return raw_response_string.strip(), None
        else:
            try:
                if self.raw_response_format == AgentResponseFormat.XML:
                    raw_response_parse = xml_to_dict(
                        raw_response_string,
                        allows_xml_lines_without_root=True,
                        merge_same_tag_elements_as_list=False,
                        always_interpret_children_as_list=self.response_fields_force_interpreted_as_list,
                        always_interpret_children_as_string=self.response_fields_force_interpreted_as_string,
                        use_lxml_parser=False,
                        lenient_parsing=True,
                        **get_relevant_named_args(xml_to_dict, **self.raw_response_parsing_args)
                    )
                elif self.raw_response_format == AgentResponseFormat.JSON:
                    raw_response_parse = json.loads(
                        raw_response_string,
                        **get_relevant_named_args(json.loads, **self.raw_response_parsing_args)
                    )
                else:
                    raw_response_parse = raw_response_string.strip()
            except Exception as err:
                if isinstance(raw_response, InferencerResponse) or hasattr(raw_response, 'base_response'):
                    # The inferencer sometimes does not want to change the base response, and produced invalid format;
                    # In this case, the base response is good.
                    self.log_debug(
                        "Unable to parse the default string representation of the raw response. "
                        "Try parsing its base response instead."
                    )
                    return self._parse_raw_response(raw_response.base_response)
                else:
                    raise err

            agent_response, agent_state = self._extract_from_raw_response_parse(raw_response_parse)
            if isinstance(agent_response, AgentResponse) or hasattr(agent_response, 'raw_response'):
                agent_response.raw_response = raw_response_string
            return agent_response, agent_state
    # endregion
