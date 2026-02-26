from collections.abc import Mapping
from typing import Tuple, Any, Union, Sequence

from attr import attrs, attrib

from science_modeling_tools.agents.agent_attachment import AgentAttachment
from science_modeling_tools.agents.agent_actor import AgentActionResult
from science_modeling_tools.agents.agent_response import AgentAction, AgentResponse
from science_modeling_tools.agents.agent_state import AgentTaskStatusFlags, AgentStateItem, AgentStates
from science_modeling_tools.agents.prompt_based_agents.prompt_based_agent import PromptBasedAgent
from rich_python_utils.common_utils import iter_, bool_
from rich_python_utils.common_utils.workflow import cleanup_obj

DEFAULT_RESPONSE_FIELD_NAME_NEXT_ACTIONS = 'ImmediateNextActions'
DEFAULT_RESPONSE_FIELD_NAME_ACTION_GROUP = 'AlternativeActions'
DEFAULT_RESPONSE_FIELD_NAME_ACTION = 'Action'
DEFAULT_RESPONSE_FIELD_NAME_ACTION_TARGET = 'Target'
DEFAULT_RESPONSE_FIELD_NAME_ACTIONS = 'Actions'
DEFAULT_RESPONSE_FIELD_NAME_INSTANT_LEARNINGS = 'InstantLearnings'
DEFAULT_RESPONSE_FIELD_NAME_LEARNING_ID = 'LearningID'
DEFAULT_RESPONSE_FIELD_NAME_LEARNING_CONTENT = 'LearningContent'


@attrs
class PromptBasedActionAgent(PromptBasedAgent):
    enable_action_groups: bool = attrib(default=True)
    response_field_next_actions: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_NEXT_ACTIONS)
    response_field_action_group: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_ACTION_GROUP)
    response_field_action: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_ACTION)
    response_field_action_target: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_ACTION_TARGET)
    response_field_actions: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_ACTIONS)
    response_field_instant_learnings: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_INSTANT_LEARNINGS)
    response_field_learning_id: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_LEARNING_ID)
    response_field_learning_content: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME_LEARNING_CONTENT)

    def __attrs_post_init__(self):
        super(PromptBasedActionAgent, self).__attrs_post_init__()
        if self.response_fields_force_interpreted_as_list is None:
            self.response_fields_force_interpreted_as_list = {
                self.response_field_next_actions, self.response_field_instant_learnings
            }
            self.response_fields_force_interpreted_as_string = {
                self.response_field_learning_content, self.response_field_action_target
            }

    def _parse_instant_response(self, instant_response):
        return instant_response

    def _create_action_item(self, raw_action_item: Mapping):
        return AgentAction(
            reasoning=raw_action_item.get('Reasoning', None),
            type=raw_action_item['Type'],
            target=raw_action_item.get('Target', None),
            is_follow_up=bool_(raw_action_item.get('IsFollowUp', False)),
            memory_target=raw_action_item.get('MemoryTarget', None),
            args=raw_action_item.get('Args', None)
        )

    def _create_next_actions(self, action_items, raw_response_parse: Mapping):
        return action_items

    def _create_agent_response(self, raw_response_parse: Mapping):
        next_actions = []
        for raw_next_action in iter_(raw_response_parse.get(self.response_field_next_actions, None)):
            if self.enable_action_groups:
                if self.response_field_action in raw_next_action:
                    raw_next_action = (raw_next_action[self.response_field_action],)
                elif self.response_field_action_group in raw_next_action:
                    raw_next_action = raw_next_action[self.response_field_action_group]
                    if self.response_field_actions in raw_next_action:
                        raw_next_action = raw_next_action[self.response_field_actions]
                    if isinstance(raw_next_action, dict):
                        raw_next_action = (raw_next_action[self.response_field_action],)
                    else:
                        raw_next_action = tuple(
                            action_item[self.response_field_action] for action_item in raw_next_action
                        )

                next_actions.append(
                    tuple(
                        self._create_action_item(raw_action_item)
                        for raw_action_item in raw_next_action
                    )
                )
            else:
                if self.response_field_action in raw_next_action:
                    raw_next_action = raw_next_action[self.response_field_action]
                    next_actions.append(self._create_action_item(raw_next_action))

        next_actions = self._create_next_actions(next_actions, raw_response_parse)
        instant_learnings = {}
        for raw_instant_learning in iter_(raw_response_parse.get(self.response_field_instant_learnings, None)):
            if isinstance(raw_instant_learning, Mapping):
                raw_instant_learning = next(iter(raw_instant_learning.values()))
                instant_learnings[
                    raw_instant_learning[self.response_field_learning_id]
                ] = raw_instant_learning[self.response_field_learning_content]
        if not instant_learnings:
            instant_learnings = None

        from science_modeling_tools.agents.prompt_based_agents.prompt_based_response_agent import \
            PromptBasedResponseActionAgent
        if isinstance(self, PromptBasedResponseActionAgent):
            pass
        instant_response = self._parse_instant_response(
                raw_response_parse.get(self.response_field_instant_response, None)
        )

        return AgentResponse(
            instant_response=instant_response,
            instant_learnings=instant_learnings,
            next_actions=next_actions,
        )

    def _create_agent_state(self, raw_response_parse: Mapping):
        new_task_flag = bool_(raw_response_parse.get(self.response_field_new_task_flag, False))
        task_status_flag = AgentTaskStatusFlags(raw_response_parse[self.response_field_task_status_flag])
        task_status_description = raw_response_parse.get(self.response_field_task_status_description, None)
        return AgentStateItem(
            new_task=new_task_flag,
            task_status=task_status_flag,
            task_status_description=task_status_description,
            task_status_description_extended=task_status_description
        )

    def _extract_from_raw_response_parse(self, raw_response_parse: Mapping) -> Tuple[
        Union[str, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        agent_response = self._create_agent_response(raw_response_parse)
        agent_state = self._create_agent_state(raw_response_parse)

        return agent_response, agent_state

    def _get_agent_results(self, trigger_action, trigger_action_results, new_states):
        if new_states:
            agent_results = []
            for new_state in new_states:
                if isinstance(new_state, AgentStateItem):
                    agent_response = new_state.response
                    if isinstance(agent_response, AgentResponse) and agent_response.instant_learnings:
                        for instant_learning_item in agent_response.instant_learnings.values():
                            agent_result = AgentActionResult(
                                summary=instant_learning_item,
                                action=new_state.last_action_type,
                                anchor_action=new_state.last_anchor_action_type,
                                source=new_state.last_action_source,  # the action source when
                                task_label=new_state.task_label,
                                last_action_response=agent_response,
                            )
                            agent_results.append(agent_result)
                        agent_results.extend(agent_response.instant_learnings.values())

                    if isinstance(new_state.action_results, AgentActionResult):
                        agent_result = new_state.action_results
                        agent_result.task_label = new_state.task_label
                        agent_result.last_action_response=agent_response
                        agent_results.append(agent_result)
            return agent_results

    def _make_attachments(self, base_obj) -> Sequence[AgentAttachment]:
        """
        Create attachments from previous agent results.

        Args:
            base_obj: AgentActionResult object(s) from previous agent execution

        Returns:
            Sequence of Attachment objects with task context
        """
        attachments = []

        # Handle both single item and list of items
        for agent_result in iter_(base_obj):
            if isinstance(agent_result, AgentActionResult):
                # Extract task_label for ID
                attachment_id = agent_result.task_label or "unknown_task"

                # Extract instant_response from last_action_response
                instant_response = ""
                if agent_result.last_action_response:
                    if isinstance(agent_result.last_action_response, AgentResponse):
                        instant_response = agent_result.last_action_response.instant_response or ""

                # Format description
                description = f"The result of the '{attachment_id}' agent."

                # Create attachment
                attachments.append(AgentAttachment(
                    id=attachment_id,
                    description=description,
                    content=agent_result
                ))

        return attachments

    def close(self):
        """
        Clean up resources used by this agent, including closing actors.

        This method iterates through all actors and attempts to clean them up
        by calling their cleanup methods (quit, close, exit) or using del if __del__ is defined.
        """
        # First call parent's close method
        super().close()

        # Clean up actors
        if self.actor:
            if isinstance(self.actor, dict):
                # Actor is a dictionary - iterate through all actors
                for actor_key, actor_instance in list(self.actor.items()):
                    # Get repr before cleanup
                    try:
                        actor_repr = type(actor_instance)
                    except Exception:
                        actor_repr = f"<{type(actor_instance).__name__} object>"

                    if cleanup_obj(actor_instance):
                        print(f"[Agent] Successfully cleaned up actor '{actor_key}': {actor_repr}")
                    else:
                        print(f"[Agent] Warning: Could not cleanup actor '{actor_key}': {actor_repr}")
            else:
                # Actor is a single instance
                # Get repr before cleanup
                try:
                    actor_repr = type(self.actor)
                except Exception:
                    actor_repr = f"<{type(self.actor).__name__} object>"

                if cleanup_obj(self.actor):
                    print(f"[Agent] Successfully cleaned up actor: {actor_repr}")
                else:
                    print(f"[Agent] Warning: Could not cleanup actor: {actor_repr}")
